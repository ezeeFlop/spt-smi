import base64
import gc
import io
import tempfile
from spt.services.service import Worker, Service
from spt.utils import create_temp_file, remove_temp_file, get_available_device
from spt.models.workers import WorkerBaseRequest
from spt.models.video import TextToVideoRequest, TextToVideoResponse
from pydantic import BaseModel
from typing import Union, Dict, Any
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging

import imageio
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod
from huggingface_hub import snapshot_download

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

def load_vae(vae_dir):
    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
    vae_config_path = vae_dir / "config.json"
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    if torch.cuda.is_available():
        vae = vae.cuda()
    return vae.to(torch.bfloat16)


def load_unet(unet_dir):
    unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
    unet_config_path = unet_dir / "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=True)
    transformer = transformer.to(torch.bfloat16)
    if torch.cuda.is_available():
        transformer = transformer.cuda()
    return transformer


def load_scheduler(scheduler_dir):
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)


def load_image_to_tensor_with_resize_and_crop(
    image_path, target_height=512, target_width=768
):
    image = Image.open(image_path).convert("RGB")
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)


# Generate output video name
def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )

def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class LTX(Worker):
    def __init__(self, id:str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.pipeline = None
        self.generator = None
        self.cache_dir = Path("/home/spt/.cache")
    
        if not self.cache_dir.exists():
            self.cache_dir = Path("/tmp")

        if not (self.cache_dir / "ltx-video-2b-v0.9.safetensors").exists():
            snapshot_download("Lightricks/LTX-Video", local_dir=self.cache_dir, local_dir_use_symlinks=False, repo_type='model')

    async def work(self, request: TextToVideoRequest) -> TextToVideoResponse:
        await super().work(request)

        seed_everething(request.seed)

        # Handle input dimensions
        height = request.height if request.height else MAX_HEIGHT
        width = request.width if request.width else MAX_WIDTH
        num_frames = request.num_frames

        if height > MAX_HEIGHT or width > MAX_WIDTH or num_frames > MAX_NUM_FRAMES:
            self.logger.warning(
                f"Input resolution or number of frames {height}x{width}x{num_frames} is too big, it is suggested to use the resolution below {MAX_HEIGHT}x{MAX_WIDTH}x{MAX_NUM_FRAMES}."
            )

        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

        padding = calculate_padding(height, width, height_padded, width_padded)

        self.logger.warning(
            f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
        )

        # Paths for the separate model directories
        ckpt_dir = self.cache_dir
        unet_dir = ckpt_dir / "unet"
        vae_dir = ckpt_dir / "vae"
        scheduler_dir = ckpt_dir / "scheduler"

        # Load models
        vae = load_vae(vae_dir)
        unet = load_unet(unet_dir)
        scheduler = load_scheduler(scheduler_dir)
        patchifier = SymmetricPatchifier(patch_size=1)
        text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
        ).to(torch.bfloat16)

        if torch.cuda.is_available():
            text_encoder = text_encoder.to("cuda")
        tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
        )

        # Use submodels for the pipeline
        submodel_dict = {
            "transformer": unet,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
        }

        self.pipeline = LTXVideoPipeline(**submodel_dict)
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")

        # Prepare input for the pipeline
        sample = {
            "prompt": request.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": request.negative_prompt or "worst quality, inconsistent motion, blurry, jittery, distorted",
            "negative_prompt_attention_mask": None,
            "media_items": None,
        }

        self.generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(request.seed)

        images = self.pipeline(
            num_inference_steps=request.steps,
            num_images_per_prompt=1,
            guidance_scale=request.guidance_scale,
            generator=self.generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=request.frame_rate,
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=ConditioningMethod.UNCONDITIONAL,
            mixed_precision=True,
            load_needed_only=True
        ).images

        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom
        pad_right = -pad_right
        if pad_bottom == 0:
            pad_bottom = images.shape[3]
        if pad_right == 0:
            pad_right = images.shape[4]

        images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]
        videos = []

        for i in range(images.shape[0]):
            # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
            video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
            # Unnormalizing images to [0, 255] range
            video_np = (video_np * 255).astype(np.uint8)
            fps = request.frame_rate

            # Create temporary file with specific codec and format
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            try:
                with imageio.get_writer(temp_file.name, fps=fps, codec='libx264', format='FFMPEG', mode='I', output_params=['-pix_fmt', 'yuv420p']) as video:
                    for frame in video_np:
                        video.append_data(frame)
                
                # Read the temporary file
                with open(temp_file.name, 'rb') as f:
                    video_data = f.read()

                # Process the video data
                if self.service.should_store():
                    url = self.service.store_bytes(
                        bytes=video_data, name=request.prompt, extension="mp4")
                    videos.append({"url": url,
                                "seed": request.seed, "finishReason": "SUCCESS"})
                else:
                    video_base64 = base64.b64encode(video_data)
                    videos.append(
                        {"base64": video_base64, "seed": request.seed, "finishReason": "SUCCESS"})
            finally:
                # Clean up the temporary file
                temp_file.close()
                os.unlink(temp_file.name)

        self.logger.warning(f"Output saved to {self.cache_dir}")
        # Clean up all resources
        del images
        del video_np
        del unet
        del vae
        del scheduler
        del patchifier
        del text_encoder
        del tokenizer
        del submodel_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cleanup()
    
        return TextToVideoResponse(artifacts=videos)

    async def stream(self, data: Union[bytes | str | Dict[str, Any]]) -> Union[bytes | str | Dict[str, Any]]:
        # do something with data

        return data

    def cleanup(self):
        super().cleanup()
        if self.pipeline is not None:
            self.logger.info(f"Closing pipeline {self.pipeline}")
            del self.pipeline
        if self.generator is not None:
            self.logger.info(f"Closing generator {self.generator}")
            del self.generator
        gc.collect()
        torch.cuda.empty_cache()