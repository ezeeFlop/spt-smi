import asyncio
import base64
import gc
import io
import os
import subprocess
import requests
from pathlib import Path
from spt.services.service import Worker, Service
from spt.models.image import TextToImageResponse, TextToImageRequest
from stable_diffusion_cpp import StableDiffusion

class FluxCpp(Worker):
    def __init__(self, id: str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.sd_binary_path = "/sd"
        self.num_inference_steps = 20
        self.cache_dir = Path("/home/spt/.cache/flux")
        if not self.cache_dir.exists():
            self.cache_dir = Path("/tmp")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.pipe = None
        self.flux_files = {
            "flux1-dev-q3_k": {
                "flux1-dev-q3_k.gguf": "https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/flux1-dev-q3_k.gguf",
                "ae.safetensors": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
            },
            "flux1-schnell-q3_k": {
                "flux1-schnell-q3_k.gguf": "https://huggingface.co/leejet/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-q3_k.gguf",
                "ae.safetensors": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
            },
            "flux1-schnell-q8_0": {
                "flux1-schnell-q8_0.gguf": "https://huggingface.co/leejet/FLUX.1-schnell-q8-gguf/resolve/main/flux1-schnell-q8_0.gguf",
                "ae.safetensors": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
            },
            "flux1-dev-q8_0": {
                "flux1-dev-q8_0.gguf": "https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/flux1-dev-q8_0.gguf",
                "ae.safetensors": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
            },
        }

    def download_file(self, url, filename):
        filepath = self.cache_dir / filename
        if not filepath.exists():
            self.logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f"Downloaded {filename}")
        else:
            self.logger.info(f"{filename} already exists, skipping download")
        return str(filepath)

    def ensure_flux_models(self):
        if self.model not in self.flux_files:
            raise ValueError(f"Unknown model: {self.model}")

        model_paths = {}
        for filename, url in self.flux_files[self.model].items():
            model_paths[filename] = self.download_file(url, filename)
        return model_paths

    async def work(self, request: TextToImageRequest) -> TextToImageResponse:
        await super().work(request)

        self.logger.info(f"Generate Image with {request}")

        # Ensure FLUX models are downloaded
        model_paths = self.ensure_flux_models()

        seed = request.seed if request.seed is not None else 42
        prompts = list(request.text_prompts)
        images = []

        if self.pipe is None:
            self.pipe = StableDiffusion(diffusion_model_path=model_paths[f"{self.model}.gguf"], 
                                        vae_path=model_paths["ae.safetensors"],
                                        clip_l_path=model_paths["clip_l.safetensors"],
                                        t5xxl_path=model_paths["t5xxl_fp16.safetensors"],
                                        wtype="default")


        for prompt in prompts:
            image = self.pipe.txt_to_img(prompt=prompt.text, 
                                       seed=seed, 
                                       sample_steps=self.num_inference_steps,
                                       sample_method="euler",
                                       cfg_scale=1.0,
                                       )
            tampon_bytes = io.BytesIO()
            image[0].save(tampon_bytes, format='PNG')
            bytes_image = tampon_bytes.getvalue()

            if self.service.should_store():
                url = self.service.store_bytes(
                    bytes=bytes_image, name=prompt.text, extension="png")
                images.append({"url": url, "seed": seed, "finishReason": "SUCCESS"})
            else:
                image_base64 = base64.b64encode(bytes_image).decode('utf-8')
                images.append(
                    {"base64": image_base64, "seed": seed, "finishReason": "SUCCESS"})

        return TextToImageResponse(artifacts=images)

    def cleanup(self):
        super().cleanup()
        if self.pipe is not None:
            del self.pipe
        gc.collect()
        # No need to clean up CUDA memory as we're using a separate process
