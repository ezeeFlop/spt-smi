import base64
import io
from pathlib import Path
from spt.models.image import TextToImageResponse
from spt.services.service import Worker, Service
from spt.utils import create_temp_file, remove_temp_file, get_available_device
from spt.models.workers import WorkerBaseRequest
from pydantic import BaseModel
from typing import Union, Dict, Any
import torch
from diffusers import FluxPipeline
import gc

class Flux(Worker):
    def __init__(self, id:str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.my_model = None
        self.pipe = None
        self.generator = None
        self.lora_dir = Path("/home/spt/.cache/lora")

    def __del__(self):
        self.logger.warning("Claiming memory")
        if self.pipe is not None:
            self.close_diffusion_pipe()
            del self.pipe
        if self.generator is not None:
            del self.generator
        torch.cuda.empty_cache()
        gc.collect()

    def close_diffusion_pipe(self):
        self.pipe = None
        self.generator = None
        torch.cuda.empty_cache()

    def get_diffusion_pipe(self):
        if self.pipe is None:
            pipe = None
            generator = None

            if torch.backends.mps.is_available():
                self.logger.warning("MPS is available")
                pipe = FluxPipeline.from_pretrained(
                    self.model,

                )
                pipe = pipe.to("mps")
                self.num_inference_steps = 30
                generator = torch.Generator(device='mps')

            elif torch.cuda.is_available():
                self.logger.warning("CUDA is available")

                pipe = FluxPipeline.from_pretrained(
                    self.model,
                    torch_dtype=torch.bfloat16,
                    device_map='balanced'
                )
                self.num_inference_steps = 50
                pipe.load_lora_weights("SamFloppy/OurSelfves", weight_name="cve.safetensors")
                pipe.fuse_lora(lora_scale=1.0)
                #torch.backends.cuda.matmul.allow_tf32 = True
                #pipe = pipe.to("cuda:0")
                #pipe.enable_model_cpu_offload()
                generator = torch.Generator(device='cpu')

            else:
                self.logger.warning("CUDA is **not** available")
                pipe = FluxPipeline.from_pretrained(
                    self.model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
                )
                pipe = pipe.to("cpu")
                generator = torch.Generator(device='cpu')

                self.num_inference_steps = 5

            pipe.enable_attention_slicing()
            pipe.safety_checker = None

            self.pipe = pipe
            self.generator = generator

    async def work(self, request: WorkerBaseRequest) -> BaseModel:
        await super().work(request)
        self.logger.warning("Starting work")
        if self.pipe == None:
            self.get_diffusion_pipe()

        if request.seed is not None:
            self.generator.manual_seed(request.seed)

        prompts = list(request.text_prompts)
        self.logger.warning(f"Prompts: {prompts}")
        images = []
        for prompt in prompts:
            image = self.pipe(
                prompt.text,
                width=512,
                height=512,
                guidance_scale      = 7,
                output_type         = "pil",
                num_inference_steps = request.steps,
                max_sequence_length = 512,
                generator           = self.generator
            ).images[0]
            tampon_bytes = io.BytesIO()
            image.save(tampon_bytes, format='PNG')

            bytes_image = tampon_bytes.getvalue()

            if self.service.should_store():
                self.logger.warning(f"Storing image...")
                url = self.service.store_bytes(
                    bytes=bytes_image, name=prompt.text, extension="png")
                images.append({"url": url,
                              "seed": 42, "finishReason": "SUCCESS"})
            else:
                image_base64 = base64.b64encode(bytes_image)
                images.append(
                    {"base64": image_base64, "seed": request.seed, "finishReason": "SUCCESS"})
        
        self.logger.warning("Ending work")
        self.cleanup() # Force cleanup
        return TextToImageResponse(artifacts=images)


    async def stream(self, data: Union[bytes | str | Dict[str, Any]]) -> Union[bytes | str | Dict[str, Any]]:
        # do something with data

        return data

    def cleanup(self):
        super().cleanup()
        self.logger.warning("Cleaning up...")
        if self.my_model is not None:
            self.logger.info(f"Closing model {self.my_model}")
            del self.my_model
            self.my_model = None
        if self.pipe is not None:
            self.close_diffusion_pipe()
            del self.pipe
            self.pipe = None
        if self.generator is not None:
            del self.generator
            self.generator = None
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.warning("Cleanup completed")