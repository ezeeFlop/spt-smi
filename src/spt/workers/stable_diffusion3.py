from spt.services.service import Worker, Service
import gc
from diffusers import StableDiffusion3Pipeline
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from spt.models.image import TextToImageResponse, TextToImageRequest
from spt.utils import get_available_device, get_cuda_available_device
import base64
import PIL
import io
from spt.services.service import Service
from spt.storage import Storage

class StableDiffusion3(Worker):

    def close_diffusion_pipe(self):
        self.pipe = None
        torch.cuda.empty_cache()   
        gc.collect()

    @classmethod
    def memory_usage(cls):
        max_memory = round(torch.cuda.max_memory_allocated(
            device= get_cuda_available_device()) / 1000000000, 2)
        return max_memory

    def get_diffusion_pipe(self):
        if self.pipe is None:
            pipe = None

            if torch.backends.mps.is_available():
                self.logger.info("MPS is available")
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    self.model,
                )
                pipe = pipe.to("mps")
                self.num_inference_steps = 30

            elif torch.cuda.is_available():
                self.logger.info("CUDA is available")

                pipe = StableDiffusion3Pipeline.from_pretrained(
                    self.model,
                )
                self.num_inference_steps = 30
                torch.backends.cuda.matmul.allow_tf32 = True
                pipe = pipe.to(get_cuda_available_device())
                # pipe.enable_model_cpu_offload()

            else:
                self.logger.info("CUDA is **not** available")
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    self.model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
                )
                pipe = pipe.to("cpu")

                self.num_inference_steps = 5

            pipe.enable_attention_slicing()
            pipe.safety_checker = None

            # _ = pipe(prompt, num_inference_steps=1)

            self.pipe = pipe

    def __del__(self):
        self.logger.info("Claiming memory")
        self.cleanup()

    def __init__(self, id:str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.pipe = None
        self.num_inference_steps = 20

    async def work(self, request: TextToImageRequest) -> TextToImageResponse:
        await super().work(request)

        self.logger.info(f"Generate Image with {request}")
        if self.pipe == None:
            self.get_diffusion_pipe()

        prompts = list(request.text_prompts)
        images = []
        for prompt in prompts:
            image = self.pipe(
                prompt=prompt.text,
                num_inference_steps=request.steps,
            ).images[0]
            tampon_bytes = io.BytesIO()
            image.save(tampon_bytes, format='PNG')

            bytes_image = tampon_bytes.getvalue()

            if self.service.should_store():
                url = self.service.store_bytes(
                    bytes=bytes_image, name=prompt.text, extension="png")
                images.append({"url": url,
                              "seed": 42, "finishReason": "SUCCESS"})
            else:
                image_base64 = base64.b64encode(bytes_image)
                images.append(
                    {"base64": image_base64, "seed": request.seed, "finishReason": "SUCCESS"})

        return TextToImageResponse(artifacts=images)

    def cleanup(self):
        super().cleanup()   
        if self.pipe is not None:
            self.close_diffusion_pipe()
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()