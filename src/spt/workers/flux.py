from spt.services.service import Worker, Service
import gc
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from spt.models.image import TextToImageResponse, TextToImageRequest
import base64
import PIL
import io
from spt.services.service import Service
from spt.storage import Storage
from diffusers import FluxPipeline

class Flux(Worker):

    def close_pipe(self):
        self.pipe = None
        self.generator = None
        torch.cuda.empty_cache()
        gc.collect()

    @classmethod
    def memory_usage(cls):
        max_memory = round(torch.cuda.max_memory_allocated(
            device='cuda') / 1000000000, 2)
        return max_memory

    def get_pipe(self):
        if self.pipe is None:
            pipe = None

            if torch.backends.mps.is_available():
                self.logger.info("MPS is available")
                pipe = FluxPipeline.from_pretrained(
                    self.model, torch_dtype=torch.bfloat16)
                # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
                pipe.enable_model_cpu_offload()
                pipe = pipe.to("mps")
                self.num_inference_steps = 30
                generator = torch.Generator(device='mps')

            elif torch.cuda.is_available():
                self.logger.info("CUDA is available")
                torch.cuda.empty_cache()

                pipe = FluxPipeline.from_pretrained(
                    self.model, torch_dtype=torch.bfloat16, device_map="balanced")
                # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
                pipe.reset_device_map()
                pipe.enable_model_cpu_offload()
                self.num_inference_steps = 30
                #pipe = pipe.to("cuda")
                # pipe.enable_model_cpu_offload()
                generator = torch.Generator(device='cuda:1')

            else:
                self.logger.info("CUDA is **not** available")
                pipe = FluxPipeline.from_pretrained(
                    self.model, torch_dtype=torch.bfloat16, device_map="balanced")
                # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
                pipe.enable_model_cpu_offload()
                pipe = pipe.to("cpu")
                generator = torch.Generator(device='cpu')

                self.num_inference_steps = 5

            self.pipe = pipe
            self.generator = generator

    def __del__(self):
        self.logger.info("Claiming memory")
        self.cleanup()

    def __init__(self, id: str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.pipe = None
        self.num_inference_steps = 20

    async def work(self, request: TextToImageRequest) -> TextToImageResponse:
        await super().work(request)

        self.logger.info(f"Generate Image with {request}")
        if self.pipe == None:
            self.get_pipe()

        if request.seed is not None:
            seed = request.seed
        else:
            seed = 42

        prompts = list(request.text_prompts)
        images = []
        for prompt in prompts:
            image=self.pipe(
                prompt.text,
                output_type="pil",
                # use a larger number if you are using [dev]
                num_inference_steps=self.num_inference_steps,
                generator=self.generator.manual_seed(seed)
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
        self.close_pipe()
        torch.cuda.empty_cache()
