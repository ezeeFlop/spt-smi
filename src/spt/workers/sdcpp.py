import asyncio
import base64
import io
import os
import subprocess
import requests
from pathlib import Path
from spt.services.service import Worker, Service
from spt.models.image import TextToImageResponse, TextToImageRequest

class StableDiffusionCpp(Worker):
    def __init__(self, id: str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.sd_binary_path = "/sd"
        self.num_inference_steps = 20
        self.cache_dir = Path("/home/spt/.cache/flux")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.flux_files = {
            "flux-dev": {
                "flux1-dev-q3_k.gguf": "https://huggingface.co/leejet/FLUX.1-dev-gguf/blob/main/flux1-dev-q3_k.gguf",
                "ae.sft": "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors",
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors"
            },
            "flux-schnell": {
                "flux1-schnell-q3_k.gguf": "https://huggingface.co/leejet/FLUX.1-schnell-gguf/blob/main/flux1-schnell-q3_k.gguf",
                "ae.sft": "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors",
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors"
            }
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

        for prompt in prompts:
            cmd = [
                self.sd_binary_path,
                "--diffusion-model", model_paths[f"flux1-{self.model.split('-')[1]}.{'safetensors' if 'schnell' in self.model else 'gguf'}"],
                "--vae", model_paths["ae.sft"],
                "--clip_l", model_paths["clip_l.safetensors"],
                "--t5xxl", model_paths["t5xxl_fp16.safetensors"],
                "-p", prompt.text,
                "--steps", str(self.num_inference_steps),
                "--seed", str(seed),
                "-o", "temp_output.png",
                "--cfg-scale", "1.0",
                "--sampling-method", "euler"
            ]

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    self.logger.error(f"Error running stable-diffusion.cpp: {stderr.decode()}")
                    continue

                with open("temp_output.png", "rb") as img_file:
                    bytes_image = img_file.read()

                os.remove("temp_output.png")

                if self.service.should_store():
                    url = self.service.store_bytes(
                        bytes=bytes_image, name=prompt.text, extension="png")
                    images.append({"url": url, "seed": seed, "finishReason": "SUCCESS"})
                else:
                    image_base64 = base64.b64encode(bytes_image).decode('utf-8')
                    images.append(
                        {"base64": image_base64, "seed": seed, "finishReason": "SUCCESS"})

            except Exception as e:
                self.logger.error(f"Error generating image: {str(e)}")

        return TextToImageResponse(artifacts=images)

    def cleanup(self):
        super().cleanup()
        # No need to clean up CUDA memory as we're using a separate process