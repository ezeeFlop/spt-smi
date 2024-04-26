import gc
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, UNet2DConditionModel, EulerDiscreteScheduler
import torch
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from spt.models.txt2img import EnginesList, ModelType, ArtifactsList, TextToImageRequest
import base64
import PIL
import io
import logging
from spt.services.service import Service
from spt.services.generic.service import GenericServiceServicer
from spt.storage import Storage

logger = logging.getLogger(__name__)

def remap(model_engines):
    models = {}
    for engine in model_engines:
        if engine["type"] == ModelType.picture.value:
            models[engine["id"]] = {
                "id": engine["id"],
                "model": engine["name"],
                "description": engine["description"],
                "pipe": None,
                "generator": None
            }
    return models

class DiffusionModels(Service):
    models = remap(EnginesList.get_engines())

    def __init__(self, servicer: GenericServiceServicer = None) -> None:
        super().__init__(servicer=servicer)
        self.model_id = None

    def __del__(self):
        pass

    def cleanup(self):
        super().cleanup()
        if self.model_id is not None:
            logger.info(f"Closing model {self.model_id}")
            DiffusionModels.close_diffusion_pipe(self.model_id)

    @classmethod
    def get_model(cls, model_id):
        if cls.models[model_id] is not None:
            return cls.models[model_id]
        return None

    @classmethod
    def close_diffusion_pipe(cls, model_id):
        model = DiffusionModels.get_model(model_id=model_id)
        if model["pipe"] is not None:
            model["pipe"] = None
            model["generator"] = None
            torch.cuda.empty_cache()
            gc.collect()

    @classmethod
    def memory_usage(cls):
        max_memory = round(torch.cuda.max_memory_allocated(
            device='cuda') / 1000000000, 2)
        return max_memory

    @classmethod
    def get_diffusion_pipe(cls, model_id='diffusion'):
        model = DiffusionModels.get_model(model_id)

        if model["pipe"] is None:
            pipe = None
            generator = None

            if torch.backends.mps.is_available():
                logger.info("MPS is available")
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model["model"],
                )
                pipe = pipe.to("mps")
                model['num_inference_steps'] = 30
                generator = torch.Generator(device='mps')

            elif torch.cuda.is_available():
                logger.info("CUDA is available")

                pipe = AutoPipelineForText2Image.from_pretrained(
                    model["model"],

                    #model["model"], torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
                )
                model['num_inference_steps'] = 30
                torch.backends.cuda.matmul.allow_tf32 = True
                pipe = pipe.to("cuda")
                # pipe.enable_model_cpu_offload()
                generator = torch.Generator(device='cuda')

            else:
                logger.info("CUDA is **not** available")
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model["model"], torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
                )
                pipe = pipe.to("cpu")
                generator = torch.Generator(device='cpu')

                model['num_inference_steps'] = 5

            pipe.enable_attention_slicing()
            pipe.safety_checker = None

            # _ = pipe(prompt, num_inference_steps=1)

            model["pipe"] = pipe
            model["generator"] = generator

        return model

    def generate_images(self, request: TextToImageRequest) -> ArtifactsList:
        logger.info(f"Generate Image with {request}")
        model = DiffusionModels.get_diffusion_pipe(request.model)
        self.model_id = request.model

        if model["pipe"] is None:
            return None

        pipe = model["pipe"]
        generator = model["generator"]

        if request.seed is not None:
            generator.manual_seed(request.seed)
        prompts = list(request.text_prompts)
        images = []
        for prompt in prompts:
            image = pipe(
                prompt=prompt.text,
                generator=generator,
                num_inference_steps=request.steps,
            ).images[0]
            tampon_bytes = io.BytesIO()
            image.save(tampon_bytes, format='PNG')  # Tu peux changer 'PNG' en 'JPEG' selon le format souhaité
   
            # Obtenir les bytes de l'image
            bytes_image = tampon_bytes.getvalue()

            if self.should_store():
                url = self.store_bytes(bytes=bytes_image, name=prompt.text, extension="png")
                images.append({"url": url,
                              "seed": 42, "finishReason": "SUCCESS"})
            else:
                 # Encoder les bytes en base64
                image_base64 = base64.b64encode(bytes_image)            
                images.append({"base64": image_base64, "seed": request.seed, "finishReason": "SUCCESS"})

        return ArtifactsList(artifacts=images)


def main():
    MODEL_NAME = "disneyPixar"

    models = DiffusionModels.list_models()
    print(models)

    image = DiffusionModels.generate_image(
        "A painting of a cat", model_name=MODEL_NAME)

    DiffusionModels.close_diffusion_pipe(MODEL_NAME)

    image.save(f'image.png')


if __name__ == '__main__':
    main()
