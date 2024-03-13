import gc
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, UNet2DConditionModel, EulerDiscreteScheduler
import torch
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class DiffusionModels:
    models = {
        "realisticVision": {
            "model": "SG161222/Realistic_Vision_V3.0_VAE",
            "pipe": None,
            "generator": None,
            "type": "diffusion",
        },
        "disneyPixar": {
            "model": "stablediffusionapi/disney-pixar-cartoon",
            "pipe": None,
            "generator": None,
            "type": "diffusion",
        },
        "stable-diffusion-xl": {
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "pipe": None,
            "generator": None,
            "type": "diffusionXL",
        },
        "stable-diffusion-turbo": {
            "pipe": None,
            "generator": None,
            "type": "diffusion",
            "model": "stabilityai/sdxl-turbo"
        }
    }

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    @classmethod
    def get_model(cls, model_name):
        if DiffusionModels.models[model_name] is not None:
            return DiffusionModels.models[model_name]
        return None

    @classmethod
    def list_models(cls):
        return list(cls.models.keys())

    @classmethod
    def model_index(cls, model_name):
        index = 0
        for model in cls.models.keys():
            if model == model_name:
                return index
            index += 1
        return 0

    @classmethod
    def close_diffusion_pipe(cls, model_name):
        model = DiffusionModels.get_model(model_name)
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
    def get_diffusion_pipe(cls, prompt, model_name='diffusion'):
        model = DiffusionModels.get_model(model_name)

        if model == model_name:
            return None

        if model["pipe"] is None:
            pipe = None
            generator = None

            if torch.backends.mps.is_available():
                print("MPS is available")
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model["model"],
                )
                pipe = pipe.to("mps")
                model['num_inference_steps'] = 30
                generator = torch.Generator(device='mps')

            elif torch.cuda.is_available():
                print("CUDA is available")

                pipe = AutoPipelineForText2Image.from_pretrained(
                    model["model"], torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
                )
                model['num_inference_steps'] = 30
                torch.backends.cuda.matmul.allow_tf32 = True
                pipe = pipe.to("cuda")
                # pipe.enable_model_cpu_offload()
                generator = torch.Generator(device='cuda')

            else:
                print("CUDA is not available")
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

    @classmethod
    def generate_image(cls, prompt, seed=None, model_name="diffusion"):
        model = DiffusionModels.get_diffusion_pipe(prompt, model_name)

        if model["pipe"] is None:
            return None

        pipe = model["pipe"]
        generator = model["generator"]

        if seed is not None:
            generator.manual_seed(seed)

        image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=model['num_inference_steps'],
        ).images[0]

        return image


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
