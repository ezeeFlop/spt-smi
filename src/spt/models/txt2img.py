from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import List, Optional
from spt.utils import load_json
from config import CONFIG_PATH

class ModelType(str, Enum):
    audio = "AUDIO",
    classification = "CLASSIFICATION",
    picture = "PICTURE",
    storage = "STORAGE",
    text = "TEXT",
    video = "VIDEO"

class StylesPreset(str, Enum):
    threeD_model = "3d-model",
    analog_film = "analog-film",
    anime = "anime",
    cinematic = "cinematic",
    comic_book = "comic-book",
    digital_art = "digital-art",
    enhance = "enhance",
    fantasy_art = "fantasy-art",
    isometric = "isometric",
    line_art = "line-art",
    low_poly = "low-poly",
    modeling_compound = "modeling-compound",
    neon_punk = "neon-punk",
    origami = "origami",
    photographic = "photographic",
    pixel_art = "pixel-art",
    tile_texture = "tile-texture"

class ClipGuidancePreset(str, Enum):
    fast_blue = "FAST_BLUE",
    fast_green = "FAST_GREEN",
    none = "NONE",
    simple = "SIMPLE",
    slow = "SLOW",
    slower = "SLOWER",
    slowest = "SLOWEST"

class SamplersPreset(str, Enum):
    ddim = "DDIM",
    ddpm = "DDPM",
    k_dpmpp_2m = "K_DPMPP_2M",
    k_dpmpp_2s_a = "K_DPMPP_2S_A",
    ncestral = "NCESTRAL",
    k_dpm_2 = "K_DPM_2",
    k_dpm_2_ancestral = "K_DPM_2_ANCESTRAL",
    k_euler = "K_EULER",
    k_euler_ancestral = "K_EULER_ANCESTRAL",
    k_heun = "K_HEUN ",
    k_lms = "K_LMS"

class TextPrompt(BaseModel):
    text: str = Field(..., example="A lighthouse on a cliff")
    weight: float = Field(default=0.5, ge=0, le=1,
                          description="Poids du prompt, entre 0 et 1")

class TextToImageRequest(BaseModel):
    model: str = Field(default=ModelType.text, description="Type of model to use")
    height: int = Field(default=512, ge=128, description="""Height of the image to generate, in pixels, in an increment divible by 64.

Engine-specific dimension validation:

SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
SDXL v1.0: same as SDXL v0.9
SD v1.6: must be between 320x320 and 1536x1536""")

    width: int = Field(default=512, ge=128, description="""Width of the image to generate, in pixels, in an increment divible by 64.

Engine-specific dimension validation:

SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
SDXL v1.0: same as SDXL v0.9
SD v1.6: must be between 320x320 and 1536x1536""")

    @validator('width')
    def must_be_multiple_of_64(cls, v):
        if v % 64 != 0:
            raise ValueError('Must be divided by 64')
        return v

    text_prompts: List[TextPrompt] = Field(..., example=[
        {"text": "A lighthouse on a cliff", "weight": 0.5}
    ])

    steps: int = Field(default=1, ge=1, le=100,
                       description="Numbers of steps to generate the image")
    samples: int = Field(default=1, ge=1, le=10,
                         description="Numbers of samples to generate the image")

    cfg_scale: int = Field(default=7, ge=1, le=35,
                           description="How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)")

    clip_guidance_preset: ClipGuidancePreset = Field(
        default=ClipGuidancePreset.none, description="How much guidance to use from the CLIP model")

    sampler: SamplersPreset = Field(
        default=SamplersPreset.ddim, description="Which sampler to use for the diffusion process. If this value is omitted we'll automatically select an appropriate sampler for you.")

    seed: int = Field(default=0, ge=0, lt=4294967295,
                      description="Random noise seed (omit this option or use 0 for a random seed)")

    style_preset: StylesPreset = Field(default=StylesPreset.photographic,
                                       description="Pass in a style preset to guide the image model towards a particular style. This list of style presets is subject to change.")


class Models(BaseModel):
    name: str = Field(..., example="SDXL Beta")
    id: str = Field(..., example="sdxl-beta")
    description: str = Field(..., example="SDXL Beta")
    type: ModelType = Field(..., example="TEXT")

class EnginesList(BaseModel):
    engines: List[Models] = Field(..., example=[
            {
                "description" : "Realistic Vision",
                "id" : "realisticVision",
                "name" : "SG161222/Realistic_Vision_V3.0_VAE",
                "type" : "PICTURE"
            },
            {
                "description": "Disney Pixar",
                "id": "disneyPixar",
                "name": "stablediffusionapi/disney-pixar-cartoon",
                "type": "PICTURE"
            }
        ])
    
    @classmethod
    def get_engines(self):
        return load_json("engines", CONFIG_PATH)

class EngineResult(str, Enum):
    success = "SUCCESS",
    error = "ERROR",
    content_filtered = "CONTENT_FILTERED"

class Artifact(BaseModel):
    base64: Optional[str] = None
    url: Optional[str] = None
    finishReason: EngineResult = Field(..., example="SUCCESS")
    seed: int = Field(..., example=1050625087)

class ArtifactsList(BaseModel):
    artifacts: List[Artifact] = Field(..., example=[
            {
                "base64": "iVBORw0KGgoAAAANSUhEUgAAAO4AAAB...",
                "url": "https://cdn.sponge-theory.io/...",
                "finishReason": "SUCCESS",
                "seed": 1050625087
            }
        ])