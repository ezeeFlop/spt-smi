from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from .workers import WorkerResult, WorkerBaseRequest

class StylesPreset(str, Enum):
    threeD_model = "3d-model"
    analog_film = "analog-film"
    anime = "anime"
    cinematic = "cinematic"
    comic_book = "comic-book"
    digital_art = "digital-art"
    enhance = "enhance"
    fantasy_art = "fantasy-art"
    isometric = "isometric"
    line_art = "line-art"
    low_poly = "low-poly"
    modeling_compound = "modeling-compound"
    neon_punk = "neon-punk"
    origami = "origami"
    photographic = "photographic"
    pixel_art = "pixel-art"
    tile_texture = "tile-texture"

class ClipGuidancePreset(str, Enum):
    fast_blue = "FAST_BLUE"
    fast_green = "FAST_GREEN"
    none = "NONE"
    simple = "SIMPLE"
    slow = "SLOW"
    slower = "SLOWER"
    slowest = "SLOWEST"

class SamplersPreset(str, Enum):
    ddim = "DDIM"
    ddpm = "DDPM"
    k_dpmpp_2m = "K_DPMPP_2M"
    k_dpmpp_2s_a = "K_DPMPP_2S_A"
    ncestral = "NCESTRAL"
    k_dpm_2 = "K_DPM_2"
    k_dpm_2_ancestral = "K_DPM_2_ANCESTRAL"
    k_euler = "K_EULER"
    k_euler_ancestral = "K_EULER_ANCESTRAL"
    k_heun = "K_HEUN"
    k_lms = "K_LMS"

class TextToImageRequest(WorkerBaseRequest):
    height: int = Field(default=512, ge=128, description="Height of the generated image")
    width: int = Field(default=768, ge=128, description="Width of the generated image")
    prompt: str = Field(..., example="A beautiful sunset over mountains")
    negative_prompt: Optional[str] = Field(default=None, example="blurry, low quality")
    steps: int = Field(default=1, ge=1, le=100, description="Numbers of steps to generate the image")
    samples: int = Field(default=1, ge=1, le=10, description="Numbers of samples to generate")
    cfg_scale: int = Field(default=7, ge=1, le=35, description="How strictly the diffusion process adheres to the prompt")
    clip_guidance_preset: ClipGuidancePreset = Field(default=ClipGuidancePreset.none)
    sampler: SamplersPreset = Field(default=SamplersPreset.ddim)
    seed: int = Field(default=0, ge=0, lt=4294967295)
    style_preset: StylesPreset = Field(default=StylesPreset.photographic)

    @field_validator('width')
    def must_be_multiple_of_64(cls, v):
        if v % 64 != 0:
            raise ValueError('Width must be divided by 64')
        return v

class Artifact(BaseModel):
    base64: Optional[str] = None
    url: Optional[str] = None
    finishReason: WorkerResult = Field(..., example="SUCCESS")
    seed: int = Field(..., example=1050625087)

class TextToImageResponse(BaseModel):
    artifacts: List[Artifact] = Field(..., example=[{
        "base64": "iVBORw0KGgoAAAANSUhEUgAAAO4AAAB...",
        "url": "https://cdn.sponge-theory.io/...",
        "finishReason": "SUCCESS",
        "seed": 1050625087
    }]) 