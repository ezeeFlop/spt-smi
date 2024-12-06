from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List, Optional
from spt.models.workers import WorkerResult, WorkerBaseRequest

class TextPrompt(BaseModel):
    text: str = Field(..., example="A lighthouse on a cliff")

class TextToVideoRequest(WorkerBaseRequest):
    height: int = Field(default=512, ge=128, description="""""")

    width: int = Field(default=768, ge=128, description="""""")

    @field_validator('width')
    def must_be_multiple_of_64(cls, v):
        if v % 64 != 0:
            raise ValueError('Must be divided by 64')
        return v

    prompt: str = Field(..., example="A lighthouse on a cliff")

    negative_prompt: Optional[str] = Field(default=None, example="A lighthouse on a cliff")

    num_frames: int = Field(default=100, ge=50, le=256,
                       description="Numbers of frames to generate the video")
    
    frame_rate: int = Field(default=25, ge=1, le=60,
                       description="Frame rate of the video")
    
    seed: int = Field(default=0, ge=0, lt=4294967295,
                      description="Random noise seed (omit this option or use 0 for a random seed)")

    steps: int = Field(default=40, ge=1, le=100,
                       description="Numbers of steps to generate the video")

    guidance_scale: int = Field(default=3, ge=1, le=35,
                       description="Guidance scale for the video generation")
    
    seed: int = Field(default=171198, ge=0, lt=4294967295,
                      description="Random noise seed variation (omit this option or use 0 for no variation)")

class Artifact(BaseModel):
    base64: Optional[str] = None
    url: Optional[str] = None
    finishReason: WorkerResult = Field(..., example="SUCCESS")
    seed: int = Field(..., example=1050625087)

class TextToVideoResponse(BaseModel):
    artifacts: List[Artifact] = Field(..., example=[
            {
                "base64": "iVBORw0KGgoAAAANSUhEUgAAAO4AAAB...",
                "url": "https://cdn.sponge-theory.io/...",
                "finishReason": "SUCCESS",
                "seed": 1050625087
            }
        ])