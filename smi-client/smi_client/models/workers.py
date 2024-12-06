from pydantic import BaseModel, Field, PrivateAttr
from enum import Enum
from typing import List, Optional, Dict

class WorkerState(str, Enum):
    idle = "IDLE"
    working = "WORKING"
    streaming = "STREAMING"

class WorkerStreamType(str, Enum):
    json = "JSON"
    bytes = "BYTES"
    text = "TEXT"

class WorkerType(str, Enum):
    audio = "AUDIO"
    classification = "CLASSIFICATION"
    picture = "IMAGE"
    storage = "STORAGE"
    llm = "LLM"
    tts = "TTS"
    stt = "STT"
    video = "VIDEO"
    embeddings = "EMBEDDING"

class WorkerResult(str, Enum):
    success = "SUCCESS"
    error = "ERROR"
    content_filtered = "CONTENT_FILTERED"

class WorkerBaseRequest(BaseModel):
    worker_id: str = Field(..., example="realisticVision")

class WorkerConfig(BaseModel):
    model: str = Field(..., example="SDXL Beta")
    description: str = Field(..., example="SDXL Beta")
    worker: str = Field(..., example="spt.workers.whisper")
    type: WorkerType = Field(..., example="TEXT")
    request_model: str = Field(..., example="spt.models.txt2img.TextToImageRequest")
    response_model: str = Field(..., example="spt.models.txt2img.TextToImageResponse")

class WorkerConfigs(BaseModel):
    _loaded_engines: Optional[List[WorkerConfig]] = PrivateAttr(None)
    workers_configs: Dict[str, WorkerConfig] = Field(..., example={
        "realisticVision": {
            "description": "Realistic Vision",
            "model": "realisticVision",
            "name": "SG161222/Realistic_Vision_V3.0_VAE",
            "type": "IMAGE"
        },
        "disneyPixar": {
            "description": "Disney Pixar",
            "model": "disneyPixar",
            "name": "stablediffusionapi/disney-pixar-cartoon",
            "type": "IMAGE"
        }
    }) 