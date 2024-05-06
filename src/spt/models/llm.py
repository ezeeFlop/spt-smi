from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

# Modèles pour l'endpoint Generate a completion

class Options(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_predict: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    num_keep: Optional[int] = None
    typical_p: Optional[float] = None
    tfs_z: Optional[float] = None
    repeat_penalty: Optional[float] = None
    mirostat: Optional[bool] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    # Add more options here as needed

class GenerateRequest(BaseModel):
    model: str = Field(..., example="text-davinci-003")
    prompt: str  = Field(..., example="Hello, World!")
    images: Optional[List[str]] = None
    format: Optional[str] = None
    options: Optional[Options] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[str] = None
    stream: Optional[bool] = None
    raw: Optional[bool] = None
    
class GenerateResponse(BaseModel):
    model: str = Field(..., example="text-davinci-003")
    created_at: str =  Field(..., example="2022-08-01T00:00:00Z")
    response: str = Field(..., example="Hello, World!")
    done: bool = Field(..., example=True)
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

# Modèles pour l'endpoint Generate a chat completion


class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    format: Optional[str] = None
    options: Optional[Options] = None
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = 0
    load_duration: Optional[int] = 0
    prompt_eval_count: Optional[int] = 0
    prompt_eval_duration: Optional[int] = 0
    eval_count: Optional[int] = 0
    eval_duration: Optional[int] = 0

# Modèles pour l'endpoint Generate Embeddings


class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Options] = None


class EmbeddingsResponse(BaseModel):
    embedding: List[float]


class EngineResult(str, Enum):
    success = "SUCCESS",
    error = "ERROR"