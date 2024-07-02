from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from spt.models.workers import WorkerBaseRequest

# Modèles pour l'endpoint Generate a completion

class LLMOptions(BaseModel):
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

# Modèles pour l'endpoint Generate a chat completion

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class ChatRequest(WorkerBaseRequest):
    messages: List[ChatMessage]
    format: Optional[str] = None
    options: Optional[LLMOptions] = None
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

class EmbeddingsRequest(WorkerBaseRequest):
    prompt: str
    options: Optional[LLMOptions] = None

class EmbeddingsResponse(BaseModel):
    embedding: List[float]
