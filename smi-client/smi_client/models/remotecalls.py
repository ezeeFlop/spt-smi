from pydantic import BaseModel, Field
from typing import List, Optional
from .jobs import JobStatuses

class GPUInfo(BaseModel):
    name: str
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_gpu_percent: int
    utilization_memory_percent: int

class GPUsInfo(BaseModel):
    gpus: List[GPUInfo]
    error: Optional[str] = None

class FunctionCallError(BaseModel):
    error: str
    message: str 