import json
from pydantic import BaseModel, ValidationError, validator
from typing import Any, Dict, Type, Optional, List
import importlib
from spt.models.jobs import JobStatuses

class MethodCallRequest(BaseModel):
    remote_class: Optional[str] = None
    remote_method: Optional[str] = None
    request_model_class: Optional[str] = None
    response_model_class: Optional[str] = None
    remote_function: Optional[str] = None
    remote_module: Optional[str] = None
    keep_alive: Optional[int] = None
    storage: Optional[str] = None
    payload: Dict[str, Any]

    @validator('remote_class')
    def validate_class_name(cls, v):
        # Validation des classes autorisées
        #if v not in ['User', 'Product']:
        #    raise ValueError("Unauthorized class")
        return v

    @validator('remote_method')
    def validate_method_name(cls, v, values, **kwargs):
        # Validation des méthodes autorisées
        #class_name = values.get('class_name')
        #if class_name == 'User' and v not in ['activate', 'deactivate']:
        #    raise ValueError("Unauthorized method for class User")
        #elif class_name == 'Product' and v not in ['update_price']:
        #    raise ValueError("Unauthorized method for class Product")
        return v

class MethodCallError(BaseModel):
    message: str
    status: JobStatuses

def string_to_class(class_path: str) -> Type[BaseModel]:
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

def class_to_string(model_class: Type[BaseModel]) -> str:
    class_path = f"{model_class.__module__}.{model_class.__qualname__}"
    return class_path

def string_to_module(module_path: str):
    module = importlib.import_module(module_path)
    return module

# remote function call models response

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
