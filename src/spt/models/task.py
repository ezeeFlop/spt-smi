from pydantic import BaseModel
from typing import Optional

class FunctionTask(BaseModel):
    function: Optional[str] = None
    module: Optional[str] = None
    payload: Optional[dict] = None

class MethodTask(BaseModel):
    method: Optional[str] = None
    className: Optional[str] = None
    payload: Optional[dict] = None