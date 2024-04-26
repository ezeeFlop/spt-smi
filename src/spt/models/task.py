from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import List, Optional
from spt.utils import load_json
from config import CONFIG_PATH


class FunctionTask(BaseModel):
    function: Optional[str] = None
    module: Optional[str] = None
    payload: Optional[dict] = None

class MethodTask(BaseModel):
    method: Optional[str] = None
    className: Optional[str] = None
    payload: Optional[dict] = None