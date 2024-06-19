from pydantic import BaseModel, Field
from typing import Optional, List, Any

class ConnectorLink(BaseModel):
    source: str
    target: str

class RequestResponseLink(BaseModel):
    source_model: str
    target_model: str
    links: List[ConnectorLink]

class SequentialGraph(BaseModel):
    steps: List[RequestResponseLink]