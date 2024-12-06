from pydantic import BaseModel, Field, field_serializer, field_validator
from typing import Optional
import base64
from .workers import WorkerBaseRequest

class TextToSpeechRequest(WorkerBaseRequest):
    text: str = Field(..., example="Hello, World!")
    language: str = Field(None, example="en")
    speaker_id: str = Field(..., example="virginie")

class TextToSpeechResponse(BaseModel):
    url: Optional[str] = Field(None, example="https://cdn.sponge-theory.io/audio.wav")
    base64: Optional[bytes] = Field(None, example="base64 audio wav file")

    @field_serializer('base64')
    def encode_file_to_base64(self, base64):
        if base64 is not None:
            return base64.b64encode(base64).decode('utf-8')
    
    @field_validator('base64', mode='before')
    def decode_file_from_base64(cls, v):
        if v is not None and isinstance(v, str):
            try:
                return base64.b64decode(v)
            except ValueError:
                raise ValueError("Invalid Base64 encoding")
        return v

class SpeechToTextRequest(WorkerBaseRequest):
    file: bytes = Field(..., example="base64 audio wav file")
    language: Optional[str] = Field(None, example="en")
    temperature: Optional[float] = Field(0.0, example=0.0)
    prompt: Optional[str] = Field(None, example="Transcribe the following audio")

    @field_serializer('file')
    def encode_file_to_base64(self, file):
        return base64.b64encode(file).decode('utf-8')
    
    @field_validator('file', mode='before')
    def decode_file_from_base64(cls, v):
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except ValueError:
                raise ValueError("Invalid Base64 encoding")
        return v

class SpeechToTextResponse(BaseModel):
    language: str = Field(..., example="en")
    text: str = Field(..., example="Hello, World!") 