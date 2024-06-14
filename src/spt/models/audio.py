from pydantic import BaseModel, Field, validator, field_serializer
from typing import Optional, List, Dict
import base64
from spt.models.workers import WorkerBaseRequest

class TextToSpeechRequest(WorkerBaseRequest):
    text: str = Field(..., example="Hello, World!")
    language: str = Field(None, example="en")
    speaker_id: str = Field(..., example="virginie")

class TextToSpeechResponse(BaseModel):
    url: Optional[str] = Field(None, example="https://www.gstatic.com/webp/gallery/1.jpg")
    base64: Optional[bytes] = Field(None, example="base64 audio wav file")

    @field_serializer('base64')
    def encode_file_to_base64(self, base64):
        if base64 is not None:
            return base64.b64encode(base64).decode('utf-8')
    
    @validator('base64', always=True, pre=True)
    def decode_file_from_base64(cls, v):
        if v is not None and isinstance(v, str):  # Assuming input will be a base64 string from JSON
            try:
                return base64.b64decode(v)
            except ValueError:
                raise ValueError("Invalid Base64 encoding")
        return v  # Pass through if it's already in bytes (not from JSON)


class TextToSpeechSpeakerRequest(WorkerBaseRequest):
    id: str = Field(..., example="virginie")
    sample: bytes = Field(..., example="base64 audio wav file")
    @field_serializer('sample')
    def encode_file_to_base64(self, sample):
        return base64.b64encode(sample).decode('utf-8')
    
    @validator('sample', always=True, pre=True)
    def decode_file_from_base64(cls, v):
        if isinstance(v, str):  # Assuming input will be a base64 string from JSON
            try:
                return base64.b64decode(v)
            except ValueError:
                raise ValueError("Invalid Base64 encoding")
        return v  # Pass through if it's already in bytes (not from JSON)
    

class SpeechToTextRequest(WorkerBaseRequest):
    file: bytes = Field(
        ...,
        description='The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.\n',
    )
    language: Optional[str] = Field(
        None,
        description='The language of the input audio. Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency.\n',
    )
    temperature: Optional[float] = Field(
        0,
        description='The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit.\n',
    )
    prompt: Optional[str] = Field(
        None,
        description="An optional text to guide the model's style or continue a previous audio segment. The [prompt](/docs/guides/speech-to-text/prompting) should match the audio language.\n",
    )

    @field_serializer('file')
    def encode_file_to_base64(self, file):
        return base64.b64encode(file).decode('utf-8')

    @validator('file', always=True, pre=True)
    @classmethod
    def decode_file_from_base64(cls, v):
        if isinstance(v, str):  # Assuming input will be a base64 string from JSON
            try:
                return base64.b64decode(v)
            except ValueError:
                raise ValueError("Invalid Base64 encoding")

        return v  # Pass through if it's already in bytes (not from JSON)

#
class SpeechToTextResponse(BaseModel):
    language: str = Field(..., example="en")
    text: str = Field(..., example="Hello, World!")