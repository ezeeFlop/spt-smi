import httpx
import asyncio
import base64
from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel

from .models.jobs import JobsTypes, JobStatuses, JobResponse, JobPriority, JobStorage
from .models.image import TextToImageRequest, TextToImageResponse, StylesPreset, ClipGuidancePreset, SamplersPreset
from .models.video import TextToVideoRequest, TextToVideoResponse
from .models.llm import ChatRequest, ChatResponse, ChatMessage, EmbeddingsRequest, EmbeddingsResponse, LLMOptions
from .models.audio import TextToSpeechRequest, TextToSpeechResponse, SpeechToTextRequest, SpeechToTextResponse
from .models.workers import WorkerConfigs
from .models.remotecalls import GPUsInfo

class SMIClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8999"):
        """Initialize the SMI client.
        
        Args:
            api_key (str): The API key for authentication
            base_url (str, optional): The base URL of the API. Defaults to "http://localhost:8999"
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "x-smi-key": self.api_key,
        }
        self.client = httpx.AsyncClient()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _add_optional_headers(self, headers: Dict[str, str], 
                            async_mode: bool = False, 
                            keep_alive: Optional[int] = None,
                            storage: Optional[str] = None,
                            priority: Optional[str] = None) -> Dict[str, str]:
        """Add optional headers to the request."""
        if async_mode:
            headers["x-smi-async"] = "true"
        if keep_alive is not None:
            headers["x-smi-keep-alive"] = str(keep_alive)
        if storage:
            headers["x-smi-storage"] = storage
        if priority:
            headers["x-smi-priority"] = priority
        return headers

    async def list_workers(self) -> WorkerConfigs:
        """Get the list of available worker configurations."""
        response = await self.client.get(f"{self.base_url}/v1/workers/list", headers=self.headers)
        response.raise_for_status()
        return WorkerConfigs.model_validate(response.json())

    async def get_gpu_info(self) -> Union[GPUsInfo, Dict[str, Any]]:
        """Get information about available GPUs."""
        response = await self.client.get(f"{self.base_url}/v1/gpu/info", headers=self.headers)
        response.raise_for_status()
        return GPUsInfo.model_validate(response.json())

    async def text_to_image(self, 
                           request: TextToImageRequest,
                           async_mode: bool = False,
                           keep_alive: Optional[int] = None,
                           storage: Optional[str] = None,
                           priority: Optional[str] = None,
                           accept_format: Optional[str] = None) -> Union[JobResponse, TextToImageResponse]:
        """Generate an image from text."""
        headers = self._add_optional_headers(
            self.headers.copy(),
            async_mode=async_mode,
            keep_alive=keep_alive,
            storage=storage,
            priority=priority
        )
        if accept_format:
            headers["accept"] = accept_format

        response = await self.client.post(
            f"{self.base_url}/v1/text-to-image",
            headers=headers,
            json=request.model_dump()
        )
        response.raise_for_status()
        
        if async_mode:
            return JobResponse.model_validate(response.json())
        return TextToImageResponse.model_validate(response.json())

    async def text_to_video(self,
                           request: TextToVideoRequest,
                           async_mode: bool = False,
                           keep_alive: Optional[int] = None,
                           storage: Optional[str] = None,
                           priority: Optional[str] = None,
                           accept_format: Optional[str] = None) -> Union[JobResponse, TextToVideoResponse]:
        """Generate a video from text."""
        headers = self._add_optional_headers(
            self.headers.copy(),
            async_mode=async_mode,
            keep_alive=keep_alive,
            storage=storage,
            priority=priority
        )
        if accept_format:
            headers["accept"] = accept_format

        response = await self.client.post(
            f"{self.base_url}/v1/text-to-video",
            headers=headers,
            json=request.model_dump()
        )
        response.raise_for_status()
        
        if async_mode:
            return JobResponse.model_validate(response.json())
        return TextToVideoResponse.model_validate(response.json())

    async def text_to_text(self,
                          request: ChatRequest,
                          async_mode: bool = False,
                          keep_alive: Optional[int] = None,
                          storage: Optional[str] = None,
                          priority: Optional[str] = None) -> Union[JobResponse, ChatResponse]:
        """Generate text from text (chat completion)."""
        headers = self._add_optional_headers(
            self.headers.copy(),
            async_mode=async_mode,
            keep_alive=keep_alive,
            storage=storage,
            priority=priority
        )

        response = await self.client.post(
            f"{self.base_url}/v1/text-to-text",
            headers=headers,
            json=request.model_dump()
        )
        response.raise_for_status()
        
        if async_mode:
            return JobResponse.model_validate(response.json())
        return ChatResponse.model_validate(response.json())

    async def image_to_text(self,
                           request: ChatRequest,
                           async_mode: bool = False,
                           keep_alive: Optional[int] = None,
                           storage: Optional[str] = None,
                           priority: Optional[str] = None) -> Union[JobResponse, ChatResponse]:
        """Generate text description from an image."""
        headers = self._add_optional_headers(
            self.headers.copy(),
            async_mode=async_mode,
            keep_alive=keep_alive,
            storage=storage,
            priority=priority
        )

        response = await self.client.post(
            f"{self.base_url}/v1/image-to-text",
            headers=headers,
            json=request.model_dump()
        )
        response.raise_for_status()
        
        if async_mode:
            return JobResponse.model_validate(response.json())
        return ChatResponse.model_validate(response.json())

    async def text_to_embeddings(self,
                                request: EmbeddingsRequest,
                                async_mode: bool = False,
                                keep_alive: Optional[int] = None,
                                storage: Optional[str] = None,
                                priority: Optional[str] = None) -> Union[JobResponse, EmbeddingsResponse]:
        """Generate embeddings from text."""
        headers = self._add_optional_headers(
            self.headers.copy(),
            async_mode=async_mode,
            keep_alive=keep_alive,
            storage=storage,
            priority=priority
        )

        response = await self.client.post(
            f"{self.base_url}/v1/text-to-embeddings",
            headers=headers,
            json=request.model_dump()
        )
        response.raise_for_status()
        
        if async_mode:
            return JobResponse.model_validate(response.json())
        return EmbeddingsResponse.model_validate(response.json())

    async def text_to_speech(self,
                            request: TextToSpeechRequest,
                            async_mode: bool = False,
                            keep_alive: Optional[int] = None,
                            storage: Optional[str] = None,
                            priority: Optional[str] = None,
                            accept_format: Optional[str] = None) -> Union[JobResponse, TextToSpeechResponse]:
        """Generate speech from text."""
        headers = self._add_optional_headers(
            self.headers.copy(),
            async_mode=async_mode,
            keep_alive=keep_alive,
            storage=storage,
            priority=priority
        )
        if accept_format:
            headers["accept"] = accept_format

        response = await self.client.post(
            f"{self.base_url}/v1/text-to-speech",
            headers=headers,
            json=request.model_dump()
        )
        response.raise_for_status()
        
        if async_mode:
            return JobResponse.model_validate(response.json())
        return TextToSpeechResponse.model_validate(response.json())

    async def speech_to_text(self,
                            audio_file: bytes,
                            worker_id: str,
                            language: Optional[str] = None,
                            temperature: float = 0.0,
                            prompt: Optional[str] = None,
                            async_mode: bool = False,
                            keep_alive: Optional[int] = None,
                            storage: Optional[str] = None,
                            priority: Optional[str] = None) -> Union[JobResponse, SpeechToTextResponse]:
        """Convert speech to text."""
        headers = self._add_optional_headers(
            self.headers.copy(),
            async_mode=async_mode,
            keep_alive=keep_alive,
            storage=storage,
            priority=priority
        )

        files = {
            'file': ('audio.wav', audio_file, 'audio/wav')
        }
        data = {
            'worker_id': worker_id,
            'language': language,
            'temperature': temperature,
            'prompt': prompt
        }

        response = await self.client.post(
            f"{self.base_url}/v1/speech-to-text",
            headers=headers,
            files=files,
            data={k: v for k, v in data.items() if v is not None}
        )
        response.raise_for_status()
        
        if async_mode:
            return JobResponse.model_validate(response.json())
        return SpeechToTextResponse.model_validate(response.json())

    async def get_job_status(self, job_id: str, endpoint: str) -> Union[JobResponse, Any]:
        """Get the status of an asynchronous job."""
        response = await self.client.get(
            f"{self.base_url}/v1/{endpoint}/{job_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json() 