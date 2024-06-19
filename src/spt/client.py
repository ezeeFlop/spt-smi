import httpx
import json
import asyncio
import websockets
from typing import Callable, Optional, Union, AsyncGenerator, Dict, Any
from pydantic import BaseModel
from spt.models.image import TextToImageRequest, TextToImageResponse
from spt.models.jobs import JobResponse, JobStatuses, JobPriority, JobStorage
from spt.models.workers import WorkerConfigs
from spt.models.llm import ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse
from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse, TextToSpeechRequest, TextToSpeechResponse
from spt.models.remotecalls import GPUsInfo, FunctionCallError
from spt.workers.utils.audio import load_audio

API_URL = "http://localhost:8999"  # Replace with your API URL
WS_API_URL = "ws://localhost:8999"  # Replace with your API URL

#API_URL = "https://smi-api.sponge-theory.dev"  # Replace with your API URL
#WS_API_URL = "wss://smi-api.sponge-theory.dev"  # Replace with your API URL



class SMIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "x-smi-key": self.api_key,
        }

    async def text_to_image(self, request_data: TextToImageRequest, worker_id: str, async_key: Optional[str] = None,
                            keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None,
                            priority_key: Optional[str] = None, accept: Optional[str] = None) -> Union[JobResponse, TextToImageResponse, bytes]:
        """
        Generates an image from text using the specified worker.

        :param request_data: Request data for text to image conversion
        :param worker_id: Worker ID to handle the request
        :param async_key: Optional async key for asynchronous processing
        :param keep_alive_key: Optional keep-alive duration
        :param storage_key: Optional storage key (local or S3)
        :param priority_key: Optional priority key (low, normal, high)
        :param accept: Optional accept header to specify response format
        :return: JobResponse, TextToImageResponse, or bytes (image data)
        """
        async with httpx.AsyncClient() as client:
            headers = self.headers.copy()
            if async_key:
                headers["x-smi-async"] = async_key
            if keep_alive_key is not None:
                headers["x-smi-keep-alive"] = str(keep_alive_key)
            if storage_key:
                headers["x-smi-storage"] = storage_key
            if priority_key:
                headers["x-smi-priority"] = priority_key
            if accept:
                headers["accept"] = accept

            response = await client.post(
                f"{API_URL}/v1/text-to-image",
                headers=headers,
                json=request_data.model_dump(),
                params={"worker_id": worker_id}
            )
            response.raise_for_status()
            if accept == "image/png":
                return response.content
            if async_key:
                return JobResponse.model_validate(response.json())
            return TextToImageResponse.model_validate(response.json())

    async def get_text_to_image(self, job_id: str, accept: Optional[str] = None) -> Union[JobResponse, TextToImageResponse, bytes]:
        """
        Retrieves the result of a text to image job.

        :param job_id: Job ID to retrieve
        :param accept: Optional accept header to specify response format
        :return: JobResponse, TextToImageResponse, or bytes (image data)
        """
        async with httpx.AsyncClient() as client:
            headers = self.headers.copy()
            if accept:
                headers["accept"] = accept

            response = await client.get(f"{API_URL}/v1/text-to-image/{job_id}", headers=headers)
            response.raise_for_status()
            if accept == "image/png":
                return response.content
            return TextToImageResponse.model_validate(response.json())

    async def list_worker_configurations(self) -> WorkerConfigs:
        """
        Lists the available worker configurations.

        :return: WorkerConfigs
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/v1/workers/list", headers=self.headers)
            response.raise_for_status()
            return WorkerConfigs.model_validate(response.json())

    async def get_gpu_infos(self) -> Union[GPUsInfo, FunctionCallError]:
        """
        Retrieves information about the GPUs.

        :return: GPUsInfo or FunctionCallError
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/v1/gpu/info", headers=self.headers)
            response.raise_for_status()
            return GPUsInfo.model_validate(response.json())

    async def text_to_text(self, request_data: ChatRequest, worker_id: str, async_key: Optional[str] = None,
                           keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None,
                           priority_key: Optional[str] = None) -> Union[JobResponse, ChatResponse]:
        """
        Generates text from text using the specified worker.

        :param request_data: Request data for text to text conversion
        :param worker_id: Worker ID to handle the request
        :param async_key: Optional async key for asynchronous processing
        :param keep_alive_key: Optional keep-alive duration
        :param storage_key: Optional storage key (local or S3)
        :param priority_key: Optional priority key (low, normal, high)
        :return: ChatResponse or JobResponse
        """
        async with httpx.AsyncClient() as client:
            headers = self.headers.copy()
            if async_key:
                headers["x-smi-async"] = async_key
            if keep_alive_key is not None:
                headers["x-smi-keep-alive"] = str(keep_alive_key)
            if storage_key:
                headers["x-smi-storage"] = storage_key
            if priority_key:
                headers["x-smi-priority"] = priority_key

            response = await client.post(
                f"{API_URL}/v1/text-to-text",
                headers=headers,
                json=request_data.model_dump(),
                params={"worker_id": worker_id}
            )
            response.raise_for_status()
            if async_key:
                return JobResponse.model_validate(response.json())
            return ChatResponse.model_validate(response.json())

    async def image_to_text(self, request_data: ChatRequest, worker_id: str, async_key: Optional[str] = None,
                            keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None,
                            priority_key: Optional[str] = None) -> Union[JobResponse, ChatResponse]:
        """
        Generates text from an image using the specified worker.

        :param request_data: Request data for image to text conversion
        :param worker_id: Worker ID to handle the request
        :param async_key: Optional async key for asynchronous processing
        :param keep_alive_key: Optional keep-alive duration
        :param storage_key: Optional storage key (local or S3)
        :param priority_key: Optional priority key (low, normal, high)
        :return: ChatResponse or JobResponse
        """
        async with httpx.AsyncClient() as client:
            headers = self.headers.copy()
            if async_key:
                headers["x-smi-async"] = async_key
            if keep_alive_key is not None:
                headers["x-smi-keep-alive"] = str(keep_alive_key)
            if storage_key:
                headers["x-smi-storage"] = storage_key
            if priority_key:
                headers["x-smi-priority"] = priority_key

            response = await client.post(
                f"{API_URL}/v1/image-to-text",
                headers=headers,
                json=request_data.model_dump(),
                params={"worker_id": worker_id}
            )
            response.raise_for_status()
            if async_key:
                return JobResponse.model_validate(response.json())
            return ChatResponse.model_validate(response.json())

    async def get_text_to_text(self, job_id: str, accept: Optional[str] = None) -> Union[JobResponse, ChatResponse]:
        """
        Retrieves the result of a text to text job.

        :param job_id: Job ID to retrieve
        :param accept: Optional accept header to specify response format
        :return: JobResponse or ChatResponse
        """
        async with httpx.AsyncClient() as client:
            headers = self.headers.copy()
            if accept:
                headers["accept"] = accept

            response = await client.get(f"{API_URL}/v1/text-to-text/{job_id}", headers=headers)
            response.raise_for_status()
            return ChatResponse.model_validate(response.json())

    async def text_to_embeddings(self, request_data: EmbeddingsRequest, worker_id: str, async_key: Optional[str] = None,
                                 keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None,
                                 priority_key: Optional[str] = None) -> Union[JobResponse, EmbeddingsResponse]:
        """
        Generates embeddings from text using the specified worker.

        :param request_data: Request data for text to embeddings conversion
        :param worker_id: Worker ID to handle the request
        :param async_key: Optional async key for asynchronous processing
        :param keep_alive_key: Optional keep-alive duration
        :param storage_key: Optional storage key (local or S3)
        :param priority_key: Optional priority key (low, normal, high)
        :return: EmbeddingsResponse or JobResponse
        """
        async with httpx.AsyncClient() as client:
            headers = self.headers.copy()
            if async_key:
                headers["x-smi-async"] = async_key
            if keep_alive_key is not None:
                headers["x-smi-keep-alive"] = str(keep_alive_key)
            if storage_key:
                headers["x-smi-storage"] = storage_key
            if priority_key:
                headers["x-smi-priority"] = priority_key

            response = await client.post(
                f"{API_URL}/v1/text-to-embeddings",
                headers=headers,
                json=request_data.model_dump(),
                params={"worker_id": worker_id}
            )
            response.raise_for_status()
            if async_key:
                return JobResponse.model_validate(response.json())
            return EmbeddingsResponse.model_validate(response.json())

    async def speech_to_text(self, file_path: str, worker_id: str, language: Optional[str] = None,
                             temperature: Optional[float] = 0.0, prompt: Optional[str] = None,
                             keep_alive: Optional[str] = None, async_key: Optional[str] = None,
                             keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None,
                             priority_key: Optional[str] = None) -> Union[JobResponse, SpeechToTextResponse]:
        """
        Converts speech to text using the specified worker.

        :param file_path: Path to the audio file
        :param worker_id: Worker ID to handle the request
        :param language: Optional language code
        :param temperature: Optional temperature setting
        :param prompt: Optional prompt
        :param keep_alive: Optional keep-alive setting
        :param async_key: Optional async key for asynchronous processing
        :param keep_alive_key: Optional keep-alive duration
        :param storage_key: Optional storage key (local or S3)
        :param priority_key: Optional priority key (low, normal, high)
        :return: SpeechToTextResponse or JobResponse
        """
        async with httpx.AsyncClient() as client:
            headers = self.headers.copy()
            if async_key:
                headers["x-smi-async"] = async_key
            if keep_alive_key is not None:
                headers["x-smi-keep-alive"] = str(keep_alive_key)
            if storage_key:
                headers["x-smi-storage"] = storage_key
            if priority_key:
                headers["x-smi-priority"] = priority_key

            files = {'file': open(file_path, 'rb')}
            data = {
                "worker_id": worker_id,
                "language": language,
                "temperature": temperature,
                "prompt": prompt,
                "keep_alive": keep_alive,
            }

            response = await client.post(
                f"{API_URL}/v1/speech-to-text",
                headers=headers,
                files=files,
                data=data
            )
            response.raise_for_status()
            if async_key:
                return JobResponse.model_validate(response.json())
            return SpeechToTextResponse.model_validate(response.json())

    async def text_to_speech(self, request_data: TextToSpeechRequest, worker_id: str, async_key: Optional[str] = None,
                             keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None,
                             priority_key: Optional[str] = None, accept: Optional[str] = None) -> Union[JobResponse, TextToSpeechResponse, bytes]:
        """
        Converts text to speech using the specified worker.

        :param request_data: Request data for text to speech conversion
        :param worker_id: Worker ID to handle the request
        :param async_key: Optional async key for asynchronous processing
        :param keep_alive_key: Optional keep-alive duration
        :param storage_key: Optional storage key (local or S3)
        :param priority_key: Optional priority key (low, normal, high)
        :param accept: Optional accept header to specify response format
        :return: TextToSpeechResponse, JobResponse, or bytes (audio data)
        """
        async with httpx.AsyncClient() as client:
            headers = self.headers.copy()
            if async_key:
                headers["x-smi-async"] = async_key
            if keep_alive_key is not None:
                headers["x-smi-keep-alive"] = str(keep_alive_key)
            if storage_key:
                headers["x-smi-storage"] = storage_key
            if priority_key:
                headers["x-smi-priority"] = priority_key
            if accept:
                headers["accept"] = accept

            response = await client.post(
                f"{API_URL}/v1/text-to-speech",
                headers=headers,
                json=request_data.model_dump(),
                params={"worker_id": worker_id}
            )
            response.raise_for_status()
            if accept == "audio/wav":
                return response.content
            if async_key:
                return JobResponse.model_validate(response.json())
            return TextToSpeechResponse.model_validate(response.json())

    async def speech_to_text(self, worker_id: str, timeout: int = 30, language:str="fr", rate:int = 48000, channels:int = 1) -> websockets.WebSocketClientProtocol:
        """
        Connects to the WebSocket and streams audio data for speech-to-text conversion.

        :param worker_id: Worker ID to handle the request
        :param file_path: Path to the audio file
        :param timeout: Timeout duration for the WebSocket connection
        :param callback: Optional callback function to handle each received JSON packet
        :return: AsyncGenerator yielding JSON responses
        """
        ws = await websockets.connect(f"{WS_API_URL}/ws/v1/speech-to-text?worker_id={worker_id}&timeout={timeout}", extra_headers={"x-smi-key": self.api_key})
        #ws.send(json.dumps({"language": language, "rate": rate, "channels": channels}).encode("utf-8"))
        return ws

async def check_routes():
    client = SMIClient(api_key="s78d8z6sdx-d058-4dd4-9c93-24761122aec5")
    response = await client.list_worker_configurations()

    stream = await client.speech_to_text(worker_id="FasterWhisperLarge", language="fr")
    audio_data = load_audio("../jfk.flac")
    #audio_data = load_audio("../test.wav")

    audio_bytes = audio_data.tobytes()

    chunk_size = 16384 * 5
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        await stream.send(chunk)
        response = await stream.recv()
        print(response)
        
# Example usage:
if __name__ == "__main__":
    client = SMIClient(api_key="s78d8z6sdx-d058-4dd4-9c93-24761122aec5")

    # Example call to a method
    asyncio.run(check_routes())

    # For the websocket
    #async def process_websocket():
    #    async for response in client.connect_websocket(worker_id="your_worker_id", file_path="path/to/your/file.wav"):
    #        print(response)

    #asyncio.run(process_websocket())
