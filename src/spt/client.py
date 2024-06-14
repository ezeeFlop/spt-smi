import requests
from typing import Optional, Type, Union
from pydantic import BaseModel, ValidationError
from spt.models.jobs import JobResponse, TextToImageResponse, JobStatuses
from spt.models.image import TextToImageRequest
from spt.models.llm import ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse
from spt.models.audio import SpeechToTextRequest, TextToSpeechRequest, SpeechToTextResponse, TextToSpeechResponse
from spt.utils import GPUsInfo

class SMIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "x-smi-key": self.api_key,
            "Content-Type": "application/json"
        }

    def _post(self, endpoint: str, data: dict, headers: Optional[dict] = None):
        full_headers = {**self.headers, **headers} if headers else self.headers
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, headers=full_headers, json=data)
        return response.json()

    def _get(self, endpoint: str, headers: Optional[dict] = None):
        full_headers = {**self.headers, **headers} if headers else self.headers
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=full_headers)
        return response.json()

    def _validate_response(self, response: dict, response_model: Type[BaseModel]):
        try:
            return response_model(**response)
        except ValidationError as e:
            print("Validation error in response:", e.json())
            return None

    # Text-to-Image Routes
    def text_to_image(self, request_data: TextToImageRequest, async_key: Optional[str] = None,
                      keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None, 
                      priority_key: Optional[str] = None) -> Union[JobResponse, TextToImageResponse, None]:
        headers = {
            "x-smi-async": async_key,
            "x-smi-keep-alive": str(keep_alive_key) if keep_alive_key else None,
            "x-smi-storage": storage_key,
            "x-smi-priority": priority_key
        }
        data = request_data.dict()
        response = self._post("/v1/text-to-image", data, headers)
        return self._validate_response(response, Union[JobResponse, TextToImageResponse])

    def retrieve_image_job(self, job_id: str) -> Union[JobResponse, TextToImageResponse, None]:
        endpoint = f"/v1/text-to-image/{job_id}"
        response = self._get(endpoint)
        return self._validate_response(response, Union[JobResponse, TextToImageResponse])

    # Chat Generation Routes
    def generate_chat(self, request_data: ChatRequest, async_key: Optional[str] = None,
                      keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None, 
                      priority_key: Optional[str] = None) -> Union[ChatResponse, JobResponse, None]:
        headers = {
            "x-smi-async": async_key,
            "x-smi-keep-alive": str(keep_alive_key) if keep_alive_key else None,
            "x-smi-storage": storage_key,
            "x-smi-priority": priority_key
        }
        data = request_data.dict()
        response = self._post("/v1/chat", data, headers)
        return self._validate_response(response, Union[ChatResponse, JobResponse])

    def retrieve_chat_job(self, job_id: str) -> Union[ChatResponse, JobResponse, None]:
        endpoint = f"/v1/chat/{job_id}"
        response = self._get(endpoint)
        return self._validate_response(response, Union[ChatResponse, JobResponse])

    # Embeddings Generation Routes
    def generate_embeddings(self, request_data: EmbeddingsRequest, async_key: Optional[str] = None,
                            keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None, 
                            priority_key: Optional[str] = None) -> Union[EmbeddingsResponse, JobResponse, None]:
        headers = {
            "x-smi-async": async_key,
            "x-smi-keep-alive": str(keep_alive_key) if keep_alive_key else None,
            "x-smi-storage": storage_key,
            "x-smi-priority": priority_key
        }
        data = request_data.dict()
        response = self._post("/v1/embeddings", data, headers)
        return self._validate_response(response, Union[EmbeddingsResponse, JobResponse])

    # Speech-to-Text Routes
    def speech_to_text(self, request_data: SpeechToTextRequest, async_key: Optional[str] = None,
                       keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None, 
                       priority_key: Optional[str] = None) -> Union[SpeechToTextResponse, JobResponse, None]:
        headers = {
            "x-smi-async": async_key,
            "x-smi-keep-alive": str(keep_alive_key) if keep_alive_key else None,
            "x-smi-storage": storage_key,
            "x-smi-priority": priority_key
        }
        data = request_data.dict()
        response = self._post("/v1/speech-to-text", data, headers)
        return self._validate_response(response, Union[SpeechToTextResponse, JobResponse])

    def text_to_speech(self, request_data: TextToSpeechRequest, async_key: Optional[str] = None,
                       keep_alive_key: Optional[int] = None, storage_key: Optional[str] = None, 
                       priority_key: Optional[str] = None) -> Union[TextToSpeechResponse, JobResponse, None]:
        headers = {
            "x-smi-async": async_key,
            "x-smi-keep-alive": str(keep_alive_key) if keep_alive_key else None,
            "x-smi-storage": storage_key,
            "x-smi-priority": priority_key
        }
        data = request_data.dict()
        response = self._post("/v1/text-to-speech", data, headers)
        return self._validate_response(response, Union[TextToSpeechResponse, JobResponse])

    # List Engines and GPU Info Routes
    def list_engines(self) -> Union[EnginesList, None]:
        response = self._get("/v1/engines/list")
        return self._validate_response(response, EnginesList)

    def gpu_infos(self) -> Union[GPUsInfo, None]:
        response = self._get("/v1/gpu/info")
        return self._validate_response(response, GPUsInfo)

# Utilisation de la classe
client = SMIClient(base_url="http://your-api-url.com", api_key="your-api-key")

# Exemple d'appel
try:
    request_data = TextToImageRequest(model="your-model-id", prompt="Describe your image here")
    response = client.text_to_image(request_data, async_key="true", keep_alive_key=300, storage_key="S3", priority_key="high")
    print(response)
except ValidationError as e:
    print("Error while creating TextToImageRequest:", e.json())
