from spt.services.service import Worker, Service
from spt.utils import create_temp_file, remove_temp_file, get_available_device
from spt.models.llm import ChatRequest, ChatResponse, ChatMessage, EmbeddingsRequest, EmbeddingsResponse
from ollama import Client, ResponseError
from config import OLLAMA_URL
import inspect
import requests


class OllamaEmbeddings(Worker):
    def __init__(self, id:str, name: str, service: Service, model: str, logger):
        super().__init__(name=name, id=id, service=service, model=model, logger=logger)
        self.logger.info(f"Connecting to {OLLAMA_URL}")
        self.client = Client(host=OLLAMA_URL, timeout=500)
        self.models = []

    def __del__(self):
        self.logger.info("Claiming memory")
        self.cleanup()

    async def work(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        await super().work(request)
        self.logger.info(f"Generate Chat with {request.prompt}")
        result = self.client.embeddings(model=self.model,
                                        prompt=request.prompt,
                                        options=request.options.model_dump()
                                        )
        if self.model not in self.models:
            self.models.append(self.model)
        self.logger.info(f"Result: {result}")
        return EmbeddingsResponse(**result)

    def cleanup(self):
        super().cleanup()
        for model in self.models:
            self.logger.info(f"Closing model {model}")
            payload = {
                "model": model,
                "keep_alive": 0
            }

            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate", json=payload)
                if response.status_code == 200:
                    self.logger.info("Ollama response : %s", response.json())
                else:
                    self.logger.error("Ollama API error: %s - %s",
                                  response.status_code, response.text)

            except requests.exceptions.RequestException as e:
                self.logger.exception("Error during HTTP request: %s", e)
