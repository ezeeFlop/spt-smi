from spt.services.service import Worker, Service
from spt.utils import create_temp_file, remove_temp_file, get_available_device
from spt.models.llm import ChatRequest, ChatResponse, ChatMessage, EmbeddingsRequest, EmbeddingsResponse
from ollama import Client, ResponseError
from config import OLLAMA_URL
import inspect
import requests


class OllamaChat(Worker):
    def __init__(self, name: str, service: Service, model: str, logger):
        super().__init__(name=name, service=service, model=model, logger=logger)
        self.logger.info(f"Connecting to {OLLAMA_URL}")
        self.client = Client(host=OLLAMA_URL, timeout=500)
        self.models = []

    def __del__(self):
        self.logger.info("Claiming memory")
        self.cleanup()

    async def work(self, request: ChatRequest) -> ChatResponse:
        await super().work(request)

        self.logger.info(
            f"Generate Chat with {request.messages} model {self.model}")
        result = None
        try:
            result = self.client.chat(model=self.model,
                                      messages=[m.model_dump()
                                                for m in request.messages],
                                      options=request.options.model_dump(),
                                      keep_alive=self.service.get_keep_alive(),
                                      stream=request.stream,
                                      format=request.format)
            if self.model not in self.models:
                self.models.append(self.model)
        except ResponseError as e:
            if e.status_code == 404:
                self.client.pull(self.model)
            result = self.client.chat(model=self.model,
                                      messages=[m.model_dump()
                                                for m in request.messages],
                                      options=request.options.model_dump(),
                                      keep_alive=self.service.get_keep_alive(),
                                      stream=request.stream,
                                      format=request.format)

        lastItem = None

        if request.stream and inspect.isgenerator(result):
            self.logger.info(f"Stream detected...")
            for item in result:
                self.logger.info(item)

                response = ChatResponse(**item)
                lastItem = response
                self.service.chunked_request(response)
        else:
            lastItem = ChatResponse(**result)
        self.logger.info(f"Result: {result}")

        return lastItem

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
