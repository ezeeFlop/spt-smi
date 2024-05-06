from spt.models.llm import ChatRequest, ChatResponse, ChatMessage, EmbeddingsRequest, EmbeddingsResponse
from ollama import Client, ResponseError
from config import OLLAMA_URL
from spt.services.service import Service
from spt.services.generic.service import GenericServiceServicer
import inspect
import logging
import requests
logger = logging.getLogger(__name__)

class LLMModels(Service):

    def __init__(self, servicer: GenericServiceServicer = None) -> None:
        super().__init__(servicer=servicer)
        logger.info(f"Connecting to {OLLAMA_URL}")
        self.client = Client(host=OLLAMA_URL, timeout=500)
        self.models = []

    def __del__(self):
        logger.info("Claiming memory")
        self.cleanup()

    def cleanup(self):
        super().cleanup()
        for model in self.models:
            logger.info(f"Closing model {model}")
            payload = {
                "model": model,
                "keep_alive": 0
            }

            try:
                response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
                if response.status_code == 200:
                    logging.info("Ollama response : %s", response.json())
                else:
                    logging.error("Ollama API error: %s - %s",
                                response.status_code, response.text)

            except requests.exceptions.RequestException as e:
                logging.exception("Error during HTTP request: %s", e)


    def generate_chat(self, request: ChatRequest):
        logger.info(f"Generate Chat with {request.messages} model {request.model}")
        result = None
        try:
            result = self.client.chat(model=request.model, 
                                    messages=[m.model_dump() for m in request.messages], 
                                    options=request.options.model_dump(), 
                                    keep_alive=self.get_keep_alive(), 
                                    stream=request.stream, 
                                    format=request.format)
            if request.model not in self.models:
                self.models.append(request.model)
        except ResponseError as e:
            if e.status_code ==  404:
                self.client.pull(request.model)
            result = self.client.chat(model=request.model,
                                        messages=[m.model_dump()
                                                for m in request.messages],
                                        options=request.options.model_dump(),
                                      keep_alive=self.get_keep_alive(),
                                        stream=request.stream,
                                        format=request.format)


        lastItem = None

        if request.stream and inspect.isgenerator(result):
            logger.info(f"Stream detected...")
            for item in result:
                logger.info(item)

                response = ChatResponse(**item)
                lastItem = response
                self.chunked_request(response)
        else:
            lastItem = ChatResponse(**result)
        logger.info(f"Result: {result}")

        return lastItem

    def generate_embeddings(self, request: EmbeddingsRequest):
        logger.info(f"Generate Chat with {request.prompt}")
        result = self.client.embeddings(model=request.model, 
                                               prompt=request.prompt, 
                                               options=request.options.model_dump()
                                               )
        logger.info(f"Result: {result}")
        return EmbeddingsResponse(**result)

def main():
    llm = LLMModels()
    result = llm.generate_chat(ChatRequest(model="mistral", stream=True, messages=[ChatMessage(role="user", content="What is the meaning of life?")]))
    
    print(result)

 

if __name__ == '__main__':
    main()
