import torch
import logging

from spt.models.llm import ChatRequest, ChatResponse, ChatMessage
from ollama import Client

from config import OLLAMA_URL

logger = logging.getLogger(__name__)

class LLMModels:

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        self.client = Client(host=OLLAMA_URL)


    def generate_chat(self, request: ChatRequest):
        logger.info(f"Generate Chat with {request}")
        return ChatResponse(model="GPT-3.5", 
                            done=True, 
                            created_at="2022-08-01T00:00:00Z",
                            total_duration=123, 
                            load_duration=123, 
                            prompt_eval_count=123, 
                            prompt_eval_duration=123, 
                            eval_count=123, 
                            eval_duration=123, 
                            message=ChatMessage(role="user", content="Hello, World!")
                            )
    @classmethod
    def memory_usage(cls):
        max_memory = round(torch.cuda.max_memory_allocated(
            device='cuda') / 1000000000, 2)
        return max_memory

def main():
    pass
if __name__ == '__main__':
    main()
