import torch
import logging

from spt.models.llm import ChatRequest, ChatResponse, ChatMessage

logger = logging.getLogger(__name__)

class LLMModels:

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

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
