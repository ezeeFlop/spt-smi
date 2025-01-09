from sentence_transformers import SentenceTransformer
import torch
from spt.models.llm import EmbeddingsRequest, EmbeddingsResponse
import gc
from spt.services.service import Worker, Service
from pydantic import BaseModel
from typing import Union, Dict, Any

class SentencesTransformer(Worker):
    def __init__(self, id:str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.my_model = None

    async def work(self, request: EmbeddingsRequest) -> BaseModel:
        await super().work(request)
        if self.my_model is None:
            self.logger.warning(f"Loading model {self.model}")
            self.my_model = SentenceTransformer(self.model, model_kwargs={"torch_dtype": torch.float16})
        
        embeddings = []
        for text in request.text:
            embeddings.append(self.my_model.encode(text))

        return EmbeddingsResponse(embeddings=embeddings)

    async def stream(self, data: Union[bytes | str | Dict[str, Any]]) -> Union[bytes | str | Dict[str, Any]]:
        # do something with data

        return data

    def cleanup(self):
        super().cleanup()
        if self.my_model is not None:
            self.logger.info(f"Closing model {self.my_model}")
            del self.my_model
        torch.cuda.empty_cache()
        gc.collect()
