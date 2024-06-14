from faster_whisper import WhisperModel
from spt.services.service import Worker, Service
from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse
from spt.utils import create_temp_file, remove_temp_file, get_available_device
import torch
from typing import Dict, Tuple, Any, Optional, Union


class FasterWhisper(Worker):

    def __init__(self, name: str, service: Service, model: str, logger):
        super().__init__(name=name, service=service, model=model, logger=logger)
        self.model_instance = None

    async def work(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        await super().work(request)

        file = create_temp_file(request.file)

        self.model_instance = WhisperModel(
            self.model, device=get_available_device(), compute_type="float16")

        segments, info = self.model_instance.transcribe(
            file, beam_size=1)

        response = SpeechToTextResponse(
            language=request.language, text=segments["text"])
        
        self.logger.info(f"Result: {response}")

        remove_temp_file(file)

        return response

    async def stream(self, data: Union[bytes | str | Dict[str, Any]]) -> Union[bytes | str | Dict[str, Any]]:
        return f"data ---> {data}"

    def cleanup(self):
        super().cleanup()
        if self.model_instance is not None:
            self.logger.info(f"Closing model {self.model}")
            del self.model_instance
            torch.cuda.empty_cache()
