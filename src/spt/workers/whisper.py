from faster_whisper import WhisperModel
from spt.services.service import Worker, Service
from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse
from spt.utils import create_temp_file, remove_temp_file, get_available_device
from transformers import pipeline
import torch

class Whisper(Worker):

    def __init__(self, id:str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.pipe = None

    async def work(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        await super().work(request)

        file = create_temp_file(request.file)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            torch_dtype=torch.float16,
            device=get_available_device(),
            model_kwargs={"attn_implementation": "sdpa"}
        )
        language = request.language

        generate_kwargs = {"task": "transcribe", "language": language}

        outputs = self.pipe(
            file,
            chunk_length_s=30,
            batch_size=24,
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
        )
        response = SpeechToTextResponse(
            language=language, text=outputs["text"])
        self.logger.info(f"Result: {response}")

        remove_temp_file(file)

        return response

    def cleanup(self):
        super().cleanup()
        if self.pipe is not None:
            self.logger.info(f"Closing model {self.model}")
            del self.pipe
            torch.cuda.empty_cache()
