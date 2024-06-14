from spt.services.service import Worker, Service
import whisper
from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse, TextToSpeechRequest, TextToSpeechResponse, TextToSpeechSpeakerRequest
from spt.utils import create_temp_file, remove_temp_file, get_available_device

class Whisper(Worker):
    def __init__(self, name: str, service: Service, model:str, logger):
        super().__init__(name=name, service=service, model=model, logger=logger)
        self.audio_to_text_model = None

    async def work(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        await super().work(request)

        result = None
        if self.audio_to_text_model is None:
            self.audio_to_text_model = whisper.load_model(self.model)

        # load audio and pad/trim it to fit 30 seconds
        file = create_temp_file(request.file)
        audio = whisper.load_audio(file)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(
            self.audio_to_text_model.device)
        language = request.language

        if language is None:
            # detect the spoken language
            _, probs = self.audio_to_text_model.detect_language(mel)
            self.logger.info(f"Detected language: {max(probs, key=probs.get)}")
            language = max(probs, key=probs.get)

        # decode the audio
        options = whisper.DecodingOptions(
            temperature=request.temperature, prompt=request.prompt, language=language, )
        result = whisper.decode(self.audio_to_text_model, mel, options)

        response = SpeechToTextResponse(language=language, text=result.text)
        self.logger.info(f"Result: {response}")

        remove_temp_file(file)
        return response
        
    def cleanup(self):
        super().cleanup()
        if self.audio_to_text_model is not None:
                self.logger.info(f"Closing model {self.audio_to_text_model}")
                del self.audio_to_text_model.encoder
                del self.audio_to_text_model.decoder

