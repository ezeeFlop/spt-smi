from spt.models.audio import AudioToTextRequest, AudioToTextResponse
from spt.services.service import Service
from spt.services.generic.service import GenericServiceServicer
import inspect
import logging
import whisper
import torch
from spt.utils import create_temp_file, remove_temp_file

logger = logging.getLogger(__name__)

class AudioModels(Service):

    def __init__(self, servicer: GenericServiceServicer = None) -> None:
        super().__init__(servicer=servicer)
        self.audio_to_text_model = None

    def __del__(self):
        logger.info("Claiming memory")
        self.cleanup()

    def cleanup(self):
        super().cleanup()
        if self.audio_to_text_model is not None:
            logger.info(f"Closing model {self.audio_to_text_model}")
            del self.audio_to_text_model.encoder
            del self.audio_to_text_model.decoder
            torch.cuda.empty_cache()

    def audio_to_text(self, request: AudioToTextRequest):
        logger.info(f"Generate Text model {request.model}")
        result = None
        if self.audio_to_text_model is None:
            self.audio_to_text_model = whisper.load_model(request.model)

        # load audio and pad/trim it to fit 30 seconds
        file = create_temp_file(request.file)
        audio = whisper.load_audio(file)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(
            self.audio_to_text_model.device)

        # detect the spoken language
        _, probs = self.audio_to_text_model.detect_language(mel)
        logger.info(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.audio_to_text_model, mel, options)

        response = AudioToTextResponse(language=result.language, text=result.text)
        logger.info(f"Result: {response}")

        remove_temp_file(file)

        return response

def main():
    audioModel = AudioModels()
    result = audioModel.audio_to_text(AudioToTextRequest(model="base", audio="audio.mp3"))
    
    print(result)

if __name__ == '__main__':
    main()
