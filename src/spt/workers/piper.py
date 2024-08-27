from spt.models.audio import TextToSpeechRequest, TextToSpeechResponse
from spt.services.service import Service, Worker
from spt.utils import get_available_device
import torch
import numpy as np
import io
import wave
import base64
import nltk
from config import NLTK_PATH
import os
from piper import PiperVoice

os.environ["NLTK_DATA"] = NLTK_PATH


class Piper(Worker):
    def __init__(self, id: str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.tts_model = None
        self.sample_rate = 22050  # PiperVoice default sample rate

    def __del__(self):
        self.logger.info("Claiming memory")
        self.cleanup()

    def cleanup(self):
        super().cleanup()
        if self.tts_model is not None:
            self.logger.info(f"Closing model")
            del self.tts_model
        torch.cuda.empty_cache()

    def load_model(self, model_path: str, config_path: str):
        self.logger.info("Loading PiperVoice TTS model")
        self.tts_model = PiperVoice.load("../models/piper/fr/siwis/fr_FR-siwis-medium.onnx",
                                         "../models/piper/fr/siwis/fr_fr_FR_siwis_medium_fr_FR-siwis-medium.onnx.json")
        self.logger.info("PiperVoice TTS Loaded.")

    async def work(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        await super().work(request)

        self.logger.info(f"Text To Speech model {self.model}")
        if self.tts_model is None:
            model_path, config_path = self.model.split(',')
            self.load_model(model_path=model_path, config_path=config_path)

        sentences = nltk.sent_tokenize(request.text, language="french")

        audio_pieces = []
        for sentence in sentences:
            audio_array = self.tts_model.synthesize(sentence)
            audio_pieces.append(audio_array)

            # Add a short silence between sentences
            silence = np.zeros(int(0.25 * self.sample_rate), dtype=np.int16)
            audio_pieces.append(silence)

        # Concatenate all audio pieces
        wav = np.concatenate(audio_pieces)

        wav_file = self.encode_audio_common(wav.tobytes(), encode_base64=False)

        if self.service.should_store():
            url = self.service.store_bytes(
                bytes=wav_file, name=request.text, extension="wav")
            return TextToSpeechResponse(url=url)
        else:
            return TextToSpeechResponse(base64=self.encode_audio_common(wav_file))

    def encode_audio_common(self, frame_input, encode_base64=True, sample_width=2, channels=1):
        """Return base64 encoded audio"""
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as vfout:
            vfout.setnchannels(channels)
            vfout.setsampwidth(sample_width)
            vfout.setframerate(self.sample_rate)
            vfout.writeframes(frame_input)

        wav_buf.seek(0)
        if encode_base64:
            b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
            return b64_encoded
        else:
            return wav_buf.read()
