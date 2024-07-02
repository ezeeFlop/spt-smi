from IPython.display import Audio
from spt.models.audio import TextToSpeechRequest, TextToSpeechResponse, TextToSpeechSpeakerRequest
from spt.services.service import Service, Worker
from spt.services.service import GenericServiceServicer
import base64
import numpy as np
import torch
from spt.utils import create_temp_file, remove_temp_file, get_available_device
import os
import io
import wave
from transformers import AutoProcessor, BarkModel

class Bark(Worker):
    def __init__(self, id:str, name: str, service: Service, model: str, logger):
        super().__init__(id=id, name=name, service=service, model=model, logger=logger)
        self.tts_model = None
        self.processor = None

    def __del__(self):
        self.logger.info("Claiming memory")
        self.cleanup()

    def cleanup(self):
        super().cleanup()
        if self.tts_model is not None:
            self.logger.info(f"Closing model")
            del self.tts_model
            del self.processor
        torch.cuda.empty_cache()


    def load_model(self, model_name: str):
        self.logger.info("Loading Bark models")
        device = get_available_device()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tts_model = BarkModel.from_pretrained(
            model_name).to("cpu" if device == "mps" else device)
        self.logger.info("Bark Loaded.")

    async def work(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        await super().work(request)

        self.logger.info(f"Text To Speech model {self.model}")
        if self.tts_model is None:
            self.load_model(model_name=self.model)
        device = get_available_device()

        voice_preset = request.speaker_id
        inputs = self.processor(request.text, voice_preset=voice_preset, return_tensors="pt").to("cpu" if device == "mps" else device)
        audio_array = self.tts_model.generate(**inputs)

        wav = self.postprocess(audio_array)
        wav_file = self.encode_audio_common(
            wav.tobytes(), encode_base64=False)

        if self.service.should_store():
            url = self.service.store_bytes(
                bytes=wav_file, name=request.text, extension="wav")
            return TextToSpeechResponse(url=url)
        else:
            return TextToSpeechResponse(base64=self.encode_audio_common(wav_file))

    def encode_audio_common(self, frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1):
        """Return base64 encoded audio"""
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as vfout:
            vfout.setnchannels(channels)
            vfout.setsampwidth(sample_width)
            vfout.setframerate(sample_rate)
            vfout.writeframes(frame_input)

        wav_buf.seek(0)
        if encode_base64:
            b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
            return b64_encoded
        else:
            return wav_buf.read()

    def postprocess(self, wav) -> np.ndarray:

        wav = wav.clone().detach().cpu().numpy()
        wav = wav[None, : int(wav.shape[0])]
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)

        return wav

