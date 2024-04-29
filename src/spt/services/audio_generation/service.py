from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse, TextToSpeechRequest, TextToSpeechResponse, TextToSpeechSpeakerRequest
from spt.services.service import Service
from spt.services.generic.service import GenericServiceServicer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import base64
import numpy as np
import inspect
import logging
import whisper
import torch
from spt.utils import create_temp_file, remove_temp_file
import os
import io
import wave

from spt.services.gpu import get_available_device
logger = logging.getLogger(__name__)


class TTSService(Service):

    def __init__(self, servicer: GenericServiceServicer = None) -> None:
        super().__init__(servicer=servicer)
        self.tts_model = None

    def __del__(self):
        logger.info("Claiming memory")
        self.cleanup()
    
    def cleanup(self):
        super().cleanup()
        if self.tts_model is not None:
            logger.info(f"Closing model")
            del self.tts_model
            torch.cuda.empty_cache()

    def load_model(self, model_name: str):
        torch.set_num_threads(os.cpu_count())
        device = get_available_device()
        logger.info(f"Downloading XTTS Model: {model_name}")
        ModelManager().download_model(model_name)
        model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
        logger.info("XTTS Model downloaded")

        logger.info("Loading XTTS")
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.tts_model = Xtts.init_from_config(config)
        self.tts_model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if device == "cuda" else False)
        self.tts_model.to(device)
        logger.info("XTTS Loaded.")

    def speech_to_text(self, request: TextToSpeechRequest):
        logger.info(f"Text To Speech model {request.model}")
        result = None
        if self.tts_model is None:
            self.load_model(model_name=request.model)
  
        speaker = self.get_speaker(request.speaker_id)
        if speaker is None:
            raise Exception("Speaker not found")
        speaker_embedding = torch.tensor(speaker["speaker_embedding"]).unsqueeze(0).unsqueeze(-1)
        gpt_cond_latent = torch.tensor(speaker["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0)
        text = request.text
        language = request.language

        out = self.tts_model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
        )

        wav = self.postprocess(torch.tensor(out["wav"]))

        wav_file = self.encode_audio_common(wav.tobytes(), encode_base64=False)

        if self.should_store():
            url = self.store_bytes(bytes=wav_file, name=request.text, extension="wav")
            return TextToSpeechResponse(url=url)
        else:
            return TextToSpeechResponse(wav=self.encode_audio_common(wav_file))

    def add_speakers(self, request: TextToSpeechSpeakerRequest):
        pass

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
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = wav[None, : int(wav.shape[0])]
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)

        return wav

    def get_speaker(self, speaker_id: str):
        if hasattr(self.tts_model, "speaker_manager") and hasattr(self.tts_model.speaker_manager, "speakers"):
            speaker = self.tts_model.speaker_manager.speakers[speaker_id]
            return {
                "speaker_embedding": speaker["speaker_embedding"].cpu().squeeze().half().tolist(),
                "gpt_cond_latent": speaker["gpt_cond_latent"].cpu().squeeze().half().tolist(),
            }
        else:
            return None
    def get_speakers(self):
        if hasattr(self.tts_model, "speaker_manager") and hasattr(self.tts_model.speaker_manager, "speakers"):
            return list(self.tts_model.speaker_manager.speakers.keys())
            return {
                speaker: {
                    "speaker_embedding": self.tts_model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                    "gpt_cond_latent": self.tts_model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
                }
                for speaker in self.tts_model.speaker_manager.speakers.keys()
            }
        else:
            return {}

class STTService(Service):

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

    def speech_to_text(self, request: SpeechToTextRequest):
        logger.info(f"Speech To Text model {request.model}")
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
        language = request.language

        if language is None:
            # detect the spoken language
            _, probs = self.audio_to_text_model.detect_language(mel)
            logger.info(f"Detected language: {max(probs, key=probs.get)}")
            language = max(probs, key=probs.get)

        # decode the audio
        options = whisper.DecodingOptions(
            temperature=request.temperature, prompt=request.prompt, language=language, )
        result = whisper.decode(self.audio_to_text_model, mel, options)

        response = SpeechToTextResponse(language=language, text=result.text)
        logger.info(f"Result: {response}")

        remove_temp_file(file)

        return response

def main():
    #stt = STTService()
    #result = stt.speech_to_text(SpeechToTextResponse(model="base", audio="audio.mp3"))
    
    tts = TTSService()
    result = tts.speech_to_text(TextToSpeechRequest(model="tts_models/multilingual/multi-dataset/xtts_v2", text="Hello, World!", language="en", speaker_id="Daisy Studious"))
    print(result)

if __name__ == '__main__':
    main()
