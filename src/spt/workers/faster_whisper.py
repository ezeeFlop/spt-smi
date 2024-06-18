from faster_whisper import WhisperModel
from spt.services.service import Worker, Service
from spt.models.audio import SpeechToTextRequest, SpeechToTextResponse
from spt.utils import create_temp_file, remove_temp_file, get_available_device
from spt.workers.utils.vad import VoiceActivityDetector
import torch
from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import soundfile
import librosa 
import io
import logging
import json
import time 

TARGET_SAMPLE_RATE = 16000  # Target sample rate for the model
TARGET_CHANNELS = 1  # Target sample rate for the model

class FasterWhisper(Worker):

    def __init__(self, name: str, service: Service, model: str, logger):
        super().__init__(name=name, service=service, model=model, logger=logger)
        self.model_instance = None
        self.vad_detector = None
        self.original_language = None
        self.no_voice_activity_chunks = 0
        self.min_chunk = 1
        self.eos: bool = False
        self.timestamp_offset = 0.0
        self.text = []
        self.current_out = ''
        self.prev_out = ''
        self.t_start = None
        self.same_output_threshold = 0
        self.show_prev_out_thresh = 5
        self.transcript = []
        self.send_last_n_segments = 10
        # text formatting
        self.pick_previous_segments = 2

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

    def receive_audio_chunk(self, raw_bytes: bytes):
        out = []
        while sum(len(x) for x in out) < self.min_chunk*TARGET_SAMPLE_RATE:
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE",
                                     samplerate=TARGET_SAMPLE_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf, sr=TARGET_SAMPLE_RATE)
            out.append(audio)
        if not out:
            return None, 0.0
        input_bytes = np.concatenate(out)
        duration = input_bytes.shape[0] / TARGET_SAMPLE_RATE

        return input_bytes, duration

    def transcribe(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        segments, info = self.model_instance.transcribe(audio_chunk, language=self.original_language, initial_prompt="",
                                                        beam_size=5, word_timestamps=True, condition_on_previous_text=True)
        self.logger.info("Language: %s", info.language)
        return list(segments)

    async def stream(self, data: Union[bytes | str | Dict[str, Any]]) -> Union[bytes | str | Dict[str, Any]]:
        if self.model_instance is None: # first data also received
            self.logger.info(f"Initializing model {self.model}")
            device = "cpu" #if get_available_device() == "mps" else get_available_device()
            self.model_instance = WhisperModel(
                self.model) #, device="cpu", compute_type="int8")

            #self.model_instance = WhisperModel(self.model)
                #device=device,
                #compute_type="int8" if device == "cpu" else "float16",
                #local_files_only=False)
            self.logger.info(f"Model {self.model} initialized")
            self.vad_detector = VoiceActivityDetector(
                frame_rate=TARGET_SAMPLE_RATE)

        audio_chunk, duration = self.receive_audio_chunk(data)
        
        self.voice_activity(audio_chunk)

        self.logger.info(
            f"Detector state: {self.eos} chunk length: {len(audio_chunk)}")

        result = self.transcribe(audio_chunk)
        return self.handle_transcription_output(result, duration)

    def cleanup(self):
        super().cleanup()
        if self.model_instance is not None:
            self.logger.info(f"Closing model {self.model}")
            del self.model_instance
            torch.cuda.empty_cache()

    def voice_activity(self, frame_np):
        if not self.vad_detector(frame_np):
            self.logger.info(
                "No voice activity detected. Setting EOS flag.")
            self.eos = True
            return False
        self.eos = False
        return True

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.

        This method formats the transcription segments into a JSON object and attempts to send
        this object to the client. If an error occurs during the send operation, it logs the error.

        Returns:
            segments (list): A list of transcription segments to be sent to the client.
        """
        return json.dumps({
                "segments": segments,
                "is_final": self.eos
            })

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (str): The result from whisper inference i.e. the list of segments.
            duration (float): Duration of the transcribed audio chunk.
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)
 
        return self.send_transcription_to_client(segments)

    def format_segment(self, start, end, text):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places and the 'text'
                of the transcription.
        """
        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text
        }

    def update_segments(self, segments, duration):
        """
        Processes the segments from whisper. Appends all the segments to the list
        except for the last segment assuming that it is incomplete.

        Updates the ongoing transcript with transcribed segments, including their start and end times.
        Complete segments are appended to the transcript in chronological order. Incomplete segments
        (assumed to be the last one) are processed to identify repeated content. If the same incomplete
        segment is seen multiple times, it updates the offset and appends the segment to the transcript.
        A threshold is used to detect repeated content and ensure it is only included once in the transcript.
        The timestamp offset is updated based on the duration of processed segments. The method returns the
        last processed segment, allowing it to be sent to the client for real-time updates.

        Args:
            segments(dict) : dictionary of segments as returned by whisper
            duration(float): duration of the current chunk

        Returns:
            dict or None: The last processed segment with its start time, end time, and transcribed text.
                     Returns None if there are no valid segments to process.
        """
        offset = None
        self.current_out = ''
        # process complete segments
        if len(segments) > 1:
            for i, s in enumerate(segments[:-1]):
                text_ = s.text
                self.text.append(text_)
                start, end = self.timestamp_offset + s.start, self.timestamp_offset + min(duration, s.end)

                if start >= end:
                    continue

                self.transcript.append(self.format_segment(start, end, text_))
                offset = min(duration, s.end)

        self.current_out += segments[-1].text
        last_segment = self.format_segment(
            self.timestamp_offset + segments[-1].start,
            self.timestamp_offset + min(duration, segments[-1].end),
            self.current_out
        )

        # if same incomplete segment is seen multiple times then update the offset
        # and append the segment to the list
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_threshold += 1
        else:
            self.same_output_threshold = 0

        if self.same_output_threshold > 5:
            if not len(self.text) or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                self.transcript.append(self.format_segment(
                    self.timestamp_offset,
                    self.timestamp_offset + duration,
                    self.current_out
                ))
            self.current_out = ''
            offset = duration
            self.same_output_threshold = 0
            last_segment = None
        else:
            self.prev_out = self.current_out

        # update offset
        if offset is not None:
            self.timestamp_offset += offset

        return last_segment
    
    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles the recent segments of transcribed text, ensuring that only the
        specified number of the most recent segments are included. It also appends the most
        recent segment of text if provided (which is considered incomplete because of the possibility
        of the last word being truncated in the audio chunk).

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments
