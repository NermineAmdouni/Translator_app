import pyaudio
import threading
import webrtcvad
import time
import queue
import numpy as np
from collections import deque  # Added for pre-buffering


class AudioRecorder:
    """Records audio with voice activity detection."""

    def __init__(self, audio_config):
        self.config = audio_config
        self.audio_interface = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(2)
        self.running = True
        self.speech_buffer = []
        self.speech_active = False
        self.last_voice_time = time.time()
        self.speech_start_time = None

    def start(self, audio_queue, running_flag):
        """Start recording audio with VAD."""
        self.running = running_flag

        stream = self.audio_interface.open(
            format=self.config.FORMAT,
            channels=self.config.CHANNELS,
            rate=self.config.RATE,
            input=True,
            frames_per_buffer=self.config.CHUNK_SIZE
        )

        print("\nListening for speech...")

        silence_chunks = 0
        SILENCE_THRESHOLD = 0.30
        max_silence_chunks = int(SILENCE_THRESHOLD * 1000 / self.config.CHUNK_DURATION_MS)

        # Pre-buffer to capture ~500ms of audio before speech
        pre_buffer = deque(maxlen=int(500 / self.config.CHUNK_DURATION_MS))

        while self.running:
            try:
                chunk = stream.read(self.config.CHUNK_SIZE, exception_on_overflow=False)
                pre_buffer.append(chunk)
                is_speech = self.vad.is_speech(chunk, self.config.RATE)

                if is_speech:
                    if not self.speech_active:
                        self.speech_start_time = time.time()
                        self.speech_active = True
                        # Include the audio just before speech
                        self.speech_buffer = list(pre_buffer)

                    self.speech_buffer.append(chunk)
                    silence_chunks = 0
                    self.last_voice_time = time.time()

                    if len(self.speech_buffer) >= self.config.MAX_AUDIO_CHUNKS:
                        audio_data = b''.join(self.speech_buffer)
                        audio_queue.put(audio_data)
                        conversion_time = time.time() - self.speech_start_time
                        print(f"Audio conversion took: {conversion_time:.2f}s")
                        self.speech_buffer = []
                        self.speech_active = False
                        self.speech_start_time = None

                elif self.speech_active:
                    silence_chunks += 1

                    if silence_chunks <= 1:  # Allow for short pauses
                        self.speech_buffer.append(chunk)

                    if silence_chunks >= max_silence_chunks:
                        if len(self.speech_buffer) >= self.config.MIN_AUDIO_CHUNKS:
                            audio_data = b''.join(self.speech_buffer)
                            audio_queue.put(audio_data)
                            conversion_time = time.time() - self.speech_start_time
                            print(f"Audio conversion took: {conversion_time:.2f}s")

                        self.speech_buffer = []
                        self.speech_active = False
                        self.speech_start_time = None

            except Exception as e:
                print(f"Recording error: {str(e)}")
                break

        stream.stop_stream()
        stream.close()

    def stop(self):
        """Stop audio recording."""
        self.running = False
        self.audio_interface.terminate()
