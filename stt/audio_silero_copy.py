import pyaudio
import asyncio
import threading
import torch
from stt.vad import AudioStreamProcessor  # assuming you have this vad package

class AudioRecorder:
    """Records audio using AudioStreamProcessor from silero-vad wrapper with callbacks."""

    def __init__(self, audio_config, message_queue, loop):
        self.config = audio_config
        self.message_queue = message_queue
        self.loop = loop  # asyncio event loop

        # Load VAD model and utils once
        self.vad_model, self.vad_utils = torch.hub.load(
            'snakers4/silero-vad', model='silero_vad', force_reload=False
        )

        # Define callbacks
        def on_speech_start():
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({
                    "type": "vad_status",
                    "status": "speech_started",
                    "should_interrupt": False,
                }),
                self.loop,
            )

        def on_speech_end(audio_data, sample_rate):
            try:
                print("Transcription starting")
                # You can call your transcription function here:
                # user_text = transcribe_audio(audio_data, sample_rate)
                # print(f"Transcription completed: '{user_text}'")
                # Or put the audio in a queue for later processing
            except Exception as e:
                print(f"VAD callback failed: {e}")

        # Initialize AudioStreamProcessor with your config and callbacks
        self.vad_processor = AudioStreamProcessor(
            model=self.vad_model,
            utils=self.vad_utils,
            sample_rate=self.config.RATE,
            vad_threshold=self.config.vad_threshold,
            callbacks={
                "on_speech_start": on_speech_start,
                "on_speech_end": on_speech_end,
            },
        )

        # PyAudio interface for audio capture
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.running = False

    def start(self):
        """Start the microphone stream and processing loop."""
        self.running = True
        self.stream = self.audio_interface.open(
            format=self.config.FORMAT,
            channels=self.config.CHANNELS,
            rate=self.config.RATE,
            input=True,
            frames_per_buffer=self.config.CHUNK_SIZE,
            input_device_index=1,  # adjust as needed
        )

        print("\nListening for speech (AudioStreamProcessor)...")

        while self.running:
            try:
                audio_chunk = self.stream.read(self.config.CHUNK_SIZE, exception_on_overflow=False)
                self.vad_processor.process_audio_chunk(audio_chunk)
            except Exception as e:
                print(f"Error reading audio stream: {e}")
                break

        self.stream.stop_stream()
        self.stream.close()
        self.audio_interface.terminate()

    def stop(self):
        """Stop the audio processing."""
        self.running = False
