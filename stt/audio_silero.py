import pyaudio
import threading
import time
import queue
import numpy as np
import torch
from collections import deque

class AudioRecorder:
    """Records audio with Silero-VAD voice activity detection."""

    def __init__(self, audio_config):
        self.config = audio_config
        
        # Initialize Silero-VAD
        torch.set_num_threads(1)
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                         model='silero_vad',
                                         force_reload=True)
        (self.get_speech_timestamps, _, _, *_) = utils
        
        # Audio interface
        self.audio_interface = pyaudio.PyAudio()
        
        # State variables
        self.running = True
        self.speech_buffer = []
        self.speech_active = False
        self.last_voice_time = time.time()
        self.speech_start_time = None
        
        # Audio processing buffer (needs to be large enough for Silero-VAD analysis)
        self.analysis_window_size = int(1.5 * self.config.RATE)  # 1.5 second window
        self.audio_buffer = deque(maxlen=self.analysis_window_size * 2)  # Double for overlap
        
        # Hysteresis counters for more stable detection
        self.speech_counter = 0
        self.NO_SPEECH_FRAMES_TO_SILENCE = 3  # ~1.5s of no speech
        self.SPEECH_FRAMES_TO_ACTIVATE = 2    # ~1s of speech

    def _process_audio_window(self, audio_data):
        """Process audio window with Silero-VAD."""
        # Convert to numpy array and then to torch tensor
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_tensor = torch.from_numpy(audio_np)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor, self.model, sampling_rate=self.config.RATE)
        
        return len(speech_timestamps) > 0

    def start(self, audio_queue, running_flag):
        """Start recording audio with Silero-VAD."""
        self.running = running_flag

        stream = self.audio_interface.open(
            format=self.config.FORMAT,
            channels=self.config.CHANNELS,
            rate=self.config.RATE,
            input=True,
            input_device_index=1, 
            frames_per_buffer=self.config.CHUNK_SIZE
        )

        print("\nListening for speech (Silero-VAD)...")

        silence_chunks = 0
        SILENCE_THRESHOLD = 0.30
        max_silence_chunks = int(SILENCE_THRESHOLD * 1000 / self.config.CHUNK_DURATION_MS)

        # Pre-buffer to capture audio before speech
        pre_buffer = deque(maxlen=int(500 / self.config.CHUNK_DURATION_MS))

        while self.running:
            try:
                chunk = stream.read(self.config.CHUNK_SIZE, exception_on_overflow=False)
                
                # Store in buffers
                self.audio_buffer.extend(np.frombuffer(chunk, dtype=np.int16))
                pre_buffer.append(chunk)
                
                # Only process when we have enough data for analysis
                if len(self.audio_buffer) >= self.analysis_window_size:
                    # Get the most recent window of audio
                    analysis_window = np.array(self.audio_buffer)[-self.analysis_window_size:]
                    analysis_data = analysis_window.tobytes()
                    
                    # Check for speech using Silero-VAD
                    is_speech = self._process_audio_window(analysis_data)
                    
                    # Update speech state with hysteresis
                    if is_speech:
                        self.speech_counter = min(self.speech_counter + 1, self.SPEECH_FRAMES_TO_ACTIVATE)
                    else:
                        self.speech_counter = max(self.speech_counter - 1, -self.NO_SPEECH_FRAMES_TO_SILENCE)
                    
                    # Determine new state
                    new_state = self.speech_active
                    if self.speech_counter >= self.SPEECH_FRAMES_TO_ACTIVATE:
                        new_state = True
                    elif self.speech_counter <= -self.NO_SPEECH_FRAMES_TO_SILENCE:
                        new_state = False
                    
                    # Handle state changes
                    if new_state and not self.speech_active:
                        # Speech started
                        self.speech_active = True
                        self.speech_start_time = time.time()
                        self.last_voice_time = time.time()
                        self.speech_buffer = list(pre_buffer)  # Include pre-buffered audio
                        #print("\nSpeech detected!")
                    
                    elif not new_state and self.speech_active:
                        # Speech ended
                        if len(self.speech_buffer) >= self.config.MIN_AUDIO_CHUNKS:
                            audio_data = b''.join(self.speech_buffer)
                            audio_queue.put(audio_data)
                            conversion_time = time.time() - self.speech_start_time
                            #print(f"Audio segment processed in {conversion_time:.2f}s")
                        
                        self.speech_buffer = []
                        self.speech_active = False
                        self.speech_start_time = None
                        #print("Silence detected")
                    
                    # If speech is active, keep adding to buffer
                    if self.speech_active:
                        self.speech_buffer.append(chunk)
                        self.last_voice_time = time.time()
                        
                        # Check if we've reached maximum recording length
                        if len(self.speech_buffer) >= self.config.MAX_AUDIO_CHUNKS:
                            audio_data = b''.join(self.speech_buffer)
                            audio_queue.put(audio_data)
                            conversion_time = time.time() - self.speech_start_time
                            #print(f"Max length reached, processed in {conversion_time:.2f}s")
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