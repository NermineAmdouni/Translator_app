import pyaudio

class AudioConfig:
    """Audio configuration settings."""
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.VAD_AGGRESSIVENESS = 2
        self.vad_threshold = 0.5
        self.vad_enabled = True
        # Additional configuration
        self.MIN_AUDIO_CHUNKS = 15
        self.MAX_AUDIO_CHUNKS = 200
        self.SILENCE_THRESHOLD = 1.0  
        self.PROCESSING_DELAY= 2.0

class Languages:
    """Language configuration."""
    def __init__(self):
        self.languages = {
            "en": {
                "name": "English",
                "tts_voice": "af_heart",
                "kokoro_code": "a",
                "translation_models": {
                    "es": "Helsinki-NLP/opus-mt-en-es",
                    "fr": "Helsinki-NLP/opus-mt-en-fr"
                }
            },
            "es": {
                "name": "Spanish",
                "tts_voice": "ef_dora",
                "kokoro_code": "e",
                "translation_models": {
                    "en": "Helsinki-NLP/opus-mt-es-en",
                    "fr": "Helsinki-NLP/opus-mt-es-fr"
                }
            },
            "fr": {
                "name": "French",
                "tts_voice": "ff_siwis",
                "kokoro_code": "f",
                "translation_models": {
                    "en": "Helsinki-NLP/opus-mt-fr-en",
                    "es": "Helsinki-NLP/opus-mt-fr-es"
                }
            }
        }
    
    def get_languages(self):
        """Return the language configuration."""
        return self.languages