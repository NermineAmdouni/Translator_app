import numpy as np
import torch
from faster_whisper import WhisperModel
import time
import re

class WhisperTranscriber:
    """Transcribes audio to text using Whisper."""
    def __init__(self):
        print("Loading multilingual Whisper model...")
        self.model = WhisperModel(
            "small",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )

        # Language code mapping
        self.lang_map = {
            "en": "en",
            "es": "es",
            "fr": "fr"
        }
        
        # Patterns to remove (only when they appear as complete utterances)
        self.hallucination_patterns = [
            r'^\s*you\s*$',  # Just "you" alone
            r'^\s*thank you\s*$',  # Just "thank you" alone
            r'^\s*thanks\s*$',  # Just "thanks" alone
        ]
        
    def _filter_hallucinations(self, text):
        """Filter out common hallucinations while preserving legitimate uses."""
        if not text:
            return text
            
        # Check if the entire text matches any hallucination pattern
        for pattern in self.hallucination_patterns:
            if re.fullmatch(pattern, text, flags=re.IGNORECASE):
                return ""
                
        # Otherwise return the original text
        return text
    
    def transcribe(self, audio_data):
        """Transcribe audio data and detect language."""
        try:
            # Convert audio bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Start timing the actual transcription
            start = time.time()
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                best_of=5,
                patience=0.1,
                length_penalty=1.0,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                language=None,
                without_timestamps=True
            )
            
            # Process results
            raw_text = " ".join(segment.text for segment in segments).strip()
            
            # Apply hallucination filter (only removes complete-utterance hallucinations)
            filtered_text = self._filter_hallucinations(raw_text)
            
            if filtered_text:
                # Map Whisper language to our language codes
                detected_lang = None
                if info.language in self.lang_map:
                    detected_lang = self.lang_map[info.language]
                return filtered_text, detected_lang
            
            return None, None
            
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return None, None