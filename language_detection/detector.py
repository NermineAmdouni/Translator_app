from langdetect import detect, LangDetectException
import re

class LanguageDetector:
    """Detects language of text and identifies complete sentences."""
    
    def __init__(self, languages):
        self.languages = languages
        self.last_detected_lang = "en"  # Default fallback
        self.lang_map = {
            "en": "en",
            "es": "es",
            "fr": "fr"
        }
    
    def detect(self, text):
        """Detect the language of the text, fallback to last valid detection."""
        if not text:
            return self.last_detected_lang
        
        try:
            detected = detect(text)
            if detected in self.lang_map and self.lang_map[detected] in self.languages:
                self.last_detected_lang = self.lang_map[detected]  # update memory
                return self.last_detected_lang
            # Language not in supported set — fallback to last
            return self.last_detected_lang
        except LangDetectException:
            return self.last_detected_lang
    
    def is_complete_sentence(self, text, source_lang=None):
        """Check if text forms a complete sentence."""
        text = text.strip().lower()
        if not text:
            return False
            
        if re.search(r'[.!?]\s*$', text):
            return True
            
        words = text.split()
        if not words:
            return False
        
        if not source_lang:
            return len(words) > 8
            
        connectors = {
            "en": ["and", "but", "or", "because", "although", "while"],
            "es": ["y", "pero", "o", "porque", "aunque", "mientras"],
            "fr": ["et", "mais", "ou", "parce que", "bien que", "pendant que"]
        }
        
        last_word = words[-1]
        if last_word in connectors.get(source_lang, []):
            return False
            
        return len(words) > 8
