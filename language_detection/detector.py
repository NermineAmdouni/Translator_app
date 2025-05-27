from langdetect import detect, LangDetectException
import re

class LanguageDetector:
    """Detects language of text and identifies complete sentences."""
    def __init__(self, languages):
        self.languages = languages
        # Mapping from langdetect codes to our language codes
        self.lang_map = {
            "en": "en",
            "es": "es",
            "fr": "fr"
        }
    
    def detect(self, text):
        """Detect the language of the text."""
        if not text:
            return None
            
        try:
            detected = detect(text)
            if detected in self.lang_map and self.lang_map[detected] in self.languages:
                return self.lang_map[detected]
            return None
        except LangDetectException:
            return None
    
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
        
        # If source language is not yet detected, we can't check connectors
        if not source_lang:
            return len(words) > 8
            
        # Common connectors that might indicate an incomplete sentence
        connectors = {
            "en": ["and", "but", "or", "because", "although", "while"],
            "es": ["y", "pero", "o", "porque", "aunque", "mientras"],
            "fr": ["et", "mais", "ou", "parce que", "bien que", "pendant que"]
        }
        
        last_word = words[-1]
        if last_word in connectors.get(source_lang, []):
            return False
            
        if len(words) > 8:
            return True
            
        return False