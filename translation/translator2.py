import torch
from transformers import MarianMTModel, MarianTokenizer
import re
from collections import deque
from typing import Dict, Optional
import time

class Translator:
    """Translator with context management (no YAKE)"""
    
    def __init__(self, languages: Dict, target_lang: str, device: Optional[str] = None):
        """
        Initialize translator with language support.
        
        Args:
            languages: Dictionary of supported languages and configurations
            target_lang: Target language code (e.g., 'fr')
            device: Hardware device ('cuda', 'mps', or None for auto)
        """
        self.languages = languages
        self.target_lang = target_lang
        self.device = self._determine_device(device)
        self.translation_models = {}
        self.context_history = {}
        self._load_translation_models()
        self.last_translation_time = 0
        self.translation_count = 0
        self.total_translation_time = 0
    
    def _determine_device(self, device: Optional[str]) -> str:
        """Auto-select the best available device."""
        if device:
            return device
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    def _load_translation_models(self):
        """Load all required translation models."""
        print("Loading translation models...")
        for src_lang, lang_info in self.languages.items():
            if src_lang != self.target_lang and self.target_lang in lang_info["translation_models"]:
                model_name = lang_info["translation_models"][self.target_lang]
                print(f"Loading {lang_info['name']} -> {self.languages[self.target_lang]['name']} model...")
                
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name).to(self.device)
                    
                    self.translation_models[src_lang] = {
                        "model": model,
                        "tokenizer": tokenizer
                    }
                    self.context_history[src_lang] = deque(maxlen=3)
                    print(f"Model loaded on {self.device}")
                except Exception as e:
                    print(f"Failed to load model: {str(e)}")
    
    def update_target_language(self, new_target: str):
        """Update target language and reload models."""
        if new_target == self.target_lang:
            return
            
        self.target_lang = new_target
        
        # Clear existing models to free memory
        self.translation_models = {}
        self.context_history = {}
        
        # Load new models
        self._load_translation_models()
    
    def is_complete_sentence(self, text: str) -> bool:
        """
        Simple sentence detection without NLTK.
        Checks for:
        1. Ending punctuation
        2. Question patterns
        3. Minimum length with common verbs
        """
        text = text.strip()
        if not text:
            return False
        
        # Ending punctuation
        if re.search(r'[.!?]\s*$', text):
            return True
        
        # Question detection
        question_words = ('who', 'what', 'when', 'where', 'why', 'how',
                         'can', 'could', 'will', 'would', 'is', 'are',
                         'do', 'does', 'did')
        first_word = text.split()[0].lower() if text.split() else ''
        if first_word in question_words:
            return True
        
        # Verb presence check
        common_verbs = ('is', 'are', 'was', 'were', 'have', 'has',
                       'do', 'does', 'did', 'can', 'could', 'will',
                       'would', 'should', 'must')
        words = text.lower().split()
        return len(words) > 6 and any(verb in words for verb in common_verbs)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean input text before translation."""
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        text = re.sub(r'([.!?])\s*', r'\1 ', text)  # Fix spacing after punctuation
        return text
    
    def translate(self, text: str, source_lang: str) -> Optional[str]:
        """
        Translate text with context management.
        
        Args:
            text: Input text to translate
            source_lang: Source language code
            
        Returns:
            Translated text or None if translation fails
        """
        if not text or source_lang == self.target_lang:
            return None
        
        if source_lang not in self.translation_models:
            print(f"No model for {source_lang}->{self.target_lang}")
            return None
        
        start_time = time.time()
        try:
            # Clean input text
            text = self._preprocess_text(text)
            if not text:
                return None
                
            model_info = self.translation_models[source_lang]
            inputs = model_info["tokenizer"](
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = model_info["model"].generate(**inputs)
            
            translated = model_info["tokenizer"].batch_decode(
                outputs, 
                skip_special_tokens=True
            )[0]
            
            # Update context if complete sentence
            if self.is_complete_sentence(text):
                self.context_history[source_lang].append(text)
            
            # Track performance
            trans_time = time.time() - start_time
            self.last_translation_time = trans_time
            self.translation_count += 1
            self.total_translation_time += trans_time
            
            return translated
        
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get translation performance metrics."""
        avg_time = (self.total_translation_time / self.translation_count 
                   if self.translation_count else 0)
        return {
            "total_translations": self.translation_count,
            "last_time_sec": round(self.last_translation_time, 2),
            "avg_time_sec": round(avg_time, 2),
            "device": self.device
        }
    
    def clear_context(self, source_lang: Optional[str] = None):
        """Clear context for specific language or all languages."""
        if source_lang:
            if source_lang in self.context_history:
                self.context_history[source_lang].clear()
        else:
            for lang in self.context_history:
                self.context_history[lang].clear()
