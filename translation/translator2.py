import torch
from transformers import MarianMTModel, MarianTokenizer
import re
from collections import deque
from typing import Dict, Optional
import time
from llm_langchain.use_llm import clean_text

class Translator:
    
    def __init__(self, languages: Dict, target_lang: str, device: Optional[str] = None):

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
        """Load only the models needed for this translator's target language."""
        print(f"Loading translation models for target language: {self.target_lang}")
        
        for src_lang, src_info in self.languages.items():
            if src_lang == self.target_lang:
                continue  # Skip same language
                
            model_name = src_info["translation_models"].get(self.target_lang)
            if not model_name:
                continue
                
            key = (src_lang, self.target_lang)
            if key in self.translation_models:
                continue  # Already loaded

            print(f"Loading {src_info['name']} -> {self.languages[self.target_lang]['name']} model...")
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name).to(self.device)

                self.translation_models[key] = {
                    "model": model,
                    "tokenizer": tokenizer
                }

                if src_lang not in self.context_history:
                    self.context_history[src_lang] = deque(maxlen=3)

                print(f"{src_lang}->{self.target_lang} loaded on {self.device}")
            except Exception as e:
                print(f"Failed to load model {src_lang}->{self.target_lang}: {str(e)}")
    
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
    

    def translate(self, text: str, source_lang: Optional[str], target_lang: Optional[str] = None) -> Optional[str]:
        target_lang = target_lang or self.target_lang
        #source_lang= "en"
        if not text:
            return ""

        # 🧠 Fallback to previous language if source_lang is None
        if source_lang is None:
            source_lang = getattr(self, "previous_source_lang", None)
            if source_lang is None:
                print("Source language is None and no previous language available.")
                return ""
        else:
            # Update previous language only when current one is valid
            self.previous_source_lang = source_lang

        if source_lang == target_lang:
            return text  # No translation needed

        key = (source_lang, target_lang)
        model_info = self.translation_models.get(key)
        if not model_info:
            print(f"No model for {source_lang}->{target_lang}")
            return ""

        start_time = time.time()
        try:
            print(f"Original text: '{text}'")

            text = self._preprocess_text(text)
            if not text:
                return ""

            inputs = model_info["tokenizer"](
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}


            with torch.no_grad():
                outputs = model_info["model"].generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )


            translated_list = model_info["tokenizer"].batch_decode(
                outputs,
                skip_special_tokens=True
            )

            translated = translated_list[0] if translated_list else ""

            print(f"Translated text before cleaning: '{translated}'")

            translated = translated.strip()
            print(f"Final translated text: '{translated}'")

            if translated == "{}":
                translated = ""


            if self.is_complete_sentence(text):
                self.context_history[source_lang].append(text)

            trans_time = time.time() - start_time
            self.last_translation_time = trans_time
            self.translation_count += 1
            self.total_translation_time += trans_time

            return translated

        except Exception as e:
            print(f"Translation error: {str(e)}")
            return ""

    
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
