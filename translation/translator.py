import torch
from transformers import MarianMTModel, MarianTokenizer
import re
from mcp.mcp import ConversationContext

class Translator:
    """Handles translation between languages."""
    def __init__(self, languages, target_lang):
        self.languages = languages
        self.target_lang = target_lang
        self.translation_models = {}
        
        # Load translation models
        self._load_translation_models()
    
    def _load_translation_models(self):
        """Load translation models for all possible source languages to target."""
        print("Loading translation models...")
        for src_lang, lang_info in self.languages.items():
            if src_lang != self.target_lang and self.target_lang in lang_info["translation_models"]:
                model_name = lang_info["translation_models"][self.target_lang]
                print(f"Loading {lang_info['name']}-{self.languages[self.target_lang]['name']} model...")
                self.translation_models[src_lang] = {
                    "model": MarianMTModel.from_pretrained(model_name),
                    "tokenizer": MarianTokenizer.from_pretrained(model_name)
                }
    
    def update_target_language(self, new_target):
        """Update target language and reload models."""
        if new_target == self.target_lang:
            return
            
        self.target_lang = new_target
        
        # Clear existing models to free memory
        self.translation_models = {}
        
        # Load new models
        self._load_translation_models()
    
    def is_complete_sentence(self, text):
        """Check if text forms a complete sentence."""
        text = text.strip().lower()
        if not text:
            return False
            
        if re.search(r'[.!?]\s*$', text):
            return True
            
        words = text.split()
        return len(words) > 8
    
    def translate(self, text, source_lang):
        """Translate text from source language to target language."""
        if not text or source_lang == self.target_lang:
            return None
            
        # Check if we have a translation model for this language pair
        if source_lang not in self.translation_models:
            print(f"No translation model available for {self.languages[source_lang]['name']} to {self.languages[self.target_lang]['name']}")
            return None
            
        try:
            # Use the appropriate translation model for the current source language
            translation_model = self.translation_models[source_lang]
            
            # Convert to tensor and handle properly
            inputs = translation_model["tokenizer"](
                [text[:300]],  # Limit input text to 300 chars
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Generate translation
            with torch.no_grad():
                outputs = translation_model["model"].generate(**inputs)
            
            # Decode output
            translated = translation_model["tokenizer"].batch_decode(
                outputs, 
                skip_special_tokens=True
            )[0]
            
            # Update conversation history with new exchange
            #if self.is_complete_sentence(text):
             #   self.mcp.add_exchange(text, translated)
                
            return translated
                
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return None