import json
import os
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict
import yake


class ConversationContext:
    """
    Manages conversation history and context for a real-time multilingual translator
    or any conversation-based system.
    """

    def __init__(self, max_history: int = 100, context_window_minutes: int = 60, save_path: str = None, yake_max_keywords=5): 
        """
        Initialize the conversation context manager.

        Args:
            max_history: Maximum number of exchanges to keep in memory
            context_window_minutes: Time window for relevant context (in minutes)
            save_path: Optional file path to save/load conversation history as JSON
        """
        self.max_history = max_history
        self.context_window = timedelta(minutes=context_window_minutes)
        self.history = deque(maxlen=max_history)  # stores dicts of exchanges
        self._topic_buffer = []
        self.topics = set()  # distinct topic keywords from conversation
        self.language_pairs = {}  # counts of language pairs encountered
        self.save_path = save_path
        if save_path and os.path.exists(save_path):
            self.load_history(save_path)
        self.yake_max_keywords = yake_max_keywords
        self.yake_extractor = yake.KeywordExtractor(top=self.yake_max_keywords, stopwords=None)

    def add_exchange(self, original_text: str, source_lang: str,
                     translated_text: str, target_lang: str):
        """
        Add a translation exchange to the conversation history.

        Args:
            original_text: The original utterance/text
            source_lang: Language code of the original text (e.g., 'en')
            translated_text: The translated version of the original text
            target_lang: Language code of the translated text (e.g., 'fr')
        """
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'original': original_text,
            'source_lang': source_lang,
            'translated': translated_text,
            'target_lang': target_lang,
            'tokens': original_text.lower().split()
        }

        self.history.append(exchange)

        # Track language pair usage frequency
        pair = f"{source_lang}->{target_lang}"
        self.language_pairs[pair] = self.language_pairs.get(pair, 0) + 1

        # Accumulate original text for topic extraction
        self._topic_buffer.append(original_text)

        # Perform topic extraction when buffer hits 50+ words
        buffered_text = " ".join(self._topic_buffer)
        if len(buffered_text.split()) >= 50:
            #extracted_topics = self.extract_topics(buffered_text)
            extracted_topics = self.extract_topics_yake(buffered_text)
            self.topics.update(extracted_topics)

            # Limit topics to last 50 if more than 100 (avoid overgrowth)
            if len(self.topics) > 100:
                self.topics = set(list(self.topics)[-50:])

            self._topic_buffer = []
    def extract_topics_yake(self, text: str) -> List[str]:
        """
        Use YAKE keyword extractor to extract important keywords from text.

        Args:
            text: Input text.

        Returns:
            List of extracted keywords.
        """
        keywords_with_scores = self.yake_extractor.extract_keywords(text)
        # Just return keywords, ignoring scores for topics
        keywords = [kw for kw, score in keywords_with_scores]
        return keywords
    
    def extract_topics(self, text: str) -> List[str]:
        """
        Extract keywords based on word frequency, length, capitalization, and filtering stopwords.

        Args:
            text: The text to extract topics from.

        Returns:
            List of extracted keyword topics.
        """
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
            'this', 'that', 'these', 'those','the', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'am', 'be', 'been', 'being',
            'do', 'does', 'did', 'get', 'got', 'go', 'went', 'come', 'came'
        }

        words = []
        for word in text.split():
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            if (len(clean_word) > 2 and
                clean_word not in stopwords and
                clean_word.isalpha() and
                not clean_word.isdigit()):
                words.append(clean_word)

        if not words:
            return []

        word_scores = {}
        for word in set(words):
            score = 0
            # Length bonus (longer words often more meaningful)
            score += min(len(word) * 0.5, 4)

            # Frequency bonus
            freq = words.count(word)
            score += freq * 2

            # Capitalization bonus (check if word capitalized in original text)
            if any(w and w[0].isupper() for w in text.split() if w.lower().strip('.,!?;:"()[]{}') == word):
                score += 3

            # Penalize common suffixes (less meaningful)
            if word.endswith(('ing', 'ed', 'ly')):
                score -= 1

            word_scores[word] = score

        # Pick top 5 scoring words with score > 1
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        important_words = [word for word, score in sorted_words[:5] if score > 1]

        return important_words

    def get_recent_context(self, minutes: int = None) -> List[Dict]:
        """
        Retrieve recent conversation exchanges within a time window.

        Args:
            minutes: Number of minutes back to retrieve context.
                     If None, returns all history.

        Returns:
            List of exchange dictionaries.
        """
        if minutes is None:
            return list(self.history)

        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = [ex for ex in self.history if datetime.fromisoformat(ex['timestamp']) >= cutoff]
        return recent

    def save_history(self, path: str = None):
        """
        Save conversation history and topics to a JSON file.

        Args:
            path: File path to save; defaults to self.save_path if set.
        """
        if path is None:
            path = self.save_path
        if path is None:
            raise ValueError("No save path provided.")

        data = {
            'history': list(self.history),
            'topics': list(self.topics),
            'language_pairs': self.language_pairs
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_history(self, path: str = None):
        """
        Load conversation history and topics from a JSON file.

        Args:
            path: File path to load from; defaults to self.save_path if set.
        """
        if path is None:
            path = self.save_path
        if path is None or not os.path.exists(path):
            return  # Nothing to load

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.history = deque(data.get('history', []), maxlen=self.max_history)
            self.topics = set(data.get('topics', []))
            self.language_pairs = data.get('language_pairs', {})

    def get_contextual_summary(self, minutes: int = 10) -> str:
        recent = self.get_recent_context(minutes)
        topics = self.get_top_topics(5)
        summary = f"Recent conversation (last {minutes} minutes):\n"
        for ex in recent:
            summary += f"- {ex['original']} ({ex['source_lang']}→{ex['target_lang']})\n"
        summary += f"\nTop topics: {', '.join(topics)}"
        return summary

    def get_top_topics(self, n: int = 5) -> List[str]:
        if not self.topics:
            return []

        topic_counts = {topic: 0 for topic in self.topics}
        for ex in self.history:
            tokens = ex.get('tokens', [])
            for topic in self.topics:
                if topic in tokens:
                    topic_counts[topic] += 1

        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:n]]
    
    def get_language_pair_frequency(self, source_lang: str, target_lang: str) -> int:
        """Get frequency count for a specific language pair."""
        pair = f"{source_lang}->{target_lang}"
        return self.language_pairs.get(pair, 0)
        
    def get_conversation_stats(self) -> Dict:
        """Get comprehensive conversation statistics."""
        recent = self.get_recent_context()

        # Define timestamp format (adjust if needed)
        ts_format = "%Y-%m-%dT%H:%M:%S.%f"

        if self.history:
            timestamps = [datetime.strptime(ex['timestamp'], ts_format) for ex in self.history]
            oldest = min(timestamps)
            newest = max(timestamps)
            time_span = newest - oldest
        else:
            time_span = timedelta(0)

        return {
            'total_exchanges': len(self.history),
            'recent_exchanges': len(recent),
            'total_topics': len(self.topics),
            'topics': list(self.topics),
            'language_pairs': self.language_pairs.copy(),
            'conversation_span_minutes': int(time_span.total_seconds() / 60),
            'context_window_minutes': int(self.context_window.total_seconds() / 60)
        }

from typing import Dict, Optional
import time

class ContextAwareTranslator:
    """
    Context-aware translator that wraps the base Translator class
    and adds conversation context management.
    """
    
    def __init__(self, base_translator, languages: Dict, target_lang: str,context_manager: ConversationContext):
        """
        Initialize the context-aware translator.
        
        Args:
            base_translator: Instance of the base Translator class
            languages: Dictionary of supported languages
            target_lang: Target language code
        """
        self.base_translator = base_translator
        self.languages = languages
        self.target_lang = target_lang
        
        # Context management
        self.conversation_topics = set()
        self.recent_translations = {}
        self.translation_patterns = {}
        self.partial_sentence = ""
        self.context_manager = context_manager


    def is_likely_start_of_new_sentence(self, text: str) -> bool:
        """Heuristic to detect the start of a new sentence based on topics and structure."""
        words = text.strip().split()
        if not words:
            return False
        topics = self.context_manager.get_top_topics()
        # Consider it a new sentence if it starts with a topic word or is capitalized and short
        return words[0][0].isupper() or any(topic in text.lower() for topic in topics)

    def is_likely_incomplete_sentence(self, text: str) -> bool:
        """Heuristic based on context, length, and non-terminal words."""
        if len(text.strip().split()) < 3:
            return True

        non_terminal_endings = {"and", "but", "so", "because", "although", "if", "when", "which"}
        last_word = text.strip().split()[-1].lower()
        if last_word in non_terminal_endings:
            return True

        topics = self.context_manager.get_top_topics()
        if topics and not any(topic in text.lower() for topic in topics):
            return True

        return False

    def is_complete_sentence(self, text: str) -> bool:
        """Combines base translator’s check with context-aware heuristics."""
        base_check = self.base_translator.is_complete_sentence(text)
        context_check = not self.is_likely_incomplete_sentence(text)
        return base_check and context_check    
    
    #def is_complete_sentence(self, text: str, current_text: str = None) -> bool:
 
     #   return self.base_translator.is_complete_sentence(text)
    
    def translate(self, text: str, source_lang: str) -> Optional[str]:
        if not text or source_lang == self.target_lang:
            return None

        self.partial_sentence += " " + text.strip()
        self.partial_sentence = self.partial_sentence.strip()

        text_key = f"{source_lang}:{self.partial_sentence.lower()}"
        current_time = time.time()

        # Skip recently translated duplicates
        if text_key in self.recent_translations and current_time - self.recent_translations[text_key] < 10:
            return None

        # Not complete yet
        if not self.is_complete_sentence(self.partial_sentence):
            return None

        translation = self.base_translator.translate(self.partial_sentence, source_lang, self.target_lang)
        if translation:
            self.recent_translations[text_key] = current_time
            self.context_manager.add_exchange(
                original_text=self.partial_sentence,
                source_lang=source_lang,
                translated_text=translation,
                target_lang=self.target_lang
            )
            self.partial_sentence = ""  # Clear after successful translation
            return translation

        return None
    
    def _update_translation_patterns(self, source_lang: str, original: str, translation: str):
        """
        Update translation patterns for context awareness.
        
        Args:
            source_lang: Source language code
            original: Original text
            translation: Translated text
        """
        # Extract potential topics/keywords
        words = original.lower().split()
        significant_words = [w for w in words if len(w) > 3]
        
        for word in significant_words[:3]:  # Keep top 3 significant words
            self.conversation_topics.add(word)
            
        # Limit topic set size
        if len(self.conversation_topics) > 100:
            # Remove oldest topics (simplified approach)
            self.conversation_topics = set(list(self.conversation_topics)[-80:])
    
    """def update_target_language(self, new_target: str):

        self.target_lang = new_target
        self.base_translator.update_target_language(new_target)
        
        # Clear context when changing language
        self.recent_translations.clear()
        self.conversation_topics.clear()
        self.translation_patterns.clear()"""
    
    def clear_context(self, source_lang: Optional[str] = None):
        """
        Clear translation context.
        
        Args:
            source_lang: Specific source language to clear, or None for all
        """
        if source_lang:
            # Clear context for specific language
            keys_to_remove = [k for k in self.recent_translations.keys() 
                            if k.startswith(f"{source_lang}:")]
            for key in keys_to_remove:
                del self.recent_translations[key]
        else:
            # Clear all context
            self.recent_translations.clear()
            self.conversation_topics.clear()
            self.translation_patterns.clear()
        
        # Also clear base translator context
        self.base_translator.clear_context(source_lang)
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics from base translator.
        
        Returns:
            Dictionary with performance metrics
        """
        base_stats = self.base_translator.get_performance_stats()
        
        # Add context-aware stats
        context_stats = {
            "recent_translations_count": len(self.recent_translations),
            "conversation_topics_count": len(self.conversation_topics),
            "current_topics": list(self.conversation_topics)[-10:] if self.conversation_topics else []
        }
        
        return {**base_stats, **context_stats}
    
    def get_conversation_context(self) -> Dict:
        """
        Get current conversation context information.
        
        Returns:
            Dictionary with context information
        """
        return {
            "target_language": self.target_lang,
            "active_topics": list(self.conversation_topics)[-20:],
            "recent_translation_count": len(self.recent_translations),
            "supported_languages": list(self.languages.keys())
        }

    def reset_context(self):
        self.context_manager.context = []
        self.context_manager.topics = []
        self.partial_sentence = ""


    # Delegate any other methods that might be called
    def __getattr__(self, name):
        """
        Delegate any missing method calls to the base translator.
        """
        if hasattr(self.base_translator, name):
            return getattr(self.base_translator, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
