import json
import os
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict


class ConversationContext:
    """
    Manages conversation history and context for a real-time multilingual translator
    or any conversation-based system.
    """

    def __init__(self, max_history: int = 100, context_window_minutes: int = 60, save_path: str = None):
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

        # Perform topic extraction heuristic when buffer hits 50+ words
        buffered_text = " ".join(self._topic_buffer)
        if len(buffered_text.split()) >= 50:
            extracted_topics = self._extract_topics_fallback(buffered_text)
            self.topics.update(extracted_topics)

            # Limit topics to last 50 if more than 100 (to avoid overgrowth)
            if len(self.topics) > 100:
                self.topics = set(list(self.topics)[-50:])

            self._topic_buffer = []

    def _extract_topics_fallback(self, text: str) -> List[str]:
        """
        Heuristic fallback topic extraction without YAKE.
        Extracts keywords based on word frequency, length, capitalization, and filtering stopwords.

        Args:
            text: The text to extract topics from.

        Returns:
            List of extracted keyword topics.
        """
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
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
            if any(w[0].isupper() for w in text.split() if w.lower().strip('.,!?;:"()[]{}') == word):
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
        summary = f"Recent conversation ({minutes} minutes):\n"
        for ex in recent:
            summary += f"- {ex['original']} ({ex['source_lang']}→{ex['target_lang']})\n"
        summary += f"\nTop topics: {', '.join(topics)}"
        return summary

    def get_top_topics(self, n: int = 5) -> List[str]:
        """
        Return the top N topics based on frequency in the conversation history.

        Args:
            n: Number of top topics to return.

        Returns:
            List of topic strings.
        """
        if not self.topics:
            return []

        # Count topic appearances in history
        topic_counts = {topic: 0 for topic in self.topics}
        for ex in self.history:
            tokens = ex['tokens']
            for topic in self.topics:
                if topic in tokens:
                    topic_counts[topic] += 1

        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:n]]

class ContextAwareTranslator:
    """Enhanced translator that uses conversation context for better translations."""
    
    def __init__(self, base_translator, languages: Dict, target_lang: str):
        """
        Initialize context-aware translator.
        
        Args:
            base_translator: The base translator instance
            languages: Dictionary of supported languages
            target_lang: Current target language code
        """
        self.base_translator = base_translator
        self.languages = languages
        self.target_lang = target_lang
        
    def translate_with_context(self, text: str, source_lang: str, context: str = "") -> str:
        """
        Translate text using conversation context.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            context: Conversation context string
            
        Returns:
            str: Translated text
        """
        if not context:
            return self.base_translator.translate(text, source_lang)
        
        # Enhanced translation with context
        # For now, we'll use the base translator but could integrate with
        # more sophisticated models that accept context
        
        # Simple context-aware translation approach:
        # 1. Use base translation
        base_translation = self.base_translator.translate(text, source_lang)
        
        # 2. Could add context refinement logic here
        # For example, checking for consistency with previous translations
        # or using the context to disambiguate meanings
        
        return base_translation
    
    def update_target_language(self, new_target: str):
        """Update target language."""
        self.target_lang = new_target
        self.base_translator.update_target_language(new_target)

