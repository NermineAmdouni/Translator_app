"""
MCP (Model Context Protocol) Conversation Context Manager
Handles conversation history, topic extraction, and contextual summaries for better translations.
Uses YAKE for advanced keyword extraction.
"""

import json
import os
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Optional, Tuple, Set

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("⚠️  YAKE not installed. Install with: pip install yake")
    print("   Falling back to simple keyword extraction")


class ConversationContext:
    """Manages conversation history for better translation context."""
    
    def __init__(self, max_history: int = 100, context_window_minutes: int = 60):
        """
        Initialize conversation context manager.
        
        Args:
            max_history: Maximum number of exchanges to keep in memory
            context_window_minutes: Time window for relevant context (in minutes)
        """
        self.max_history = max_history
        self.context_window = timedelta(minutes=context_window_minutes)
        self.history = deque(maxlen=max_history)
        self.topics = set()  # Track conversation topics
        self.language_pairs = {}  # Track language pair frequencies
        
        # Initialize YAKE keyword extractor
        if YAKE_AVAILABLE:
            self.yake_extractor = yake.KeywordExtractor(
                lan="en",  # Default language, will be updated per text
                n=3,       # Maximum number of words in keyphrase
                dedupLim=0.7,  # Deduplication threshold
                top=10,    # Number of keywords to extract
                features=None
            )
        else:
            self.yake_extractor = None
        
    def add_exchange(self, original_text: str, source_lang: str, 
                    translated_text: str, target_lang: str):
        """Add a translation exchange to the conversation history."""
        exchange = {
            'timestamp': datetime.now(),
            'original': original_text,
            'source_lang': source_lang,
            'translated': translated_text,
            'target_lang': target_lang,
            'tokens': original_text.lower().split()
        }
        
        self.history.append(exchange)
        
        # Update language pair tracking
        pair = f"{source_lang}->{target_lang}"
        self.language_pairs[pair] = self.language_pairs.get(pair, 0) + 1
        
        # Extract important topics from the original text using YAKE
        self._extract_topics(original_text, source_lang)
    
    def _extract_topics_with_yake(self, text: str, language: str = "en") -> List[str]:
        """Robust YAKE keyword extraction with comprehensive error handling."""
        print("🔍 YAKE received text:", repr(text))
        if not YAKE_AVAILABLE or not self.yake_extractor or len(text.split()) < 3:
            return self._extract_topics_fallback(text)
        
        try:
            # Configure language
            lang_map = {'en':'en', 'es':'es', 'fr':'fr'}
            yake_lang = lang_map.get(language, 'en')
            
            if hasattr(self.yake_extractor, 'lan') and self.yake_extractor.lan != yake_lang:
                self.yake_extractor = yake.KeywordExtractor(
                    lan=yake_lang,
                    n=2,
                    dedupLim=0.8,
                    top=5,
                    features=None
                )
            
            # Extract and process keywords with multiple safety checks
            keyword_results = self.yake_extractor.extract_keywords(text)
            extracted_topics = []
            
            for result in keyword_results:
                try:
                    # Handle different YAKE output formats
                    if isinstance(result, tuple) and len(result) >= 2:
                        score, keyword = result[0], result[1]
                    elif hasattr(result, 'score') and hasattr(result, 'ngram'):
                        score, keyword = result.score, result.ngram
                    else:
                        continue
                    
                    # Convert score to float safely
                    try:
                        score = float(score) if not isinstance(score, (int, float)) else score
                    except (ValueError, TypeError):
                        continue
                    
                    # Validate keyword
                    if keyword and isinstance(keyword, str):
                        extracted_topics.append(keyword.strip().lower())

                        #extracted_topics.append(keyword)
                        
                except Exception as e:
                    continue
            print(extracted_topics[:3])
            return extracted_topics[:3]  # Return top 3 most relevant
        
        except Exception as e:
            print(f"⚠️ YAKE extraction failed: {str(e)}")
            return self._extract_topics_fallback(text)

    def _extract_topics_fallback(self, text: str) -> List[str]:
        """Fallback topic extraction when YAKE is not available."""
        # Expanded stopwords for better filtering
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'am', 'be', 'been', 'being',
            'do', 'does', 'did', 'get', 'got', 'go', 'went', 'come', 'came'
        }
        
        # Clean and filter words
        words = []
        for word in text.split():
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            if (len(clean_word) > 2 and 
                clean_word not in stopwords and 
                clean_word.isalpha() and  # Only alphabetic words
                not clean_word.isdigit()):  # Exclude pure numbers
                words.append(clean_word)
        
        if not words:
            return []
            
        # Score words based on importance criteria
        word_scores = {}
        for word in words:
            score = 0
            
            # Length bonus (longer words often more meaningful)
            score += min(len(word) * 0.5, 4)  # Cap at 4 points
            
            # Frequency in current text
            score += words.count(word) * 2
            
            # Capitalization bonus (proper nouns are important)
            if any(w[0].isupper() for w in text.split() if w.lower().strip('.,!?;:"()[]{}') == word):
                score += 3
            
            # Penalize very common patterns
            if word.endswith('ing') or word.endswith('ed') or word.endswith('ly'):
                score -= 1
                
            word_scores[word] = score
        
        # Get top scoring words
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        important_words = [word for word, score in sorted_words[:5] if score > 1]  # Top 5 with minimum score
        
        return important_words
    

    def _extract_topics(self, text: str, language: str = "en"):
        """Main topic extraction method with fallback handling."""
        try:
            if YAKE_AVAILABLE and self.yake_extractor:
                topics = self._extract_topics_with_yake(text, language)
            else:
                topics = self._extract_topics_fallback(text)
            
            # Add new topics and maintain size limit
            self.topics.update(topics)
            if len(self.topics) > 100:
                self.topics = set(list(self.topics)[-50:])
                
        except Exception as e:
            print(f"Topic extraction error: {e}")
            topics = []
            
        return topics
    def get_recent_context(self) -> List[Dict]:
        """Get recent conversation context within the time window."""
        cutoff_time = datetime.now() - self.context_window
        recent = [ex for ex in self.history if ex['timestamp'] > cutoff_time]
        return recent[-10:]  # Last 10 recent exchanges for context
    
    def get_contextual_summary(self, current_text: str, source_lang: str) -> str:
        """Generate a contextual summary for the translator."""
        recent_context = self.get_recent_context()
        
        if not recent_context:
            return ""
        
        # Build context summary
        context_parts = []
        
        # Add recent exchanges
        if len(recent_context) > 0:
            context_parts.append("Recent conversation:")
            for ex in recent_context[-3:]:  # Last 3 exchanges
                context_parts.append(f"- {ex['source_lang']}: {ex['original']}")
                context_parts.append(f"  {ex['target_lang']}: {ex['translated']}")
        
        # Add topic context if relevant
        current_tokens = set(current_text.lower().split())
        relevant_topics = current_tokens.intersection(self.topics)
        if relevant_topics:
            context_parts.append(f"Related topics discussed: {', '.join(list(relevant_topics)[:3])}")
        
        return "\n".join(context_parts)
    
    def get_topic_relevance_score(self, text: str) -> float:
        """Calculate how relevant the current text is to ongoing conversation topics."""
        if not self.topics:
            return 0.0
            
        current_tokens = set(text.lower().split())
        matching_topics = current_tokens.intersection(self.topics)
        
        if not matching_topics:
            return 0.0
            
        # Score based on ratio of matching topics
        relevance = len(matching_topics) / min(len(current_tokens), len(self.topics))
        return min(relevance, 1.0)
    
    def get_language_pair_frequency(self, source_lang: str, target_lang: str) -> int:
        """Get frequency count for a specific language pair."""
        pair = f"{source_lang}->{target_lang}"
        return self.language_pairs.get(pair, 0)
    
    def get_conversation_stats(self) -> Dict:
        """Get comprehensive conversation statistics."""
        recent = self.get_recent_context()
        
        # Calculate time span
        if self.history:
            oldest = min(ex['timestamp'] for ex in self.history)
            newest = max(ex['timestamp'] for ex in self.history)
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
    
    def clear_history(self):
        """Clear all conversation history."""
        self.history.clear()
        self.topics.clear()
        self.language_pairs.clear()
    
    def save_to_file(self, filepath: str):
        """Save conversation history to file."""
        try:
            history_data = []
            for ex in self.history:
                history_data.append({
                    'timestamp': ex['timestamp'].isoformat(),
                    'original': ex['original'],
                    'source_lang': ex['source_lang'],
                    'translated': ex['translated'],
                    'target_lang': ex['target_lang']
                })
            
            data_to_save = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'history': history_data,
                'topics': list(self.topics),
                'language_pairs': self.language_pairs,
                'stats': self.get_conversation_stats()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load conversation history from file.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load history
            for ex_data in data.get('history', []):
                exchange = {
                    'timestamp': datetime.fromisoformat(ex_data['timestamp']),
                    'original': ex_data['original'],
                    'source_lang': ex_data['source_lang'],
                    'translated': ex_data['translated'],
                    'target_lang': ex_data['target_lang'],
                    'tokens': ex_data['original'].lower().split()
                }
                self.history.append(exchange)
            
            # Load topics and language pairs
            self.topics = set(data.get('topics', []))
            self.language_pairs = data.get('language_pairs', {})
            
            print(f"✅ Loaded {len(self.history)} conversation exchanges from history")
            return True
            
        except Exception as e:
            print(f"❌ Error loading conversation history: {e}")
            return False
    
    def export_readable_history(self, filepath: str):
        """Export conversation history in a human-readable format."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("CONVERSATION HISTORY EXPORT\n")
                f.write("=" * 50 + "\n\n")
                
                stats = self.get_conversation_stats()
                f.write(f"Total Exchanges: {stats['total_exchanges']}\n")
                f.write(f"Conversation Span: {stats['conversation_span_minutes']} minutes\n")
                f.write(f"Languages Used: {', '.join(stats['language_pairs'].keys())}\n")
                f.write(f"Topics Discussed: {', '.join(stats['topics'][:10])}...\n\n")
                
                f.write("CONVERSATION LOG\n")
                f.write("-" * 30 + "\n\n")
                
                for i, ex in enumerate(self.history, 1):
                    timestamp = ex['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{i:03d}] {timestamp}\n")
                    f.write(f"  🗣️  ({ex['source_lang'].upper()}) {ex['original']}\n")
                    f.write(f"  🔄  ({ex['target_lang'].upper()}) {ex['translated']}\n\n")
                    
        except Exception as e:
            print(f"Error exporting readable history: {e}")


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


# Example usage and testing
if __name__ == "__main__":
    # Test the conversation context manager
    context = ConversationContext(max_history=50, context_window_minutes=30)
    
    # Add some sample exchanges
    context.add_exchange("Hello, how are you?", "en", "Hola, ¿cómo estás?", "es")
    context.add_exchange("I'm going to the restaurant", "en", "Voy al restaurante", "es")
    context.add_exchange("¿Qué recomiendas?", "es", "What do you recommend?", "en")
    
    # Test contextual summary
    current_text = "The food at the restaurant was excellent"
    summary = context.get_contextual_summary(current_text, "en")
    print("Contextual Summary:")
    print(summary)
    print()
    
    # Test topic relevance
    relevance = context.get_topic_relevance_score(current_text)
    print(f"Topic Relevance Score: {relevance:.2f}")
    print()
    
    # Show stats
    stats = context.get_conversation_stats()
    print("Conversation Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test file operations
    test_file = "test_conversation.json"
    context.save_to_file(test_file)
    
    # Create new context and load
    new_context = ConversationContext()
    if new_context.load_from_file(test_file):
        print(f"Successfully loaded {len(new_context.history)} exchanges")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)