import argparse
import threading
import queue
import time
import warnings
import os

from stt.audio_silero import AudioRecorder
from stt.whisper_transcriber import WhisperTranscriber
from language_detection.detector import LanguageDetector
from translation.translator2 import Translator
from tts.synthesizer import KokoroSynthesizer
from utils.config import Languages, AudioConfig

# Import the MCP conversation context manager
from mcp.mcp2 import ConversationContext, ContextAwareTranslator

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrilingualTranslator:
    def __init__(self):
        # Initialize language configuration
        language_config = Languages()
        self.languages = language_config.languages
        
        # Get target language configuration from user
        self.target_lang = self.get_target_language()
        print(f"\nTranslation setup: Auto-detect → {self.languages[self.target_lang]['name']}")
        
        # Current source language (will be detected automatically)
        self.source_lang = None
        
        # Initialize conversation context with MCP
        self.conversation_context = ConversationContext(
            max_history=100, 
            context_window_minutes=60
        )
        self.history_file = "conversation_history.json"
        self.conversation_context.load_history(self.history_file)
        
        # Initialize components
        self.audio_config = AudioConfig()
        self.audio_recorder = AudioRecorder(self.audio_config)
        self.transcriber = WhisperTranscriber()
        self.language_detector = LanguageDetector(self.languages)
        
        # Create base translator and wrap it with context-aware translator
        base_translator = Translator(self.languages, self.target_lang)
        self.translator = ContextAwareTranslator(
            base_translator, self.languages, self.target_lang
        )
        
        self.synthesizer = KokoroSynthesizer(
            self.languages[self.target_lang]["kokoro_code"],
            self.languages[self.target_lang]["tts_voice"]
        )
        
        # Queues for communication between components
        self.audio_queue = queue.Queue(maxsize=10)
        self.transcription_queue = queue.Queue(maxsize=10)
        self.translation_queue = queue.Queue(maxsize=5)
        self.tts_queue = queue.Queue(maxsize=5)
        
        # State management
        self.running = True
        self.sentence_buffer = ""
        self.recent_translations = set()
        self.last_processed_time = time.time()
        
        # Constants
        self.processing_delay = 2.0
        
        # Auto-save conversation history periodically
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes

    def get_target_language(self):
        """Let user select target language only."""
        print("\nAvailable target languages:")
        for code, lang in self.languages.items():
            print(f"{code}: {lang['name']}")
        
        while True:
            target = input("Enter TARGET language code (en/es/fr): ").lower().strip()
            if target in self.languages:
                return target
            print("Invalid target language")

    def change_target_language(self, new_target):
        """Change the target language and reload necessary components."""
        if new_target not in self.languages:
            print(f"Invalid target language: {new_target}")
            return False
            
        if new_target == self.target_lang:
            print(f"Already set to {self.languages[new_target]['name']}")
            return False
            
        print(f"\nChanging target language to {self.languages[new_target]['name']}...")
        
        # Update target language
        self.target_lang = new_target
        
        # Clear all queues
        self._clear_queues()
        
        # Update components
        self.translator.update_target_language(new_target)
        self.synthesizer.update_language(
            self.languages[self.target_lang]["kokoro_code"],
            self.languages[self.target_lang]["tts_voice"]
        )
        
        # Reset language detection
        self.source_lang = None
        self.sentence_buffer = ""
        
        print(f"\nNew translation setup: Auto-detect → {self.languages[self.target_lang]['name']}")
        return True

    def _clear_queues(self):
        """Clear all queues."""
        for q in [self.audio_queue, self.transcription_queue, self.translation_queue, self.tts_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break

    def _save_conversation_history(self):
        """Periodically save conversation history."""
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.conversation_context.save_history(self.history_file)
            self.last_save_time = current_time

    def audio_worker(self):
        """Thread for audio recording."""
        self.audio_recorder.start(self.audio_queue, self.running)

    def transcription_worker(self):
        """Thread for speech-to-text processing."""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # Transcribe audio
                text, detected_lang = self.transcriber.transcribe(audio_data)
                
                if text:
                    # Verify language with additional detector
                    if not detected_lang or detected_lang not in self.languages:
                        detected_lang = self.language_detector.detect(text)
                    
                    if detected_lang and detected_lang in self.languages:
                        if detected_lang != self.target_lang:
                            if self.source_lang != detected_lang:
                                self.source_lang = detected_lang
                                print(f"\nDetected language: {self.languages[self.source_lang]['name']}")
                                
                            print(f"\n🗣️  Detected speech ({self.languages[self.source_lang]['name']}): {text}")
                            self.transcription_queue.put((text, detected_lang))
                
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {str(e)}")

    def translation_worker(self):
        """Thread for translation processing with conversation context."""
        while self.running:
            try:
                # Check for new transcriptions
                try:
                    text, source_lang = self.transcription_queue.get(timeout=0.2)
                    
                    if source_lang == self.target_lang:
                        self.transcription_queue.task_done()
                        continue
                        
                    self.sentence_buffer = (self.sentence_buffer + " " + text).strip()
                    self.transcription_queue.task_done()
                except queue.Empty:
                    pass
                
                # Process accumulated text if needed
                current_time = time.time()
                if self.sentence_buffer and (
                    self.translator.base_translator.is_complete_sentence(self.sentence_buffer) or
                    (current_time - self.last_processed_time > self.processing_delay and 
                     len(self.sentence_buffer.split()) >= 3)
                ):
                    if self.sentence_buffer not in self.recent_translations and self.source_lang:
                        # Get conversation context from MCP
                        context = self.conversation_context.get_contextual_summary(
                            minutes=10
                        )
                        
                        # Calculate topic relevance
                        """relevance_score = self.conversation_context.get_topic_relevance_score(
                            self.sentence_buffer
                        )"""
                        
                        # Translate with context
                        translation = self.translator.translate_with_context(
                            self.sentence_buffer, self.source_lang, context
                        )
                        
                        if translation:
                            print(f"🔄  Translated to {self.languages[self.target_lang]['name']}: {translation}")
                            
                            # Add to conversation history via MCP
                            self.conversation_context.add_exchange(
                                self.sentence_buffer, self.source_lang,
                                translation, self.target_lang
                            )
                            
                            self.translation_queue.put(translation)
                            
                            # Update recent translations cache
                            self.recent_translations.add(self.sentence_buffer)
                            if len(self.recent_translations) > 10:
                                self.recent_translations.pop()
                            
                            # Show context info if available
                            """if context:
                                print(f"    💡 Using conversation context (relevance: {relevance_score:.2f})")
                            """
                            # Show language pair frequency
                            pair_freq = self.conversation_context.get_language_pair_frequency(
                                self.source_lang, self.target_lang
                            )
                            if pair_freq > 1:
                                print(f"    📊 {self.source_lang}→{self.target_lang} used {pair_freq} times")
                    
                    self.sentence_buffer = ""
                    self.last_processed_time = current_time
                
                # Periodically save conversation history
                self._save_conversation_history()
                    
            except Exception as e:
                print(f"Translation error: {str(e)}")

    def tts_worker(self):
        """Thread for text-to-speech processing."""
        while self.running:
            try:
                translation = self.translation_queue.get(timeout=0.5)
                self.synthesizer.speak(translation)
                self.translation_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {str(e)}")

    def show_conversation_stats(self):
        """Display conversation statistics using MCP."""
        stats = self.conversation_context.get_conversation_stats()
        
        print("\n" + "="*60)
        print("📊 CONVERSATION STATISTICS (MCP)")
        print("="*60)
        print(f"Total exchanges: {stats['total_exchanges']}")
        print(f"Recent exchanges (in context window): {stats['recent_exchanges']}")
        print(f"Conversation span: {stats['conversation_span_minutes']} minutes")
        print(f"Context window: {stats['context_window_minutes']} minutes")
        print(f"Total topics identified: {stats['total_topics']}")
        
        if stats['topics']:
            print(f"Topics discussed: {', '.join(stats['topics'][:10])}")
            if len(stats['topics']) > 10:
                print(f"  ... and {len(stats['topics']) - 10} more")
        
        print(f"\nLanguage pairs used:")
        for pair, count in stats['language_pairs'].items():
            print(f"  {pair}: {count} times")
        
        print("="*60)

    def export_conversation(self):
        """Export conversation history in readable format."""
        export_file = f"conversation_export_{int(time.time())}.txt"
        self.conversation_context.export_readable_history(export_file)
        print(f"📄 Conversation exported to: {export_file}")

    def start(self):
        """Start the translation pipeline."""
        try:
            # Create and start worker threads
            threads = [
                threading.Thread(target=self.audio_worker, daemon=True),
                threading.Thread(target=self.transcription_worker, daemon=True),
                threading.Thread(target=self.translation_worker, daemon=True),
                threading.Thread(target=self.tts_worker, daemon=True)
            ]
            
            for thread in threads:
                thread.start()

            print("\n🚀 Context-Aware Real-time Translator Ready! (MCP Enabled)")
            print(f"📍 Auto-detecting source language → {self.languages[self.target_lang]['name']}")
            print("💾 Conversation history loaded and will be saved automatically")
            print("🧠 MCP (Model Context Protocol) providing intelligent context")
            print("="*70)
            print("Commands:")
            print("  'lang'    - Change target language")
            print("  'stats'   - Show conversation statistics")
            print("  'export'  - Export conversation to readable file")
            print("  'clear'   - Clear conversation history")
            print("  'save'    - Save conversation history now")
            print("  'help'    - Show this help")
            print("  Ctrl+C    - Exit")
            print("="*70)
            
            while self.running:
                try:
                    cmd = input("").strip().lower()
                    if cmd == "lang":
                        print("\nAvailable target languages:")
                        for code, lang in self.languages.items():
                            print(f"  {code}: {lang['name']}")
                        new_lang = input("Enter new target language code: ").lower().strip()
                        if self.change_target_language(new_lang):
                            print("\n🎧 Listening for speech... (Type command or Ctrl+C to exit)")
                    elif cmd == "stats":
                        self.show_conversation_stats()
                    elif cmd == "export":
                        self.export_conversation()
                    elif cmd == "clear":
                        confirm = input("❓ Clear conversation history? This cannot be undone! (y/N): ").lower().strip()
                        if confirm == 'y':
                            self.conversation_context.clear_history()
                            print("🗑️  Conversation history cleared.")
                    elif cmd == "save":
                        self.conversation_context.save_to_file(self.history_file)
                        print("💾 Conversation history saved.")
                    elif cmd == "help":
                        print("\n📋 Available Commands:")
                        print("  lang    - Change target language")
                        print("  stats   - Show detailed conversation statistics")
                        print("  export  - Export conversation to readable text file")
                        print("  clear   - Clear all conversation history")
                        print("  save    - Force save conversation history")
                        print("  help    - Show this help message")
                        print("  Ctrl+C  - Exit the translator")
                    elif cmd:
                        print("❓ Unknown command. Type 'help' for available commands.")
                except Exception as e:
                    print(f"Command error: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down translator...")
        finally:
            self.running = False
            self.audio_recorder.stop()
            # Save conversation history before exit
            self.conversation_context.save_history(self.history_file)
            print("💾 Final conversation history saved.")
            print("👋 Goodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context-Aware Trilingual Real-time Translator with MCP")
    parser.add_argument("--target", "-t", 
                       help="Set target language directly (en/es/fr)", 
                       choices=['en', 'es', 'fr'])
    parser.add_argument("--history-size", "-hs", 
                       type=int, default=100,
                       help="Maximum conversation history size (default: 100)")
    parser.add_argument("--context-window", "-cw", 
                       type=int, default=60,
                       help="Context window in minutes (default: 60)")
    parser.add_argument("--no-yake", 
                       action="store_true",
                       help="Disable YAKE keyword extraction")
    parser.add_argument("--save-interval", "-si", 
                       type=int, default=300,
                       help="Auto-save interval in seconds (default: 300)")
    parser.add_argument("--debug", 
                       action="store_true",
                       help="Enable debug mode with verbose output")
    
    args = parser.parse_args()
    
    # Configure based on arguments
    if args.debug:
        print("🐛 Debug mode enabled")
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        translator = TrilingualTranslator()
        
        # Override target language if specified
        if args.target:
            translator.target_lang = args.target
            print(f"Target language set via command line: {args.target}")
        
        # Override conversation context settings
        if args.history_size != 100 or args.context_window != 60:
            translator.conversation_context = ConversationContext(
                max_history=args.history_size,
                context_window_minutes=args.context_window
            )
            translator.conversation_context.load_history(translator.history_file)
            print(f"Custom context settings: history={args.history_size}, window={args.context_window}min")
        
        # Override save interval
        if args.save_interval != 300:
            translator.save_interval = args.save_interval
            print(f"Auto-save interval set to {args.save_interval} seconds")
        
        if args.no_yake:
            print("⚠️  YAKE keyword extraction disabled via command line")
        
        translator.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        print("🏁 Translator shutdown complete")