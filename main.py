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
from mcp.mcp2 import ConversationContext, ContextAwareTranslator

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class TrilingualTranslator:
    def __init__(self):
        # Initialize language configuration
        language_config = Languages()
        self.languages = language_config.languages

        self.target_lang = self.get_target_language()
        print(f"\nTranslation setup: Auto-detect → {self.languages[self.target_lang]['name']}")

        self.source_lang = None
        self.conversation_context = ConversationContext(max_history=100, context_window_minutes=60)
        self.history_file = "conversation_history.json"
        self.conversation_context.load_history(self.history_file)

        self.audio_config = AudioConfig()
        self.audio_recorder = AudioRecorder(self.audio_config)
        self.transcriber = WhisperTranscriber()
        self.language_detector = LanguageDetector(self.languages)

        # 🗂️ Preload translators and synthesizers
        self.translators = {
            lang: Translator(self.languages, lang) for lang in self.languages
        }
        self.synthesizers = {
            lang_code: KokoroSynthesizer(
                self.languages[lang_code]["kokoro_code"],
                self.languages[lang_code]["tts_voice"]
            )
            for lang_code in self.languages
        }
        self.synthesizer = self.synthesizers[self.target_lang]

        self.context_manager = self.conversation_context
        self.translator = ContextAwareTranslator(self.translators[self.target_lang], self.languages, self.target_lang, self.context_manager)
        #self.synthesizer = self.synthesizers[self.target_lang]

        self.audio_queue = queue.Queue(maxsize=10)
        self.transcription_queue = queue.Queue(maxsize=10)
        self.translation_queue = queue.Queue(maxsize=5)
        self.tts_queue = queue.Queue(maxsize=5)

        self.running = True
        self.sentence_buffer = ""
        self.recent_translations = set()
        self.last_processed_time = time.time()
        self.last_save_time = time.time()
        self.save_interval = 300
        self.processing_delay = 2.0

    def update_target_language(self, new_lang):
        # Update the internal translator to target new language
        self.translator = ContextAwareTranslator(
            self.translators[new_lang],
            self.languages,
            new_lang,
            self.context_manager
        )
        # Update synthesizer ref too
        self.synthesizer = self.synthesizers[new_lang]

    def change_language(self, new_lang):
        if new_lang not in self.languages or new_lang == self.target_lang:
            return False

        self.target_lang = new_lang
        self.update_target_language(new_lang)  # update translator & synthesizer references
        return True



    def get_target_language(self):
        print("\nAvailable target languages:")
        for code, lang in self.languages.items():
            print(f"{code}: {lang['name']}")
        while True:
            target = input("Enter TARGET language code (en/es/fr): ").lower().strip()
            if target in self.languages:
                return target
            print("Invalid target language")

    def _clear_queues(self):
        for q in [self.audio_queue, self.transcription_queue, self.translation_queue, self.tts_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break

    def _save_conversation_history(self):
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.conversation_context.save_history(self.history_file)
            self.last_save_time = current_time

    def audio_worker(self):
        self.audio_recorder.start(self.audio_queue, self.running)

    def transcription_worker(self):
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
                text, detected_lang = self.transcriber.transcribe(audio_data)

                if text:
                    if not detected_lang or detected_lang not in self.languages:
                        detected_lang = self.language_detector.detect(text)
                    if detected_lang and detected_lang in self.languages and detected_lang != self.target_lang:
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
        while self.running:
            try:
                try:
                    text, source_lang = self.transcription_queue.get(timeout=0.2)
                    if source_lang == self.target_lang:
                        self.transcription_queue.task_done()
                        continue
                    self.sentence_buffer = (self.sentence_buffer + " " + text).strip()
                    self.transcription_queue.task_done()
                except queue.Empty:
                    pass

                current_time = time.time()
                if self.sentence_buffer and (
                    self.translator.base_translator.is_complete_sentence(self.sentence_buffer) or
                    (current_time - self.last_processed_time > self.processing_delay and len(self.sentence_buffer.split()) >= 3)
                ):
                    if self.sentence_buffer not in self.recent_translations and self.source_lang:
                        translation = self.translator.translate(self.sentence_buffer, self.source_lang)
                        if translation:
                            print(f"🔄  Translated to {self.languages[self.target_lang]['name']}: {translation}")
                            self.conversation_context.add_exchange(
                                self.sentence_buffer, self.source_lang,
                                translation, self.target_lang
                            )
                            self.translation_queue.put(translation)
                            self.recent_translations.add(self.sentence_buffer)
                            if len(self.recent_translations) > 10:
                                self.recent_translations.pop()
                            freq = self.conversation_context.get_language_pair_frequency(self.source_lang, self.target_lang)
                            if freq > 1:
                                print(f"    📊 {self.source_lang}→{self.target_lang} used {freq} times")
                    self.sentence_buffer = ""
                    self.last_processed_time = current_time

                self._save_conversation_history()
            except Exception as e:
                print(f"Translation error: {str(e)}")

    def tts_worker(self):
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
        stats = self.conversation_context.get_conversation_stats()
        print("\n" + "=" * 60)
        print("📊 CONVERSATION STATISTICS (MCP)")
        print("=" * 60)
        print(f"Total exchanges: {stats['total_exchanges']}")
        print(f"Recent exchanges: {stats['recent_exchanges']}")
        print(f"Conversation span: {stats['conversation_span_minutes']} minutes")
        print(f"Context window: {stats['context_window_minutes']} minutes")
        print(f"Total topics identified: {stats['total_topics']}")
        if stats['topics']:
            print(f"Topics discussed: {', '.join(stats['topics'][:10])}")
            if len(stats['topics']) > 10:
                print(f"  ... and {len(stats['topics']) - 10} more")
        print("\nLanguage pairs used:")
        for pair, count in stats['language_pairs'].items():
            print(f"  {pair}: {count} times")
        print("=" * 60)

    def export_conversation(self):
        export_file = f"conversation_export_{int(time.time())}.txt"
        self.conversation_context.export_readable_history(export_file)
        print(f"📄 Conversation exported to: {export_file}")

    def start(self):
        try:
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
            print("📂 Conversation history loaded and will be saved automatically")
            print("🧠 MCP (Model Context Protocol) providing intelligent context")
            print("=" * 70)
            print("Commands: lang | stats | export | clear | save | help | Ctrl+C to exit")
            print("=" * 70)

            while self.running:
                try:
                    cmd = input("").strip().lower()
                    if cmd == "lang":
                        print("\nAvailable target languages:")
                        for code, lang in self.languages.items():
                            print(f"  {code}: {lang['name']}")
                        new_lang = input("Enter new target language code: ").lower().strip()
                        if self.change_language(new_lang):
                            print("\n🎧 Listening for speech... (Type command or Ctrl+C to exit)")
                    elif cmd == "stats":
                        self.show_conversation_stats()
                    elif cmd == "export":
                        self.export_conversation()
                    elif cmd == "clear":
                        if input("Clear conversation history? (y/N): ").lower().strip() == 'y':
                            self.conversation_context.clear_history()
                            print("🗑️ Conversation history cleared.")
                    elif cmd == "save":
                        self.conversation_context.save_to_file(self.history_file)
                        print("📂 Conversation history saved.")
                    elif cmd == "help":
                        print("\nCommands: lang | stats | export | clear | save | help | Ctrl+C to exit")
                    elif cmd:
                        print("Unknown command. Type 'help' for available commands.")
                except Exception as e:
                    print(f"Command error: {str(e)}")
        except KeyboardInterrupt:
            print("\n\n🚩 Shutting down translator...")
        finally:
            self.running = False
            self.audio_recorder.stop()
            self.conversation_context.save_history(self.history_file)
            print("📂 Final conversation history saved.")
            print("👋 Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context-Aware Trilingual Real-time Translator with MCP")
    parser.add_argument("--target", "-t", help="Set target language directly (en/es/fr)", choices=['en', 'es', 'fr'])
    parser.add_argument("--history-size", "-hs", type=int, default=100)
    args = parser.parse_args()

    translator = TrilingualTranslator()
    translator.start()
