from flask import Flask, render_template, jsonify, request
from translation.translator2 import Translator
from stt.whisper_transcriber import WhisperTranscriber
from stt.audio_recorder import AudioRecorder
from tts.synthesizer import KokoroSynthesizer
from chatbot.voice_chatbot import VoiceChatbot
from language_detection.detector import LanguageDetector
from utils.config import Languages, AudioConfig
from mcp.mcp2 import ConversationContext, ContextAwareTranslator
import signal
import sys
import gc
import threading
import queue
import time
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import asyncio
from contextlib import contextmanager

app = Flask(__name__)

class OptimizedTrilingualTranslator:
    def __init__(self, target_lang='en'):
        # Initialize configurations
        self.languages = Languages().languages
        self.audio_config = AudioConfig()
        self.tts_lock = threading.Lock()

        # Threading controls
        self.paused_event = threading.Event()
        self.paused_event.set()  # Initially running
        self.shutdown_event = threading.Event()
        
        # Core components - initialize all upfront
        self.audio_recorder = AudioRecorder(self.audio_config)
        self.transcriber = WhisperTranscriber()
        self.language_detector = LanguageDetector(self.languages)
        
        # Use thread pool for better resource management
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="TranslatorPool")
        
        # Initialize all translators and synthesizers upfront
        print("Initializing translators and synthesizers for all languages...")
        self.translators = {
            lang_code: Translator(self.languages, lang_code)
            for lang_code in self.languages
        }
        
        self.synthesizers = {
            lang_code: KokoroSynthesizer(
                self.languages[lang_code]["kokoro_code"],
                self.languages[lang_code]["tts_voice"]
            )
            for lang_code in self.languages
        }
        
        self.target_lang = target_lang
        print(f"Initialization complete. Target language: {self.languages[target_lang]['name']}")
        
        # Optimized queues with better sizing
        self.audio_queue = queue.Queue(maxsize=10)  # Reduced size
        self.transcription_queue = queue.Queue(maxsize=20)
        self.translation_queue = queue.Queue(maxsize=10)
        self.tts_queue = queue.Queue(maxsize=5)  # Actually use this queue
        
        # Enhanced state management
        self.running = False
        self.source_lang = None
        self.last_transcription = None
        self.last_translation = None
        
        # Adaptive processing parameters
        self.min_processing_delay = 0.1  # Minimum delay
        self.max_processing_delay = 2.0  # Maximum delay
        self.adaptive_delay = 1.0  # Current adaptive delay
        self.last_processed_time = time.time()
        
        # Improved sentence buffering
        self.sentence_buffer = deque(maxlen=10)
        self.buffer_lock = threading.Lock()
        
        # Performance monitoring
        self.stats = {
            'transcriptions': 0,
            'translations': 0,
            'errors': 0,
            'avg_processing_time': 0.0
        }
        
        # Thread references
        self.worker_threads = []

    @property
    def translator(self):
        """Get translator for current target language."""
        return self.translators[self.target_lang]

    @property
    def synthesizer(self):
        """Get synthesizer for current target language."""
        return self.synthesizers[self.target_lang]

    def start(self):
        """Start the optimized translation pipeline."""
        if self.running:
            return False
            
        self.running = True
        self.shutdown_event.clear()
        
        # Create optimized worker threads
        self.worker_threads = [
            threading.Thread(target=self._audio_worker, name="AudioWorker", daemon=True),
            threading.Thread(target=self._transcription_worker, name="TranscriptionWorker", daemon=True),
            threading.Thread(target=self._translation_worker, name="TranslationWorker", daemon=True),
            threading.Thread(target=self._tts_worker, name="TTSWorker", daemon=True),
            threading.Thread(target=self._stats_worker, name="StatsWorker", daemon=True)
        ]
        
        for thread in self.worker_threads:
            thread.start()
        
        return True

    def stop(self):
        """Stop the translation pipeline with proper cleanup."""
        self.running = False
        self.shutdown_event.set()
        self.audio_recorder.stop()
        
        # Wait for threads to finish gracefully
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Cleanup resources
        self._cleanup_resources()
        self._clear_queues()

    def _cleanup_resources(self):
        """Clean up synthesizers and other resources."""
        for synthesizer in self.synthesizers.values():
            try:
                if hasattr(synthesizer, 'stop'):
                    synthesizer.stop()
                if hasattr(synthesizer, 'cleanup'):
                    synthesizer.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up synthesizer: {e}")
        
        # Clear references
        self.synthesizers.clear()
        self.translators.clear()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)

    def change_language(self, new_lang):
        """Change target language - all components already initialized."""
        if new_lang not in self.languages or new_lang == self.target_lang:
            return False

        self.target_lang = new_lang
        return True

    def _audio_worker(self):
        """Optimized audio recording worker."""
        try:
            self.audio_recorder.start(self.audio_queue, lambda: self.running and not self.shutdown_event.is_set())
        except Exception as e:
            print(f"Audio worker error: {e}")
            self.stats['errors'] += 1

    def _transcription_worker(self):
        """Optimized transcription worker with better error handling."""
        while self.running and not self.shutdown_event.is_set():
            try:
                self.paused_event.wait(timeout=0.1)
                if not self.paused_event.is_set():
                    continue
                
                audio_data = self.audio_queue.get(timeout=0.5)
                start_time = time.time()
                
                # Submit transcription to thread pool for parallel processing
                future = self.executor.submit(self._process_transcription, audio_data)
                
                try:
                    result = future.result(timeout=5.0)  # 5 second timeout
                    if result:
                        text, detected_lang = result
                        if text and detected_lang in self.languages and detected_lang != self.target_lang:
                            self.source_lang = detected_lang
                            self.last_transcription = text
                            self._save_transcription_to_file(text)

                            self.transcription_queue.put((text, detected_lang, time.time()))
                            self.stats['transcriptions'] += 1
                except Exception as e:
                    print(f"Transcription processing error: {e}")
                    self.stats['errors'] += 1
                
                # Update adaptive timing
                processing_time = time.time() - start_time
                self._update_adaptive_delay(processing_time)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription worker error: {e}")
                self.stats['errors'] += 1

    def _process_transcription(self, audio_data):
        """Process transcription in thread pool."""
        try:
            return self.transcriber.transcribe(audio_data)
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def _translation_worker(self):
        """Stream-like translation: process each transcription as it arrives."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Wait for a transcription from the queue
                try:
                    text, lang, timestamp = self.transcription_queue.get(timeout=0.1)
                    self.transcription_queue.task_done()
                except queue.Empty:
                    continue

                if text.strip():
                    # Translate immediately
                    future = self.executor.submit(self._translate_text, text.strip(), lang)
                    translation = future.result(timeout=3.0)

                    if translation and translation.strip() and translation.strip() != '...':
                        self.last_translation = translation
                        self.translation_queue.put(translation)
                        self.tts_queue.put(translation)
                        self.stats['translations'] += 1

            except Exception as e:
                print(f"Translation worker error: {e}")
                self.stats['errors'] += 1
    
    def _combine_texts(self, texts):
        """Intelligently combine multiple text segments."""
        if not texts:
            return ""
        
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in texts:
            if text.lower() not in seen:
                unique_texts.append(text)
                seen.add(text.lower())
        
        return " ".join(unique_texts).strip()

    def _contains_sentence_ending(self, buffer):
        """Check if buffer contains sentence-ending punctuation."""
        if not buffer:
            return False
        
        combined_text = " ".join([item[0] for item in buffer])
        return any(punct in combined_text for punct in '.!?;')

    def _process_translation_buffer(self, source_lang):
        """Process accumulated text in buffer."""
        with self.buffer_lock:
            if not self.sentence_buffer:
                return
            
            # Combine all buffered text
            combined_text = " ".join([item[0] for item in self.sentence_buffer])
            self.sentence_buffer.clear()
        
        if combined_text.strip():
            # Submit translation to thread pool
            future = self.executor.submit(self._translate_text, combined_text, source_lang)
            
            try:
                translation = future.result(timeout=3.0)
                if translation and translation.strip() and translation.strip() != '...':
                    self.last_translation = translation
                    self.translation_queue.put(translation)
                    self.tts_queue.put(translation)  # Actually use the TTS queue
                    self.stats['translations'] += 1
                    
            except Exception as e:
                print(f"Translation processing error: {e}")
                self.stats['errors'] += 1
        
        self.last_processed_time = time.time()
    def _save_transcription_to_file(self, text, filename='transcriptions.txt'):
        """Append a transcription to a file."""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(text.strip() + '\n')
        except Exception as e:
            print(f"[ERROR] Could not save transcription: {e}")

    def _translate_text(self, text, source_lang):
        """Translate text in thread pool."""
        try:
            return self.translator.translate(text, source_lang)
        except Exception as e:
            print(f"Translation error: {e}")
            return None

    def _tts_worker(self):
        """Optimized TTS worker that actually processes the queue."""
        while self.running and not self.shutdown_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.5)
                if text and text.strip() and text.strip() != '...':
                    # Submit TTS to thread pool for non-blocking processing
                    future = self.executor.submit(self._synthesize_speech, text)
                    # Don't wait for completion to allow overlapping TTS
                    
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}")
                self.stats['errors'] += 1

    def _synthesize_speech(self, text):
        """Synthesize speech in thread pool."""
        with self.tts_lock:
            try:
                self.synthesizer.speak(text)
            except Exception as e:
                print(f"TTS synthesis error: {e}")

    def _stats_worker(self):
        """Monitor performance statistics."""
        while self.running and not self.shutdown_event.is_set():
            try:
                time.sleep(5.0)  # Update stats every 5 seconds
                if self.stats['translations'] > 0:
                    print(f"Stats: {self.stats['transcriptions']} transcriptions, "
                          f"{self.stats['translations']} translations, "
                          f"{self.stats['errors']} errors, "
                          f"Adaptive delay: {self.adaptive_delay:.2f}s")
            except Exception:
                continue

    def _update_adaptive_delay(self, processing_time):
        """Update adaptive delay based on processing performance."""
        # Exponential moving average
        alpha = 0.1
        self.adaptive_delay = (1 - alpha) * self.adaptive_delay + alpha * processing_time
        
        # Clamp to reasonable bounds
        self.adaptive_delay = max(self.min_processing_delay, 
                                min(self.max_processing_delay, self.adaptive_delay))

    def _clear_queues(self):
        """Clear all processing queues efficiently."""
        queues = [self.audio_queue, self.transcription_queue, self.translation_queue, self.tts_queue]
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break

    def pause(self):
        """Pause processing."""
        if not self.running:
            return False
        self.paused_event.clear()
        return True

    def resume(self):
        """Resume processing."""
        if not self.running:
            return False
        self.paused_event.set()
        return True

    def get_status(self):
        """Get comprehensive status information."""
        return {
            'running': self.running,
            'paused': not self.paused_event.is_set(),
            'source_lang': self.languages[self.source_lang]['name'] if self.source_lang else None,
            'target_lang': self.languages[self.target_lang]['name'],
            'transcription': self.last_transcription,
            'translation': self.last_translation,
            'stats': self.stats.copy(),
            'adaptive_delay': round(self.adaptive_delay, 2),
            'queue_sizes': {
                'audio': self.audio_queue.qsize(),
                'transcription': self.transcription_queue.qsize(),
                'translation': self.translation_queue.qsize(),
                'tts': self.tts_queue.qsize()
            }
        }

# Initialize with optimized translator
translator = OptimizedTrilingualTranslator()
translator_lock = threading.Lock()

# Initialize chatbot instance
chatbot = VoiceChatbot()
chatbot_lock = threading.Lock()

translator_running = threading.Event()
chatbot_running = threading.Event()

# [Rest of your Flask routes remain the same, just replace 'translator' usage]

# Helper function for safe path check
def is_safe_user_path(path):
    try:
        user_home = os.path.expanduser("~")
        abs_path = os.path.abspath(path)
        abs_home = os.path.abspath(user_home)
        return os.path.commonpath([abs_home]) == os.path.commonpath([abs_home, abs_path])
    except Exception:
        return False

def onexc(func, path, exc):
    print(f"[ERROR] Cannot {func.__name__} on {path}: {exc}")

def safe_rmtree(path):
    if not is_safe_user_path(path):
        raise ValueError(f"Unsafe path detected! Refusing to delete outside user directory: {path}")
    if os.path.exists(path):
        from shutil import rmtree
        rmtree(path, onexc=onexc)
    else:
        print(f"Directory does not exist: {path}")

# Flask routes (keeping your existing routes with minor optimizations)

@app.route('/hybridaction/zybTrackerStatisticsAction')
def ignore_zyb_tracker():
    return '', 204  # Réponse vide sans erreur

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html', 
                         languages=translator.languages,
                         current_lang=translator.target_lang)

@app.route('/start_chatbot', methods=['POST'])
def start_chatbot():
    with chatbot_lock:
        language = request.json.get('language', 'en')
        if chatbot.start(language):
            return jsonify({'status': 'started'})
        return jsonify({'status': 'already_running'})

@app.route('/stop_chatbot', methods=['POST'])
def stop_chatbot():
    with chatbot_lock:
        chatbot.stop()
        return jsonify({'status': 'stopped'})

@app.route('/change_chatbot_language', methods=['POST'])
def change_chatbot_language():
    new_lang = request.json.get('language')
    if not new_lang or new_lang not in translator.languages:
        return jsonify({'error': 'Invalid language'}), 400
    
    with chatbot_lock:
        previous_lang = chatbot.current_language
        if chatbot.change_language(new_lang):
            return jsonify({
                'status': 'language_changed',
                'new_language': new_lang
            })
        
        # Only include previous_language in response if it's not "none"
        response_data = {
            'status': 'language_not_changed'
        }
        if previous_lang and previous_lang.lower() != "none":
            response_data['previous_language'] = previous_lang
            
        return jsonify(response_data), 400

@app.route('/get_chatbot_status', methods=['GET'])
def get_chatbot_status():
    with chatbot_lock:
        return jsonify(chatbot.get_status())

@app.route('/')
def index():
    return render_template('index.html', 
                         languages=translator.languages,
                         current_lang=translator.target_lang)

@app.route('/pause', methods=['POST'])
def pause_translation():
    with translator_lock:
        if translator.pause():
            return jsonify({'status': 'paused'})
        return jsonify({'status': 'not_running'}), 400

@app.route('/resume', methods=['POST'])
def resume_translation():
    with translator_lock:
        if translator.resume():
            return jsonify({'status': 'resumed'})
        return jsonify({'status': 'not_running'}), 400

@app.route('/start', methods=['POST'])
def start_translation():
    with translator_lock:
        if translator_running.is_set():
            return jsonify({'status': 'already_running'}), 400
        
        translator_running.set()
        threading.Thread(target=_start_translator_thread, daemon=True).start()
        return jsonify({'status': 'started'}), 200

def _start_translator_thread():
    try:
        translator.start()
    finally:
        translator_running.clear()

@app.route('/stop', methods=['POST'])
def stop_translation():
    with translator_lock:
        translator.stop()
        translator_running.clear()
        return jsonify({'status': 'stopped'})

@app.route('/status', methods=['GET'])
def get_status():
    with translator_lock:
        return jsonify(translator.get_status())

@app.route('/change_language', methods=['POST'])
def changelanguage():
    new_lang = request.json.get('language')
    if not new_lang or new_lang not in translator.languages:
        return jsonify({'error': 'Invalid language'}), 400

    with translator_lock:
        if translator.change_language(new_lang):
            language_entry = translator.languages.get(new_lang, {})
            language_name = language_entry.get('name') or new_lang

            return jsonify({
                'status': 'language_changed',
                'new_language': new_lang,
                'language_name': language_name
            })
        return jsonify({'status': 'language_not_changed'}), 400

@app.route('/delete-folder', methods=['POST'])
def delete_folder():
    data = request.json
    folder_path = data.get('path')
    try:
        safe_rmtree(folder_path)
        return jsonify({"status": "success", "message": "Folder deleted safely."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

def shutdown_handler(signum, frame):
    print("\nShutting down gracefully...")
    
    with translator_lock:
        translator.stop()
    
    with chatbot_lock:
        chatbot.stop()
    
    # Enhanced cleanup
    for _ in range(3):
        gc.collect()
        time.sleep(0.1)
    
    print("Cleanup completed.")
    sys.exit(0)

# Register graceful shutdown handlers
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)