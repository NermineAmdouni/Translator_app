from flask import Flask, render_template, jsonify, request
from translation.translator2 import Translator
from stt.whisper_transcriber import WhisperTranscriber
from stt.audio_silero_copy import AudioRecorder
from tts.synthesizer import KokoroSynthesizer
from chatbot.voice_chatbot import VoiceChatbot
from language_detection.detector import LanguageDetector
from utils.config import Languages, AudioConfig
from mcp.mcp2 import ConversationContext, ContextAwareTranslator

import threading
import queue
import asyncio
import time

app = Flask(__name__)

class TrilingualTranslator:
    def __init__(self, target_lang='en'):
        # Initialize configurations
        self.languages = Languages().languages
        self.audio_config = AudioConfig()
        self.paused_event = threading.Event()  # When set => running, when cleared => paused
        self.paused_event.set()  # Initially not paused (running)

        # Queues for audio data: async and sync bridge
        self.audio_queue = asyncio.Queue(maxsize=10)        # async queue for AudioRecorder
        self.audio_queue_sync = queue.Queue(maxsize=10)     # sync queue for transcription worker

        # Event loop for async audio recorder
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Core components
        self.audio_recorder = AudioRecorder(self.audio_config, self.audio_queue, self.loop)
        self.transcriber = WhisperTranscriber()
        self.language_detector = LanguageDetector(self.languages)
        self.translators = {
            lang_code: Translator(self.languages, lang_code)
            for lang_code in self.languages
        }
        self.translator = self.translators[target_lang]        

        # Preload all synthesizers at startup
        self.synthesizers = {
            lang_code: KokoroSynthesizer(
                self.languages[lang_code]["kokoro_code"],
                self.languages[lang_code]["tts_voice"]
            )
            for lang_code in self.languages
        }
        self.synthesizer = self.synthesizers[target_lang]
        
        # Translation and transcription queues
        self.transcription_queue = queue.Queue(maxsize=10)
        self.translation_queue = queue.Queue(maxsize=5)
        
        # State management
        self.running = False
        self.source_lang = None
        self.target_lang = target_lang
        self.last_transcription = None
        self.last_translation = None
        self.processing_delay = 2.0
        self.last_processed_time = time.time()
        self.sentence_buffer = ""
        
        # Thread management
        self.threads = []

    def _start_audio_recorder(self):
        def run():
            self.loop.run_until_complete(self.audio_recorder.start())
        threading.Thread(target=run, daemon=True).start()

    def _async_to_sync_audio_queue(self):
        async def bridge():
            while self.running:
                audio_chunk = await self.audio_queue.get()
                try:
                    self.audio_queue_sync.put(audio_chunk, timeout=1)
                except queue.Full:
                    pass
                self.audio_queue.task_done()
        self.loop.create_task(bridge())

    def start(self):
        """Start the translation pipeline."""
        if self.running:
            return False
            
        self.running = True

        # Start audio recorder async loop in thread
        self._start_audio_recorder()
        # Start bridging async queue to sync queue
        self._async_to_sync_audio_queue()

        # Create and start worker threads (no _audio_worker thread anymore)
        self.threads = [
            threading.Thread(target=self._transcription_worker, daemon=True),
            threading.Thread(target=self._translation_worker, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        return True
    
    def stop(self):
        """Stop the translation pipeline."""
        self.running = False
        self.audio_recorder.stop()
        # Stop asyncio loop safely
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1)
        
        self._clear_queues()

    def update_target_language(self, new_lang):
        self.translator = self.translators[new_lang]
        self.synthesizer = self.synthesizers[new_lang]

    def change_language(self, new_lang):
        if new_lang not in self.languages or new_lang == self.target_lang:
            return False

        self.target_lang = new_lang
        self.update_target_language(new_lang)  # update translator & synthesizer references
        return True

    def _clear_queues(self):
        """Clear all processing queues."""
        for q in [self.audio_queue_sync, self.transcription_queue, self.translation_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _transcription_worker(self):
        """Speech-to-text processing thread."""
        while self.running:
            self.paused_event.wait()
            try:
                audio_data = self.audio_queue_sync.get(timeout=0.5)
                
                # Transcribe audio
                text, detected_lang = self.transcriber.transcribe(audio_data)
                
                if text and detected_lang in self.languages:
                    if detected_lang != self.target_lang:
                        self.source_lang = detected_lang
                        self.last_transcription = text
                        self.transcription_queue.put((text, detected_lang))
                
                self.audio_queue_sync.task_done()
            except queue.Empty:
                continue

    def _translation_worker(self):
        """Translation processing thread."""
        while self.running:
            try:
                # Process transcription queue
                try:
                    text, source_lang = self.transcription_queue.get(timeout=0.2)
                    self.sentence_buffer = (self.sentence_buffer + " " + text).strip()
                    self.transcription_queue.task_done()
                except queue.Empty:
                    pass
                
                # Process accumulated text
                current_time = time.time()
                if self.sentence_buffer and (
                    current_time - self.last_processed_time > self.processing_delay or 
                    len(self.sentence_buffer.split()) >= 3
                ):
                    translation = self.translator.translate(self.sentence_buffer, self.source_lang)
                    if translation:
                        self.last_translation = translation
                        self.translation_queue.put(translation)
                        self.synthesizer.speak(translation)
                    
                    self.sentence_buffer = ""
                    self.last_processed_time = current_time
                    
            except Exception as e:
                print(f"Translation error: {str(e)}")

    def get_status(self):
        """Get current translation status."""
        return {
            'running': self.running,
            'paused': not self.paused_event.is_set(),
            'source_lang': self.languages[self.source_lang]['name'] if self.source_lang else None,
            'target_lang': self.languages[self.target_lang]['name'],
            'transcription': self.last_transcription,
            'translation': self.last_translation
        }
    def pause(self):
        if not self.running:
            return False  # Can't pause if not running
        self.paused_event.clear()  # Pause the processing
        return True

    def resume(self):
        if not self.running:
            return False  # Can't resume if not running
        self.paused_event.set()  # Resume the processing
        return True

# Initialize translator instance
translator = TrilingualTranslator()
translator_lock = threading.Lock()

# Initialize chatbot instance
chatbot = VoiceChatbot()
chatbot_lock = threading.Lock()

# Add these new routes
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
        return jsonify({
            'status': 'language_not_changed',
            'previous_language': previous_lang
        }), 400

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
        if translator.start():
            return jsonify({'status': 'started'})
        return jsonify({'status': 'already_running'})

@app.route('/stop', methods=['POST'])
def stop_translation():
    with translator_lock:
        translator.stop()
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
            return jsonify({
                'status': 'language_changed',
                'new_language': new_lang,
                'language_name': translator.languages[new_lang]['name']
            })
        return jsonify({'status': 'language_not_changed'}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

