from flask import Flask, render_template, jsonify, request
from translation.translator import Translator
from stt.whisper_transcriber import WhisperTranscriber
from stt.audio_recorder import AudioRecorder
from tts.synthesizer import KokoroSynthesizer
from chatbot.voice_chatbot import VoiceChatbot
from language_detection.detector import LanguageDetector
from utils.config import Languages, AudioConfig
import threading
import queue
import time

app = Flask(__name__)

class TrilingualTranslator:
    def __init__(self, target_lang='en'):
        # Initialize configurations
        self.languages = Languages().languages
        self.audio_config = AudioConfig()
        self.paused_event = threading.Event()  # When set => running, when cleared => paused
        self.paused_event.set()  # Initially not paused (running)
        # Core components
        self.audio_recorder = AudioRecorder(self.audio_config)
        self.transcriber = WhisperTranscriber()
        self.language_detector = LanguageDetector(self.languages)
        self.translator = Translator(self.languages, target_lang)
        self.synthesizer = KokoroSynthesizer(
            self.languages[target_lang]["kokoro_code"],
            self.languages[target_lang]["tts_voice"]
        )
        
        # Queues for inter-thread communication
        self.audio_queue = queue.Queue(maxsize=10)
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

    def start(self):
        """Start the translation pipeline."""
        if self.running:
            return False
            
        self.running = True
        
        # Create and start worker threads
        self.threads = [
            threading.Thread(target=self._audio_worker, daemon=True),
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
        
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1)
        
        self._clear_queues()

    def change_language(self, new_lang):
        """Change target language."""
        if new_lang not in self.languages or new_lang == self.target_lang:
            return False
            
        self.target_lang = new_lang
        self.translator.update_target_language(new_lang)
        self.synthesizer.update_language(
            self.languages[new_lang]["kokoro_code"],
            self.languages[new_lang]["tts_voice"]
        )
        return True

    def _clear_queues(self):
        """Clear all processing queues."""
        for q in [self.audio_queue, self.transcription_queue, self.translation_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _audio_worker(self):
        """Audio recording thread."""
        self.audio_recorder.start(self.audio_queue, lambda: self.running)

    def _transcription_worker(self):
        """Speech-to-text processing thread."""
        while self.running:
            self.paused_event.wait()
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # Transcribe audio
                text, detected_lang = self.transcriber.transcribe(audio_data)
                
                if text and detected_lang in self.languages:
                    if detected_lang != self.target_lang:
                        self.source_lang = detected_lang
                        self.last_transcription = text
                        self.transcription_queue.put((text, detected_lang))
                
                self.audio_queue.task_done()
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
def change_language():
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

