import numpy as np
import sounddevice as sd
from kokoro import KPipeline

class KokoroSynthesizer:
    """Text-to-speech synthesis using Kokoro."""
    def __init__(self, lang_code, voice):
        self.lang_code = lang_code
        self.voice = voice
        
        print("Initializing Kokoro TTS...")
        self.pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
        self.list_audio_devices()
    
    def list_audio_devices(self):
        """List all available audio output devices."""
        print("\nAvailable audio output devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:  # Only show output devices
                print(f"[{i}] {device['name']} (outputs: {device['max_output_channels']})")
        print(f"\nCurrent default device: {sd.default.device[1]}\n")
    
    def set_output_device(self, device_id):
        """Set the output device for audio playback."""
        try:
            # Validate the device exists
            devices = sd.query_devices()
            if isinstance(device_id, int) and 0 <= device_id < len(devices):
                if devices[device_id]['max_output_channels'] > 0:
                    self.device = device_id
                    print(f"Output device set to: {devices[device_id]['name']}")
                else:
                    print(f"Device {device_id} has no output channels")
            else:
                print(f"Invalid device ID: {device_id}")
        except Exception as e:
            print(f"Error setting output device: {str(e)}")
    def update_language(self, lang_code, voice):
        """Update language settings."""
        # Only reinitialize if needed
        if self.lang_code != lang_code:
            self.lang_code = lang_code
            self.voice = voice
            
            # Stop any current playback
            sd.stop()
            
            # Reinitialize with new language
            print("Reinitializing Kokoro TTS...")
            self.pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
        else:
            # Just update voice if only that changed
            self.voice = voice
    
    def speak(self, text):
        """Synthesize and play speech."""
        if not text:
            return
            
        try:
            audio_data = []
            for _, _, audio in self.pipeline(text, voice=self.voice):
                if audio is not None:
                    audio_data.extend(audio.numpy() if hasattr(audio, 'numpy') else audio)
            
            if audio_data:
                sd.play(np.array(audio_data), samplerate=24000)
                sd.wait()
                
        except Exception as e:
            print(f"TTS synthesis error: {str(e)}")