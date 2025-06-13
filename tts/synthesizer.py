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
        #self.list_audio_devices()
    
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

    
    def speak(self, text):
        print(f"[KokoroSynthesizer] Speaking: {text}")

        """Synthesize and play speech."""
        if not text:
            return
        
        cleaned = text.strip()
        # Skip if text is empty or consists only of dots (e.g. "...", "..", ".")
        if cleaned == '' or all(c == '.' for c in cleaned):
            print(f"Skipping TTS for text: '{text}'")
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
            
    def stop(self):
        """Stop ongoing audio playback and clean resources."""
        try:
            sd.stop()  # Immediately stop any sound playback
            # If your pipeline has a cleanup or close method, call it here:
            if hasattr(self.pipeline, "close"):
                self.pipeline.close()
            elif hasattr(self.pipeline, "cleanup"):
                self.pipeline.cleanup()
            print("KokoroSynthesizer stopped and cleaned up.")
        except Exception as e:
            print(f"Error stopping KokoroSynthesizer: {e}")