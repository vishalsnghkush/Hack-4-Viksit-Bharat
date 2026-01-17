
import threading
import time
import queue

class VoiceCommander:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.is_listening = True
        self.last_command = None
        self.last_command_time = 0
        
        # Try to import speech_recognition
        self.sr_available = False
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.sr_available = True
            # Adjust sensitivity
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
        except ImportError:
            print("[VoiceCommander] Warning: 'speech_recognition' library not found. Voice control disabled.")
            print("Run: pip install SpeechRecognition pyaudio")

        # Start background thread
        if self.sr_available:
            self.bg_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.bg_thread.start()

    def _listen_loop(self):
        import speech_recognition as sr
        
        # List of valid keywords
        KEYWORDS = {
            "start": "START",
            "go": "START",
            "drive": "START",
            "stop": "STOP",
            "halt": "STOP",
            "wait": "STOP",
            "slow": "SLOW_DOWN",
            "speed": "SPEED_UP",
            "faster": "SPEED_UP",
            "park": "PARK",
            "emergency": "EMERGENCY"
        }

        with sr.Microphone() as source:
            print("[VoiceCommander] Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("[VoiceCommander] Listening for commands: Start, Stop, Slow, Speed, Park...")
            
            while self.is_listening:
                try:
                    # Listen for audio (short timeout to allow loop to check is_listening)
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=3.0)
                    
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"[VoiceCommander] Heard: '{text}'")
                        
                        # Keyword matching
                        found_cmd = None
                        for key, cmd in KEYWORDS.items():
                            if key in text:
                                found_cmd = cmd
                                break
                        
                        if found_cmd:
                            self.command_queue.put(found_cmd)
                            self.last_command = found_cmd
                            self.last_command_time = time.time()
                            print(f"[VoiceCommander] COMMAND RECOGNIZED: {found_cmd}")
                            
                    except sr.UnknownValueError:
                        pass # unintelligible
                    except sr.RequestError:
                        print("[VoiceCommander] API Error")
                        
                except sr.WaitTimeoutError:
                    continue # Loop back
                except Exception as e:
                    print(f"[VoiceCommander] Error: {e}")
                    time.sleep(1)

    def get_latest_command(self):
        """Returns the latest command from the queue, or None if empty."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.is_listening = False
