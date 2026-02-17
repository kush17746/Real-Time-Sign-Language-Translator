import pyttsx3
import threading

# Global lock to ensure only one voice thread runs at a time
speech_lock = threading.Lock()

class VoiceEngine:
    def speak(self, text):
        """
        Thread-safe speech function.
        If the computer is already speaking, this ignores the new request 
        (prevents crashing).
        """
        # If already speaking, do nothing (return immediately)
        if speech_lock.locked():
            return 

        # Define the task for the thread
        def _speak_task():
            with speech_lock:  # Acquire lock
                try:
                    # Initialize a fresh engine instance for this thread
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"Voice Error: {e}")

        # Run in a separate thread so video doesn't freeze
        t = threading.Thread(target=_speak_task)
        t.start()