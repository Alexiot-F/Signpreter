"""
text_speech.py
--------------
Text-to-Speech engine supporting both pyttsx3 (offline) and gTTS (online).
Runs speech in a background thread to avoid blocking the main loop.
"""

import threading
import queue
import os
import sys


class TextSpeechEngine:
    def __init__(self, prefer_online=False):
        """
        Initialize TTS engine.
        Args:
            prefer_online: If True, prefer gTTS over pyttsx3.
        """
        self.prefer_online = prefer_online
        self.engine_type = None
        self.engine = None
        self._speech_queue = queue.Queue()
        self._speaking = False
        self._stop_event = threading.Event()

        # Try to initialize engine
        self._init_engine()

        # Start background speech worker
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def _init_engine(self):
        """Initialize the best available TTS engine."""
        if not self.prefer_online:
            if self._try_pyttsx3():
                return
        if self._try_gtts():
            return
        if self._try_pyttsx3():
            return
        print("[WARN] No TTS engine available. Speech output disabled.")

    def _try_pyttsx3(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)    # Speed
            engine.setProperty('volume', 0.9)  # Volume
            # Try to select a clearer voice
            voices = engine.getProperty('voices')
            for v in voices:
                if 'english' in v.name.lower() or 'en' in v.id.lower():
                    engine.setProperty('voice', v.id)
                    break
            self.engine = engine
            self.engine_type = "pyttsx3"
            print("[INFO] TTS engine: pyttsx3 (offline)")
            return True
        except Exception as e:
            print(f"[WARN] pyttsx3 not available: {e}")
            return False

    def _try_gtts(self):
        try:
            from gtts import gTTS
            import pygame
            self.engine_type = "gtts"
            print("[INFO] TTS engine: gTTS + pygame (online)")
            return True
        except Exception as e:
            print(f"[WARN] gTTS/pygame not available: {e}")
            return False

    def speak_async(self, text):
        """Queue text for asynchronous speech output."""
        if not text or not text.strip():
            return
        # Clear queue if already speaking to avoid backlog
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        self._speech_queue.put(text.strip())

    def _speech_worker(self):
        """Background worker thread that processes the speech queue."""
        while not self._stop_event.is_set():
            try:
                text = self._speech_queue.get(timeout=0.5)
                self._speak_now(text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Speech worker: {e}")

    def _speak_now(self, text):
        """Actually perform the speech (called from worker thread)."""
        self._speaking = True
        try:
            if self.engine_type == "pyttsx3":
                self._speak_pyttsx3(text)
            elif self.engine_type == "gtts":
                self._speak_gtts(text)
            else:
                # Fallback: print to console
                print(f"[SPEECH] {text}")
        except Exception as e:
            print(f"[ERROR] Speaking failed: {e}")
        finally:
            self._speaking = False

    def _speak_pyttsx3(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def _speak_gtts(self, text):
        from gtts import gTTS
        import pygame
        import tempfile

        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        tts.save(tmp_path)

        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            import time
            time.sleep(0.05)
        pygame.mixer.quit()
        os.unlink(tmp_path)

    def is_speaking(self):
        return self._speaking

    def cleanup(self):
        """Clean up resources."""
        self._stop_event.set()
        if self.engine_type == "pyttsx3" and self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
        print("[INFO] TTS engine stopped.")