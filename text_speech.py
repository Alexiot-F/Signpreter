"""
text_speech.py
--------------
Text-to-Speech engine supporting both pyttsx3 (offline) and gTTS (online).
Runs speech in a background thread to avoid blocking the main loop.
"""

import threading
import queue
import os


class TextSpeechEngine:
    def __init__(self, prefer_online=False):
        self.prefer_online = prefer_online
        self.engine_type = None
        self.engine = None
        self._speech_queue = queue.Queue()
        self._speaking = False
        self._stop_event = threading.Event()

        self._init_engine()

        self._worker_thread = threading.Thread(
            target=self._speech_worker, daemon=True
        )
        self._worker_thread.start()

    # ---------------- INIT ---------------- #
    def _init_engine(self):
        if not self.prefer_online:
            if self._try_pyttsx3():
                return
        if self._try_gtts():
            return
        if self._try_pyttsx3():
            return

        print("[WARN] No TTS engine available.")

    # ---------------- PYTTSX3 ---------------- #
    def _try_pyttsx3(self):
        try:
            import pyttsx3

            engine = pyttsx3.init()

            # ✅ Slower + clearer speech
            engine.setProperty('rate', 120)   # try 110–130
            engine.setProperty('volume', 1.0)

            # ✅ Voice selection (female + English preferred)
            voices = engine.getProperty('voices')
            selected_voice = None

            for v in voices:
                name = v.name.lower()
                vid = v.id.lower()

                if ("female" in name or "zira" in name or "susan" in name) and \
                   ("en" in vid or "english" in name):
                    selected_voice = v.id
                    break

            # fallback
            if not selected_voice:
                for v in voices:
                    if "en" in v.id.lower() or "english" in v.name.lower():
                        selected_voice = v.id
                        break

            if selected_voice:
                engine.setProperty('voice', selected_voice)

            self.engine = engine
            self.engine_type = "pyttsx3"
            print("[INFO] TTS: pyttsx3 (offline)")

            return True

        except Exception as e:
            print(f"[WARN] pyttsx3 failed: {e}")
            return False

    # ---------------- GTTS ---------------- #
    def _try_gtts(self):
        try:
            from gtts import gTTS
            import pygame

            self.engine_type = "gtts"
            print("[INFO] TTS: gTTS (online, better voice)")
            return True

        except Exception as e:
            print(f"[WARN] gTTS failed: {e}")
            return False

    # ---------------- PUBLIC ---------------- #
    def speak_async(self, text):
        if not text or not text.strip():
            return

        # clear queue (no backlog)
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

        self._speech_queue.put(text.strip())

    def is_speaking(self):
        return self._speaking

    def cleanup(self):
        self._stop_event.set()

        if self.engine_type == "pyttsx3" and self.engine:
            try:
                self.engine.stop()
            except:
                pass

        print("[INFO] TTS stopped.")

    # ---------------- WORKER ---------------- #
    def _speech_worker(self):
        while not self._stop_event.is_set():
            try:
                text = self._speech_queue.get(timeout=0.5)
                self._speak_now(text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Worker: {e}")

    def _speak_now(self, text):
        self._speaking = True

        try:
            if self.engine_type == "pyttsx3":
                self.engine.say(text)
                self.engine.runAndWait()

            elif self.engine_type == "gtts":
                self._speak_gtts(text)

            else:
                print(f"[SPEECH] {text}")

        except Exception as e:
            print(f"[ERROR] Speak failed: {e}")

        finally:
            self._speaking = False

    def _speak_gtts(self, text):
        from gtts import gTTS
        import pygame
        import tempfile
        import time

        tts = gTTS(text=text, lang='en', slow=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name

        tts.save(path)

        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

        pygame.mixer.quit()
        os.unlink(path)


# ✅ GLOBAL INSTANCE
_tts_engine = TextSpeechEngine(prefer_online=False)


def speak(text):
    _tts_engine.speak_async(text)