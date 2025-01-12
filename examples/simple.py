from kokoro_tts import Kokoro
import soundfile as sf

kk = Kokoro()
text = "Hello world! I am Kokoro text-to-speech model. Let's build something fun."
audio = kk.generate(text, kk.AVAILABLE_VOICES[0])
sf.write('simple_test_out.wav', audio, kk.SAMPLING_RATE)
