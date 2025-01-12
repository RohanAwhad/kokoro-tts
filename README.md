# Kokoro TTS

This is a simple wrapper on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text to speech model that is super fast.
I like them, but they needed some boilerplate code for ONNX, so i just wrote it down for you.

> It's fast: ~3.8 secs to generate audio of ~15 secs (ONNX Runtime on M1 Mac 8 Core)

## Demo

https://github.com/user-attachments/assets/d1cb269b-3f95-4089-a7f7-be48d13891f1

## How to use it?

Step 1: Install espeak

- For Linux:
- For MacOS:
    ```bash
    brew tap justinokamoto/espeak-ng
    brew install espeak-ng
    ```
    If phonemizer cannot find espeak you can set up env var `PHONEMIZER_ESPEAK_LIBRARY` like: `export PHONEMIZER_ESPEAK_LIBRARY="/opt/homebrew/Cellar/espeak-ng/<version>/bin/espeak"`

Step 2: Install this kokoro-tts
```bash
pip install git+https://github.com/RohanAwhad/kokoro-tts.git
```

Step 3: Generate Audio
```python
from kokoro_tts import Kokoro

kk = Kokoro()
text = "Hello world! I am Kokoro text-to-speech model. Let's build something fun."
audio: "np.ndarray" = kk.generate(text, kk.AVAILABLE_VOICES[0])
```

## Examples

You can also run examples/simple.py and see how you can use `soundfile` to create an audio file.

```bash
pip install soundfile
python examples/simple.py
```

All credits go to the original hexgrad for open-sourcing the model and the code to run it.
