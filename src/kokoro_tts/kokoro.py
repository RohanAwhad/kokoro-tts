# Copied over from https://huggingface.co/hexgrad/Kokoro-82M/blob/c97b7bbc3e60f447383c79b2f94fee861ff156ac/kokoro.py
# class Kokoro is my contribution at EOF

import phonemizer
import re
import torch
import requests
import os
import numpy as np
from onnxruntime import InferenceSession


def split_num(num):
    num = num.group()
    if '.' in num:
        return num
    elif ':' in num:
        h, m = [int(n) for n in num.split(':')]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f'{h} oh {m}'
        return f'{h} {m}'
    year = int(num[:4])
    if year < 1100 or year % 1000 < 10:
        return num
    left, right = num[:2], int(num[2:4])
    s = 's' if num.endswith('s') else ''
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f'{left} hundred{s}'
        elif right < 10:
            return f'{left} oh {right}{s}'
    return f'{left} {right}{s}'

def flip_money(m):
    m = m.group()
    bill = 'dollar' if m[0] == '$' else 'pound'
    if m[-1].isalpha():
        return f'{m[1:]} {bill}s'
    elif '.' not in m:
        s = '' if m[1:] == '1' else 's'
        return f'{m[1:]} {bill}{s}'
    b, c = m[1:].split('.')
    s = '' if b == '1' else 's'
    c = int(c.ljust(2, '0'))
    coins = f"cent{'' if c == 1 else 's'}" if m[0] == '$' else ('penny' if c == 1 else 'pence')
    return f'{b} {bill}{s} and {c} {coins}'

def point_num(num):
    a, b = num.group().split('.')
    return ' point '.join([a, ' '.join(b)])

def normalize_text(text):
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace('«', chr(8220)).replace('»', chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = text.replace('(', '«').replace(')', '»')
    for a, b in zip('、。！，：；？', ',.!,:;?'):
        text = text.replace(a, b+' ')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    text = re.sub(r'\betc\.(?! [A-Z])', 'etc', text)
    text = re.sub(r'(?i)\b(y)eah?\b', r"\1e'a", text)
    text = re.sub(r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)', split_num, text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b', flip_money, text)
    text = re.sub(r'\d*\.\d+', point_num, text)
    text = re.sub(r'(?<=\d)-(?=\d)', ' to ', text)
    text = re.sub(r'(?<=\d)S', ' S', text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", 's', text)
    text = re.sub(r'(?:[A-Za-z]\.){2,} [a-z]', lambda m: m.group().replace('.', '-'), text)
    text = re.sub(r'(?i)(?<=[A-Z])\.(?=[A-Z])', '-', text)
    return text.strip()

def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts

VOCAB = get_vocab()
def tokenize(ps):
    return [i for i in map(VOCAB.get, ps) if i is not None]

phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
    b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True),
)
def phonemize(text, lang, norm=True):
    if norm:
        text = normalize_text(text)
    ps = phonemizers[lang].phonemize([text])
    ps = ps[0] if ps else ''
    # https://en.wiktionary.org/wiki/kokoro#English
    ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
    ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
    ps = re.sub(r'(?<=[a-zɹː])(?=hˈʌndɹɪd)', ' ', ps)
    ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', 'z', ps)
    if lang == 'a':
        ps = re.sub(r'(?<=nˈaɪn)ti(?!ː)', 'di', ps)
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    return ps.strip()

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

@torch.no_grad()
def forward(model, tokens, ref_s, speed):
    device = ref_s.device
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
        c_frame += pred_dur[0,i].item()
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
    return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()

def generate(model, text, voicepack, lang='a', speed=1, ps=None):
    ps = ps or phonemize(text, lang)
    tokens = tokenize(ps)
    if not tokens:
        return None
    elif len(tokens) > 510:
        tokens = tokens[:510]
        print('Truncated to 510 tokens')
    ref_s = voicepack[len(tokens)]
    out = forward(model, tokens, ref_s, speed)
    ps = ''.join(next(k for k, v in VOCAB.items() if i == v) for i in tokens)
    return out, ps


class Kokoro:
    # a => american english | b => british english
    # m => male | f => female
    AVAILABLE_VOICES = [
        'af',
        'af_bella', 'af_sarah', 'am_adam', 'am_michael',
        'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
        'af_nicole', 'af_sky',
    ]
    _AVAILABLE_VOICES_COMMIT_HASH = [
        '3767727882dd08a67a1b91a7513c28dc3887a9e9',
        '3767727882dd08a67a1b91a7513c28dc3887a9e9',
        '3767727882dd08a67a1b91a7513c28dc3887a9e9',
        'b869fc97ed68d0ada08e84f5b4bc6a97e346f0a5',
        'b869fc97ed68d0ada08e84f5b4bc6a97e346f0a5',
        'a67f11354c3e38c58c3327498bc4bd1e57e71c50',
        'a67f11354c3e38c58c3327498bc4bd1e57e71c50',
        'a67f11354c3e38c58c3327498bc4bd1e57e71c50',
        'a67f11354c3e38c58c3327498bc4bd1e57e71c50',
        '8228a351f87c8a6076502c1e3b7e72e821ebec9a',
        '7e9ebc5be7f66a1843b585b63d19d55b5d58ce30',
    ]
    SAMPLING_RATE = 24_000

    def __init__(self, model_url=None, cache_dir=None):
        self.model_url = model_url or self.get_model_url()
        self.cache_dir = cache_dir or self.get_cache_dir()
        self.model_path = os.path.join(self.cache_dir, 'kokoro-v0_19.onnx')

        # lazy loading these
        self._voice = None
        self._curr_voice = None
        self._sess = None

    @staticmethod
    def get_cache_dir():
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache", "kokoro-tts")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def get_model_url(commit_hash=None):
        commit_hash = commit_hash or "3095858c40fc22e28c46429da9340dfda1f8cf28"
        return f'https://huggingface.co/hexgrad/Kokoro-82M/resolve/{commit_hash}/kokoro-v0_19.onnx'

    def get_voice_path(self, voice_name):
        voice_dir = os.path.join(self.cache_dir, 'voices')
        os.makedirs(voice_dir, exist_ok=True)
        return os.path.join(voice_dir, voice_name)

    def download_voice(self, voice_name):
        voice_hash = self._AVAILABLE_VOICES_COMMIT_HASH[self.AVAILABLE_VOICES.index(voice_name)]
        base_url = self.get_model_url(voice_hash).rsplit('/', 1)[0]  # Remove the model filename from the URL
        voice_url = f"{base_url}/voices/{voice_name}.pt"
        voice_path = self.get_voice_path(voice_name)
        print(f'Downloading voice: {voice_name} ...')
        response = requests.get(voice_url)
        response.raise_for_status()
        with open(voice_path, 'wb') as f:
            f.write(response.content)
        print(f'Voice {voice_name} download completed.')
        return voice_path

    def load_voice(self, voice_name):
        voice_path = self.get_voice_path(voice_name)
        if not os.path.exists(voice_path):
            print('Voice not found locally. Downloading now...')
            self.download_voice(voice_name)
        self._voice = torch.load(voice_path, weights_only=True)
        self._curr_voice = voice_name

    def download_model(self):
        print('Downloading model ...')
        response = requests.get(self.model_url)
        response.raise_for_status()
        with open(self.model_path, 'wb') as f:
            f.write(response.content)
        print('Model Download completed.')

    def load_model(self):
        if not os.path.exists(self.model_path):
            self.download_model()
        self._sess = InferenceSession(self.model_path)


    def generate(self, text, voice_name):
        assert isinstance(text, str), f'text should be of type str, but found "{type(text)}"'
        assert voice_name in self.AVAILABLE_VOICES, f'voice_name should be one of "{self.AVAILABLE_VOICES}", but got "{voice_name}"'

        if self._voice is None or voice_name != self._curr_voice:
            self.load_voice(voice_name)

        tokens = tokenize(phonemize(text, lang=voice_name[0]))
        assert len(tokens) <= 510, f'len of tokens generated should be <=510, but got "{len(tokens)}"'
        tokens = [[0, *tokens, 0]]

        if self._sess is None:
            self.load_model()

        ref_s = self._voice[len(tokens)].numpy()
        audio = self._sess.run(None, dict(
            tokens=tokens, 
            style=ref_s,
            speed=np.ones(1, dtype=np.float32)
        ))[0]
        return audio
