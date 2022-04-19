import os
import time
import torch
import warnings
from tqdm import tqdm
import soundfile as sf
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()


class Wave2Vec2Inference():
    def __init__(self, model_name):
        self.kenlm = 'mini_5grams_week_58.binary'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.vocab_dict = self.processor.tokenizer.get_vocab()
        self.sorted_dict = {k.lower(): v for k, v in sorted(self.vocab_dict.items(), key=lambda item: item[1])}
        self.decoder = build_ctcdecoder(list(self.sorted_dict.keys()), self.kenlm)

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000,
                                return_tensors="pt", padding=True).input_values.to(device)

        with torch.no_grad():
            logits = self.model(inputs).logits.cpu().numpy()[0]

        transcription = self.decoder.decode(logits)
        return transcription.lower()

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
        