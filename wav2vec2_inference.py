import soundfile as sf
import torch
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
start_time = time.time()


# Improvements:
# - gpu / cpu flag
# - convert non 16 khz sample rates
# - inference time log

class Wave2Vec2Inference():
    def __init__(self, model_name):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)


if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2Inference(".models/checkpoint-423000")
    text = asr.file_to_text("audios/th_test_audio_1min.wav")
    print(text)
    print("--- %s seconds ---" % (time.time() - start_time))
