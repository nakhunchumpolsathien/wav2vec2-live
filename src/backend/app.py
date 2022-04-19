# coding: utf-8
import os
import json
import pytz
import time
import torch
import warnings
import pprint
from io import BytesIO
from flask import Flask
from flask import request
from flask_cors import CORS
from datetime import datetime
from pydub import AudioSegment
from tools.wav2vec2_lm_inference import Wave2Vec2Inference


def write_wav(blob):
    buf = BytesIO()
    opus_data = BytesIO(blob)
    audio = AudioSegment.from_file(opus_data, codec="opus")
    duration = audio.duration_seconds
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    # audio.export(get_file_name(), format='wav')
    return {'file': audio.export(buf, format='wav'), 'duration': duration}


def get_file_name(log_dir='../../audios/from_front'):
    now = datetime.now(pytz.timezone('Asia/Bangkok'))
    name = now.strftime('%Y%m%dT%H%M%S.%f')
    file_name = f'{name}.wav'

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    return os.path.join(log_dir, file_name)


def write_log(log_path, content):
    with open(log_path, 'w') as temp:
        temp.write(content)
        temp.close()


app = Flask(__name__)
CORS(app)

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
asr = Wave2Vec2Inference("checkpoint-900000")


@app.route('/asr', methods=['POST'])
def home():
    if request.method == 'POST':
        start_time = datetime.now()
        wav = write_wav(request.data)
        text = asr.file_to_text(wav['file'])
        res = {'result': text,
               'character_count': len(''.join(text.split())),
               'audio_duration': wav['duration'],
               'receive_time': start_time.isoformat(),
               'return_time': datetime.now().isoformat(),
               'transcribe_time': (datetime.now()-start_time).total_seconds()}
        print(res)
        return json.dumps(res)
    return "ASR"


@app.route('/asr_json', methods=['POST'])
def return_json():
    if request.method == 'POST':
        start_time = time.time()
        text = asr.file_to_text(write_wav(request.data))
        res = {'result': text.encode('utf8'),
               'character_count': len(''.join(text.split())),
               'receive_time': start_time,
               'return_time': time.time()}
        pprint(res)
        return res
    return 'ASR JSON'


if __name__ == '__main__':
    # app.run()
    app.run(host="192.168.10.44", port=8080, debug=False)
