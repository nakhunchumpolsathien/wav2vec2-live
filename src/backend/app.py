# coding: utf-8
import os
import wave
import pytz
from flask import Flask
from flask import request
from datetime import datetime
from flask_cors import CORS


def write_wav(blob):
    with open(get_file_name(), 'ab') as f:
        f.write(blob)


def get_file_name(log_dir='audios/from_front'):
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

@app.route('/', methods=['POST'])
def home():
    log_dir = ''

    if request.method == 'POST':
        write_wav(request.data)
    return "<h1>HikVisionAPI<h1>"


if __name__ == '__main__':
    # app.run()
    app.run(host="127.0.0.1", port=8080, debug=False)
