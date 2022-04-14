from io import BytesIO
from pydub import AudioSegment

# opus_data = BytesIO(opus_audio_bytes)
# sound = AudioSegment.from_file(opus_data, codec="opus")


def convert_webm_to_wav(file_path='/Users/Nakhun/Projects/wav2vec2-live/audios/from_front/20220414T144141.474924.wav'):
    audio = AudioSegment.from_file(file_path, codec="opus")
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export('yes_5.wav', format='wav')





if __name__ == '__main__':

    convert_webm_to_wav()