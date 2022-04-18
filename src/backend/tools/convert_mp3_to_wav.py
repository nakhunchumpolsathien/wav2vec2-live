import os
from pydub import AudioSegment
from tqdm import tqdm
from glob import glob


def convert_to_wav(src_path, dst_path):
    audio = AudioSegment.from_mp3(src_path)
    audio.export(dst_path, format='wav')


if __name__ == '__main__':
    miniadsmp3_dir = '/Users/Nakhun/Projects/wav2vec2-live'
    miniadswav_dir = ''
    subfolders = [f.path for f in os.scandir(miniadsmp3_dir) if f.is_dir()]
    index = 1
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        dst_dir = os.path.join(miniadswav_dir, subfolder_name)

        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)

        mp3s = glob(os.path.join(subfolder, '*.mp3'))

        for mp3 in tqdm(mp3s, desc=f'{index}/{len(subfolder)}'):
            mp3_name = os.path.basename(mp3).replace('.mp3', '.wav')
            save_path = os.path.join(dst_dir, mp3_name)
            convert_to_wav(mp3, save_path)

        index = index + 1


