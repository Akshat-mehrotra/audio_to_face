import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

AUDIO_SOURCE = "G:/sounds/rebuilt.vox1_dev_wav_partaa/wav"
IMAGE_SOURCE = "G:/sounds/unzippedFaces"
REPAIR_REQ = True

def create_spectrogram(path,name,dir):
    plt.interactive(False)
    clip, sample_rate = librosa.load(path, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(f'{dir}/{name}', dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del name,clip,sample_rate,fig,ax,S

base_dir = os.getcwd()
test_dir = f"{base_dir}/test"
train_dir = f"{base_dir}/train"

df = pd.read_csv("vox1_meta.csv", sep='\t')

def create_data(target_dir):
    n = 0
    for folder in os.listdir(AUDIO_SOURCE):
        speaker = df[df['VoxCeleb1 ID'] == folder]['VGGFace1 ID'].values[0]

        for subfolder in os.listdir(f'{AUDIO_SOURCE}/{folder}'):

            for file in os.listdir(f'{AUDIO_SOURCE}/{folder}/{subfolder}'):
                name = 'audio.jpg'
                os.makedirs(f'{target_dir}/{n}', exist_ok = True)
                os.makedirs(f'{target_dir}/{n}', exist_ok = True)

                if not os.path.isfile(f'{target_dir}/{n}/audio.jpg'):
                    print(f'making image for {n}')

                    create_spectrogram(f'{AUDIO_SOURCE}/{folder}/{subfolder}/{file}',name,f'{target_dir}/{n}')
                if not os.path.isfile(f'{target_dir}/{n}/image.jpg'):
                    print(f'making image for {n}')
                    imagefolder = f'{IMAGE_SOURCE}/{speaker}/1.6/{subfolder}'
                    img_no = np.random.randint(0,len(os.listdir(imagefolder)))
                    img = os.listdir(imagefolder)[img_no]
                    shutil.copy(f'{imagefolder}/{img}', f'{target_dir}/{n}/image.jpg')
                n+=1

if REPAIR_REQ:
    create_data(train_dir)
