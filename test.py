import glob
import wave
import numpy as np
import os
import math
import  librosa
import sounddevice as sd
from mfccMine import mfccMine
from rmvNan import rmvNan
from disteu import disteu
from train import train
from vqlbg import vqlbg

#def test(file,code):
dir='C:\\Users\\OUISSAL\\Downloads\\data\\myTrain\\'
code=train(dir)
file='C:\\Users\\OUISSAL\\Downloads\\data\\myTest\\LoubnaTest.wav'
with wave.open(file) as f :
    # Get the audio parameters
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    # Read the audio data
    data = f.readframes(nframes)
    samples = np.frombuffer(data, dtype=np.int16)
    samples = samples / np.iinfo(np.int16).max
    y = mfccMine(samples, framerate, nchannels)
    y = rmvNan(y)
    distmin = float('inf')
    filePattern = os.path.join('C:\\Users\\OUISSAL\\Downloads\\data\\myTrain\\', '*.wav')
    theFiles = glob.glob(filePattern)
    for l in range(len(code)):
        d = disteu(y, code[l])
        dist = np.sum(np.min(d, axis=1)) / d.shape[0]
        if dist < distmin:
            distmin = dist
            k1 = l
    print(f'Speaker matches with speaker {theFiles[k1]}')
    #return

#test('C:\\Users\\OUISSAL\\Downloads\\data\\myTest\\OuissalTest.wav',code)