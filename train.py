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
from vqlbg import vqlbg
def train(trainDir) :
    filePattern = os.path.join(trainDir, '*.wav')
    theFiles = glob.glob(filePattern)
    code={}
    for i in range(len(theFiles)):

        #with wave.open('C:\\Users\\OUISSAL\\Downloads\\data\\myTrain\\MeS.wav', 'rb') as f:
        with wave.open(theFiles[i]) as f:

            # Get the audio parameters
            params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]

            # Read the audio data
            data = f.readframes(nframes)
            samples = np.frombuffer(data, dtype=np.int16)
            samples = samples / np.iinfo(np.int16).max
        # Print some information about the audio
        #print(f'Number of channels: {nchannels}')
        #print(f'Sample width: {sampwidth}')
        #print(f'Frame rate: {framerate}')
        print(f'Number of frames: {nframes}')
        #sd.play(samples, framerate)
        #sd.wait()
        y=mfccMine(samples,framerate,nchannels)
        #print(y[0:5,0:5])
        y=rmvNan(y)
        code[i]=vqlbg(y,16)
        print(f'now reading {theFiles[i]}')
    return code


dir='C:\\Users\\OUISSAL\\Downloads\\data\\myTrain\\'
code=train(dir)
