# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 01:06:59 2018

@author: Pablo
"""

import wave, struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def wavereader(filename):   #The audiofile must be a .wav 
    data = wave.open(filename, 'rb')    #Open the file
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = data.getparams()     #Get file audio info, number of channels, bitrate, sampling frequency, number of samples and compression info
    print(data.getparams())
    codificado = data.readframes(nframes) #little endian in bits, audio samples array
    data.close()
    decodificado = '<{}h'.format(nframes*nchannels) # '<' little endian, '{0}' format
    x = np.array(list(struct.unpack(decodificado, codificado)))*2/(2**(sampwidth*8)) #Byte strings to float array, depending on bitrate
    if nchannels == 1:   #Mono audio
        return x, framerate
    elif nchannels ==2:  #Stereo Audio
        xl = x[1::nchannels]  #Left Channel
        xr = x[0::nchannels]  #Right Channel
        x = xl+ xr
    return x, framerate #x = Audio float array and sampling frequency



def spectrogram(data, fs):     #data = numpy array containing .wavfile data and its sampling frequency
    
    
    duration = len(data)/fs
    nfft = 2**9                                            #fft size
    hopsize = int(nfft * 0.5)                                   #Overlapping size
    zp =  int(np.ceil(len(data)/nfft)*nfft - len(data))         #zeropadding in the end of the audio calcule
    data = np.append(data, np.zeros(zp))                      #Zeropadding
    
    window = np.hanning(nfft)                                   #window for framing
    
    xwarray = np.zeros((int(len(data)/hopsize),nfft))         #Frames array prelocation
    fftarray = np.zeros((int(len(data)/hopsize),nfft*2), dtype=np.complex_)      #FFT result array prelocation
    index = 0
    
    for x in range (int(len(data)/hopsize)-1):
       frame = data[index:index+nfft]                           #segment of the audio for windowing
       xw = frame * window                                      #windowing
       xwarray[x][:] = xw                                      #Windowed frames array
       xwfft = np.append(xw, np.zeros(nfft))                    #Zeropadding for a larger fft (nfft*2)
       fftvec = np.fft.fft(xwfft)                               #fft
       fftarray[x][:] = fftvec                                 #every frame's fft is stored in a array
       index = index + hopsize                                  #Index of the overlapping iteration
    
    [fftarray1, mirror] = np.split(fftarray,2,1)              #Take half of the matrix, fft has a mirror effect at nfft/2
    fftmatrixabs = (np.abs(fftarray1))**2                      #Energy density of the matrix
    fftdb = 20*np.log10(fftmatrixabs) 

    tv2 = np.linspace(0,len(data)/fs,len(data))
    
    #plot audio data
    fig = plt.figure(1)             #Dmain figure
    plt.suptitle('Audio analysis tool', fontsize=16)
    ax0 = plt.subplot(211)          #Audio amplitude subplot
    plt.title(r'Loaded audio', fontsize=10)   #Titulo
    plt.plot(tv2, data)             #Data vs time graph
    plt.xlabel('t(s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, duration)
    
    ax1 = plt.subplot(212)
    plt.title(r'Audio spectre', fontsize=10)
    cm = plt.cm.get_cmap('hot')              #color palette for the spectrogram
    plt.imshow(np.transpose(fftdb), interpolation='nearest', aspect='auto', cmap=cm, origin='lower', extent=[0,duration,0,fs/2])     #imshow for bidimensional arrai plotting
    plt.xlabel('t(s)')
    plt.ylabel('Freq (Hz)')
    cbaxes = inset_axes(ax1, width="30%", height="10%", loc=1)          #Frequency magnitude colorbar
    cbaxes.set_ylabel('Magnitude (db)', rotation=360, labelpad=(-250))                   #Colorbar labeñ
    plt.colorbar(cax=cbaxes, orientation='horizontal')                  #Colorbar orientation
    
    
    plt.show()
              
    return(fftarray1, fftdb)
    
    
    
    
#Example
[data, fs] = wavereader('sine-1000.wav')
[fftarray, fftdb]=spectrogram(data, fs)