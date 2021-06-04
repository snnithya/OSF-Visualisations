import os
import librosa
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from librosa import display

#file-paths
audio_dir = '../Fig 1/Data'
file_name='NIR_VS_Bhoop_Vox.wav'

#audio loading params
fs = 16000
offset = 0 #seconds
duration = 10 #seconds

#STFT parameters
winsize = int(np.ceil(fs*40e-3))
hopsize = int(np.ceil(fs*10e-3))
nfft = int(2**np.ceil(np.log2(winsize)))

#prepare file path
file_path = os.path.join(audio_dir, file_name)
if not file_path.endswith('wav'):  file_path+='.wav'

#load audio, get spectrogram
x,fs = librosa.load(file_path, sr=fs, duration=duration, offset=offset)
f,t,X = sig.stft(x, fs=fs, window='hann', nperseg=winsize, noverlap=winsize-hopsize, nfft=nfft)
X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=1e-5)

#plot
fig,ax = plt.subplots(1,1,figsize=(9,3))
img = display.specshow(X_dB, x_axis='time', y_axis='linear', sr=fs, fmax=fs//2, hop_length=hopsize, ax=ax, cmap='magma') #y_axis can be set to 'log' to get log-freq scale - better for large freq. ranges

#copy and remove default ticks
x_ticks = ax.get_xticks()
y_ticks = ax.get_yticks()
ax.set_xticks([])
ax.set_yticks([])

'''
#make new ticks - as a subset of default
x_ticks = x_ticks[1:][::2]
y_ticks = y_ticks[1:][::2]
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.tick_params(labelsize=12)
'''

#OR make new ticks - user-defined
x_ticks = np.arange(0, duration+0.1, 2)
y_ticks = np.arange(0, 10e3, 4e3)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_yticklabels(['0', '4k', '8k'])
ax.tick_params(labelsize=12)

#remove default axis labels and set new ones
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Spectrogram')
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('Freq. (Hz)', fontsize=14)

#add grid if needed
ax.grid(axis='y', linewidth=0.9, linestyle='--', color='white')

plt.tight_layout()
plt.show()
