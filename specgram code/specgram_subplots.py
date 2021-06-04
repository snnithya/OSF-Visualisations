import os
import librosa
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from librosa import display

#file-paths
audio_dir = '../Fig 1/Data'
file_list=['NIR_VS_Bhoop_Vox.wav', 'NIR_VS_Bhoop_Vox.wav']

#audio loading params
fs = 16000
offset = 0 #seconds
duration = 10 #seconds

#STFT parameters
winsize = int(np.ceil(fs*40e-3))
hopsize = int(np.ceil(fs*10e-3))
nfft = int(2**np.ceil(np.log2(winsize)))

#subplot parameters
n_horiz=len(file_list)//2
n_vert=len(file_list)//n_horiz
fig,ax = plt.subplots(n_vert, n_horiz, figsize=(9,3), sharex=True, sharey=True, squeeze=False)

for i,item in enumerate(file_list):
	#prepare path
	file_path = os.path.join(audio_dir, item)
	if not file_path.endswith('wav'):  file_path+='.wav'

	#load audio, make spectrogram
	x,fs = librosa.load(file_path, sr=fs, duration=duration, offset=offset)
	f,t,X = sig.stft(x, fs=fs, window='hann', nperseg=winsize, noverlap=winsize-hopsize, nfft=nfft)
	X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=1e-5)

	#plot
	img = display.specshow(X_dB, x_axis='time', y_axis='log', sr=fs, fmax=fs//2, hop_length=hopsize, ax=ax[i//n_horiz,i%n_horiz], cmap='magma') #y_axis can be set to 'log' to get log-freq scale - better for large freq. ranges

	#'''
	#remove default axis labels
	ax[i//n_horiz,i%n_horiz].set_xlabel('')
	ax[i//n_horiz,i%n_horiz].set_ylabel('')
	#'''	

	#'''
	#set title
	ax[i//n_horiz,i%n_horiz].set_title(item)
	#'''
	
	'''
	#set new axis labels for each subplot
	ax[i//n_horiz,i%n_horiz].set_xlabel('Time (s)', fontsize=14)
	ax[i//n_horiz,i%n_horiz].set_ylabel('Freq. (Hz)', fontsize=14)
	'''

	#'''
	#set new axis labels only for 1st row and 1st column subplots
	if i%n_horiz==0:
		ax[i//n_horiz,i%n_horiz].set_ylabel('Freq. (Hz)', fontsize=14)
	if (i//n_vert)==(n_vert-1):
		ax[i//n_horiz,i%n_horiz].set_xlabel('Time (s)', fontsize=14)
	#'''

	#'''
	#copy and remove default ticks
	x_ticks = ax[i//n_horiz,i%n_horiz].get_xticks()
	y_ticks = ax[i//n_horiz,i%n_horiz].get_yticks()
	ax[i//n_horiz,i%n_horiz].set_xticks([])
	ax[i//n_horiz,i%n_horiz].set_yticks([])
	#'''

	'''
	#make new ticks - as a subset of default
	x_ticks = x_ticks[1:][::2]
	y_ticks = y_ticks[1:][::2]
	ax[i//n_horiz,i%n_horiz].set_xticks(x_ticks)
	ax[i//n_horiz,i%n_horiz].set_yticks(y_ticks)
	ax[i//n_horiz,i%n_horiz].tick_params(labelsize=12)
	'''

	#'''
	#OR make new ticks - user-defined
	x_ticks = np.arange(0, duration+0.1, 2) #seconds
	y_ticks = np.arange(0, 10e3, 4e3) #hertz
	ax[i//n_horiz,i%n_horiz].set_xticks(x_ticks)
	ax[i//n_horiz,i%n_horiz].set_yticks(y_ticks)
	ax[i//n_horiz,i%n_horiz].set_yticklabels(['0', '4k', '8k'])
	ax[i//n_horiz,i%n_horiz].tick_params(labelsize=12)
	#'''

	#remove minor ticks (only required for log freq plot)
	ax[i//n_horiz,i%n_horiz].tick_params(which='minor', length=0)

	'''
	#add grid if needed
	ax[i//n_horiz,i%n_horiz].grid(axis='y', linewidth=0.9, linestyle='--', color='white')
	'''

'''
#add centered, common axis labels if all subplots share same labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time(s)", fontsize=14)
plt.ylabel("Freq.(Hz)\n", fontsize=14)
'''

plt.tight_layout()
plt.show()
