from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.separator import Separator
import numpy as np
from scipy.signal import resample
import soundfile as sf

def source_separate(src, dest, sample_rate=44100, new_sample_rate=16000):
        '''
        Source Separate voice from the audio recording using Spleeter:4stems

        Parameters
            src (str): source audio file
            dest (str): destination audio file
        
        Returns
            waveform: source separated audio
        '''
        audio_loader = get_default_audio_adapter()
        waveform, _ = audio_loader.load(src, sample_rate=sample_rate)
        separator = Separator('spleeter:4stems', multiprocess=True)
        waveform = separator.separate(waveform)
        
        #convert vocal audio to mono
        waveform = np.mean(waveform['vocals'], axis=1)
        
        #downsample audio to 16kHz
        ratio = float(new_sample_rate)/sample_rate
        n_samples = int(np.ceil(waveform.shape[-1]*ratio))
        waveform = resample(waveform, n_samples, axis=-1)
        if dest is not None:
            sf.write(dest, waveform, new_sample_rate, subtype='PCM_16')
        separator.__del__()

source_separate('Data/VS_Shree_1235_1321.wmv', 'Data/VS_Shree_1235_1321.wav')