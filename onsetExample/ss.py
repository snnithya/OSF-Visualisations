import os
import sys
sys.path.append('../../CommonScripts/')
from extract_pitch_contours import source_separate
from utils import addBack, checkPath

srcFolder = 'Data/OSF/Audio/'
ssDestF = 'Data/OSF/SSAudio/'

for root, _, fileNames in os.walk(os.path.join(srcFolder)):
    for fileName in fileNames:
        if fileName.endswith('mp3') or fileName.endswith('.wav') or fileName.endswith('mp4'):
            if fileName.endswith('-SS.wav'):
                # source separated audio
                continue
            print('Processing ' + os.path.join(fileName))

            ssDest = checkPath(os.path.join(root, fileName).replace(addBack(srcFolder), addBack(ssDestF)).rsplit('.', 1)[0] + '-SS.wav')
            if not os.path.isfile(ssDest):
                source_separate(os.path.join(root, fileName), ssDest, sample_rate_init=48000)