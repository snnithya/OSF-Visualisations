import os
import sys
sys.path.append('../')
from utils import *
import librosa
import pandas as pd
import json

sys.path.append('../../CommonScripts/')
from common_utils import checkPath

audioFolder = 'Data/OSF/Audio/'
tonicFolder = 'Data/OSF/Tonic/'
ssFolder = 'Data/OSF/SSAudio/'
annotationFolder = 'Data/OSF/Annotations/'
metadataFiles = {'TilakK': 'Data/metadata-TilakK.json', 'RagBah': 'Data/metadata-RagBah.json'}
destFolder = 'Data/OSF/pictures/'

def generateFigure(metadata, dest):
    '''
    Generates figure based on the metadata given, and saves the image in dest

    Parameters
        metadata (dict): dictionary with metadata used to generate figure
        dest (str): file path to store image in
    '''
    fig, axs = generateFig(4, (28, 10), [4, 5, 3, 4])
    # plot waveplot
    axs[0], df = drawWave(audioPath=metadata['vocal_audio_path'], startTime=metadata['start_time'], duration=metadata['duration'], ax=axs[0], annotate=True, onsetPath=metadata['onset_file'], odf=True)
    # plot spetogram 1
    axs[1] = pitchCountour(audioPath=metadata['vocal_audio_path'], startTime=metadata['start_time'], duration=metadata['duration'], minPitch=metadata['min_pitch'], maxPitch=metadata['max_pitch'], notes=metadata['notes'], tonic=metadata['tonic'], ax=axs[1], yticks=True, annotate=True, annotLabel=False, onsetPath=metadata['onset_file'])
    # plot spetogram 2
    axs[2] = spectrogram(audioPath=metadata['vocal_audio_path'], startTime=metadata['start_time'], duration=metadata['duration'], cmap='Blues', ax=axs[2], yticks=True, annotate=True, onsetPath=metadata['onset_file'], xticks=False, freqXlabels=1, annotLabel=False)
    # plot energy contour
    axs[3] = plotEnergy(audioPath=metadata['vocal_audio_path'], startTime=metadata['start_time'], duration=metadata['duration'], ax=axs[3], annotate=True, onsetPath=metadata['onset_file'], xticks=True, freqXlabels=1, annotLabel=False)

    fig.savefig(dest)

def parseFolder(ssFolder=ssFolder):
    '''
    Function to parse audio files

    Parameters
        ssFolder (str): file path to SS audios
    '''
    for root, _, fileNames in os.walk(ssFolder):
        for fileName in fileNames:
            print(f'Processing {fileName}')
            if 'TilakK' in fileName:
                f = open(metadataFiles['TilakK'])
            else:
                f = open(metadataFiles['RagBah'])
            metadata = json.load(f)
            metadata['vocal_audio_path'] = os.path.join(root, fileName)
            metadata['onset_file'] = os.path.join(root, fileName).replace(ssFolder, annotationFolder).rsplit('-', 1)[0] + '.csv'
            dest = checkPath(os.path.join(root, fileName).replace(ssFolder, destFolder).rsplit('-', 1)[0] + '.png')
            generateFigure(metadata, dest)

parseFolder(ssFolder)