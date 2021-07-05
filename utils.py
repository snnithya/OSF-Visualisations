from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import scipy.signal as sig
from librosa.display import waveplot, specshow
from IPython.display import Audio
import parselmouth
sns.set_theme()

def readCycleAnnotation(cyclePath, numDiv, startTime, endTime):
    '''Function to read cycle annotation and add divisions in the middle if required.

    Parameters:
        cyclePath: path to the cycle annotation file
        numDiv: number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime: start time of audio being analysed
        endTime: end time of audio being analysed

    Returns:
        vibhaags: a numpy array of annotations from the file
        matras: a numpy array of division between annotations
    '''

    cycle_df = pd.read_csv(cyclePath)
    index_values = cycle_df.loc[(cycle_df['Time'] >= startTime) & (cycle_df['Time'] <= endTime)].index.values
    vibhaags = cycle_df.iloc[max(index_values[0]-1, 0):min(index_values[-1]+2, cycle_df.shape[0])]
    # add divisions in the middle
    matras = []
    for ind, vibhaag in enumerate(vibhaags['Time'].values[:-1]):
        matras.extend(np.around(np.linspace(vibhaag, vibhaags['Time'].values[ind+1], num = numDiv, endpoint=False), 2)[1:])
    
    return vibhaags, matras

def drawAnnotation(cyclePath, numDiv, startTime, endTime, ax, c='purple'):
    '''Draws annotations on ax

    Parameters
        cyclePath: path to the cycle annotation file
        numDiv: number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime: start time of audio being analysed
        endTime: end time of audio being analysed
        ax: axis to plot in
        c: colour to plot lines in

    Returns
        ax: axis that has been plotted in
    '''
    vibhaags, matras = readCycleAnnotation(cyclePath, numDiv, startTime, endTime)
    if matras is not None:
        for matra in matras:
            ax.axvline(matra - startTime, linestyle='--', c=c)
    if vibhaags is not None:
        for _, vibhaag in vibhaags.iterrows():
            ax.axvline((vibhaag['Time']) - startTime, linestyle='-', c=c)
            ax.annotate(vibhaag['Cycle'], (vibhaag['Time']-startTime, -0.4), bbox=dict(facecolor='grey', edgecolor='white'), c='white')
    return ax

def pitchCountour(audioPath, startTime, endTime, minPitch, maxPitch, notes, tonic, timeStep=0.01, octaveJumpCost=0.9, veryAccurate=True, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, xticks=False, yticks=False):
    '''Returns pitch contour for the audio file

    Uses `plotPitch` to plot pitch contour if ax is not None.

    Parameters
        audioPath: path to audio file
        startTime: time to start reading audio file
        endTime: time to end reading audio file
        minPitch: minimum pitch to read for contour extraction
        maxPitch: maximum pitch to read for contour extraction
        notes: list of note objects indicating notes present in the raga
        tonic: tonic of the audio
        timeStep: time steps in which audio is extracted
        octaveJumpCost: parameter passed to pitch detection function
        veryAccurate: parameter passed to pitch detection function
        ax: axis to plot the pitch contour in
        freqXlabels: time (in seconds) after which each x label occurs
        annotate: if True, will annotate tala markings
        cyclePath: path to file with tala cycle annotations
        numDiv: number of divisions to put between each annotation marking
        xticks: if True, will plot xticklabels
        yticks: if True, will plot yticklabels

    Returns:
        ax: plot of pitch contour if ax was not None
        pitch: pitch object extracted from audio if ax was None
    '''
    
    audio, sr = librosa.load(audioPath, sr=None, mono=True, offset=startTime, duration = endTime - startTime)
    snd = parselmouth.Sound(audio, sr)
    pitch = snd.to_pitch_ac(time_step=timeStep, pitch_floor=minPitch, very_accurate=veryAccurate, octave_jump_cost=octaveJumpCost, pitch_ceiling=maxPitch)
    
    # plot the contour
    if ax is not None:  return plotPitch(pitch, notes, ax, tonic, startTime, endTime, freqXlabels, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, xticks=xticks, yticks=yticks)
    else:   return pitch

def plotPitch(pitch, notes, ax, tonic, startTime, endTime, freqXlabels=5, xticks=True, yticks=True, annotate=False, cyclePath=None, numDiv=0, cAnnot='purple'):
    '''Converts the pitch contour from Hz to Cents, and plots it

    Parameters
        pitch: pitch object from `pitchCountour`
        notes: object for each note used for labelling y-axis
        ax: axis object on which plot is to be plotted
        tonic: tonic (in Hz) of audio clip
        freqXlabels: time (in seconds) after which each x label occurs
        annotate: if true will mark taala markings 
        xticks: if True, will print x tick labels
        yticks: if True, will print y tick labels
        annotate: if True, 
    Returns
        ax: plotted axis
    '''
    yvals = pitch.selected_array['frequency']
    yvals[yvals==0] = np.nan    # mark unvoiced regions as np.nan
    yvals[~(np.isnan(yvals))] = 1200*np.log2(yvals[~(np.isnan(yvals))]/tonic)   #convert Hz to cents
    xvals = pitch.xs()
    ax = sns.lineplot(x=xvals, y=yvals, ax=ax)
    ax.set(xlabel='Time Stamp (s)', 
    ylabel='Notes', 
    title='Pitch Contour (in Cents)', 
    xlim=(0, endTime-startTime), 
    xticks=np.arange(0, endTime-startTime, freqXlabels), 
    xticklabels=np.around(np.arange(startTime, endTime, freqXlabels) )if xticks else [],
    yticks=[x['cents'] for x in notes if (x['cents'] >= min(yvals)) & (x['cents'] <= max(yvals))] if yticks else [], 
    yticklabels=[x['label'] for x in notes if (x['cents'] >= min(yvals)) & (x['cents'] <= max(yvals))] if yticks else [])

    if annotate:
        ax = drawAnnotation(cyclePath, numDiv, startTime, endTime, ax, c=cAnnot)
    return ax

def spectogram(audioPath, startTime, endTime, cmap, ax, amin=1e-5, freqXlabels=5, xticks=False, yticks=False, annotate=False, cyclePath=None, numDiv=0, cAnnot='purple'):
    '''Plots spectogram

    Parameters
        audioPath: path to the audio file
        startTime: time to start reading the audio at
        endTime: time to stop reading audio at
        cmap: colormap to use to plot spectogram
        ax: axis to plot spectogram in
        amin: controls the contrast of the spectogram; passed into librosa.power_to_db function
        freqXlabels: time (in seconds) after which each x label occurs
        xticks: if true, will print x labels
        yticks: if true, will print y labels
        annotate: if True, will annotate tala markings
        cyclePath: path to file with tala cycle annotations
        numDiv: number of divisions to put between each annotation marking
        cAnnot: colour for the annotation marking
    '''
    audio, sr = librosa.load(audioPath, sr=None, mono=True, offset=startTime, duration=endTime-startTime)
    
    # stft params
    winsize = int(np.ceil(sr*40e-3))
    hopsize = int(np.ceil(sr*10e-3))
    nfft = int(2**np.ceil(np.log2(winsize)))

    # STFT
    f,t,X = sig.stft(audio, fs=sr, window='hann', nperseg=winsize, noverlap=(winsize-hopsize), nfft=nfft)
    X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=amin)

    specshow(X_dB, x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopsize, ax=ax, cmap=cmap)
    ax.set(ylabel='Frequency (Hz)', 
    xlabel='', 
    title='Spectogram',
    xlim=(0, endTime-startTime), 
    xticks=np.arange(0, endTime-startTime, freqXlabels) if xticks else [], 
    xticklabels=np.around(np.arange(startTime, endTime, freqXlabels)) if xticks else [],
    ylim=(0, 5000),
    yticks=[0, 2e3, 4e3] if yticks else [], 
    yticklabels=['0', '2k', '4k'] if yticks else [])

    if annotate:
        ax = drawAnnotation(cyclePath, numDiv, startTime, endTime, ax, c=cAnnot)
    return ax

def drawWave(audioPath, startTime, endTime, ax, xticks=False, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, cAnnot='purple'):
    '''Plots the wave plot of the audio

    audioPath: path to the audio file
    startTime: time to start reading the audio at
    endTime: time to stop reading the audio at
    ax: axis to plot waveplot in
    xticks: if True, will plot xticklabels
    freqXlabels: time (in seconds) after which each x label occurs
    annotate: if True, will annotate tala markings
    cyclePath: path to file with tala cycle annotations
    numDiv: number of divisions to put between each annotation marking
    cAnnot: colour for the annotation marking
    '''
    audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=endTime-startTime)
    waveplot(audio, sr, ax=ax)
    ax.set(xlabel='' if not xticks else 'Time (s)', 
    xlim=(0, endTime-startTime), 
    xticks=[] if not xticks else np.arange(0, endTime-startTime, freqXlabels),
    xticklabels=[] if not xticks else np.around(np.arange(startTime, endTime, freqXlabels), 2),
    title='Waveplot')
    if annotate:
        ax = drawAnnotation(cyclePath, numDiv, startTime, endTime, ax, c=cAnnot)
    return ax

def playAudio(audioPath, startTime, endTime):
    '''Plays relevant part of audio

    Parameters
        audioPath: file path to audio
        startTime: time to start reading audio at
        endTime: time to stop reading audio at

    Returns:
        iPython.display.Audio object that plays the audio
    '''
    audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=endTime-startTime)
    return Audio(audio, rate=sr)

def generateFig(noRows, figSize, heightRatios):
    '''Generates a matplotlib.pyplot.figure and axes to plot in

    Axes in the plot are stacked vertically in one column, with height of each axis determined by heightRatios

    Parameters
        noRows: number of rows in the figure
        figSize: (width, height) in inches  of the figure
        heightRatios: list of the fraction of height that each axis should take; len(heightRatios) has to be equal to noRows

    Returns:
        fig: figure object
        axs: list of axes objects
    '''
    if len(heightRatios) != noRows:
        Exception("Length of heightRatios has to be equal to noRows")
    fig = plt.figure(figsize=figSize)
    specs = fig.add_gridspec(noRows, 1, height_ratios = heightRatios)
    axs = [fig.add_subplot(specs[i, 0]) for i in range(noRows)]
    return fig, axs