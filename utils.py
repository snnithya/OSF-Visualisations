import warnings
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import scipy.signal as sig
from librosa.display import waveplot, specshow
from IPython.display import Audio, Video 
import parselmouth
import math
import soundfile as sf
import ffmpeg
import os
import cv2
from collections import defaultdict

import pdb
sns.set_theme(rc={"xtick.bottom" : True, "ytick.left" : False, "xtick.major.size":4, "xtick.minor.size":2, "ytick.major.size":4, "ytick.minor.size":2, "xtick.labelsize": 10, "ytick.labelsize": 10})

def readCycleAnnotation(cyclePath, numDiv, startTime, duration):
    '''Function to read cycle annotation and add divisions in the middle if required.


    Parameters
    ----------
    	:param cyclePath (str): path to the cycle annotation file
		:param numDiv (int): number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
		:param startTime (float): start time of audio being analysed
		:param duration (float): duration of the audio to be analysed

    Returns:
        provided (np.ndarray): a numpy array of annotations from the file
        computed (list): a list of division between annotations
    '''

    cycle_df = pd.read_csv(cyclePath)
    index_values = cycle_df.loc[(cycle_df['Time'] >= startTime) & (cycle_df['Time'] <= startTime + duration)].index.values
    if len(index_values) == 0:
        return None, None
    provided = cycle_df.iloc[max(index_values[0]-1, 0):min(index_values[-1]+2, cycle_df.shape[0])]
    # add divisions in the middle
    computed = []
    for ind, val in enumerate(provided['Time'].values[:-1]):
        computed.extend(np.around(np.linspace(val, provided['Time'].values[ind+1], num = numDiv, endpoint=False), 2)[1:])
    return [provided], computed

def readOnsetAnnotation(onsetPath, startTime, duration, onsetKeyword=['Inst']):
    '''Function to read cycle annotation and add divisions in the middle if required.

    Parameters:
        onsetPath (str): path to the cycle annotation file
        startTime (float): start time of audio being analysed
        duration (float): duration of the audio to be analysed
        onsetKeyword (list): list of column names in the onset file to take onsets from

    Returns:
        provided (list): list of numpy arrays of annotations from the file
    '''
    
    onset_df = pd.read_csv(onsetPath)
    provided = []   # variable to store onset timestamps
    for keyword in onsetKeyword:
        provided.append(onset_df.loc[(onset_df[keyword] >= startTime) & (onset_df[keyword] <= startTime + duration)])
    return provided

def drawAnnotation(cyclePath=None, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', numDiv=0, startTime=0, duration=None, ax=None, annotLabel=True, c='purple', alpha=0.8, y=0.7, size=10, textColour='white'):
    '''Draws annotations on ax

    Parameters:
        cyclePath (str): path to the cycle annotation file
        onsetPath (str): path to onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        numDiv (int): number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime (float): start time of audio being analysed
        duration (float): duration of the audio to be analysed
        ax (plt.Axes.axis): axis to plot in
        annotLabel (bool): if True, will print annotation label along with line
        c (str or list): list of colour to plot lines in, one for each onsetTimeKeyword (if provided)
        alpha (float): controls opacity of the annotation lines drawn
        y (float): float value from [0, 1] indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.
        size (int): font size for annotated text
        textColour (str or list): Text colour for annotation. Can be a single string or a list of strings for each onsetTimeKeyword

    Returns
        ax (plt.Axes.axis): axis that has been plotted in
    '''
    if cyclePath is not None:
        provided, computed = readCycleAnnotation(cyclePath, numDiv, startTime, duration)
        timeCol = ['Time']    # name of column with time readings
        labelCol = ['Cycle']  # name of column to extract label of annotation from
        textColours = [textColour]  # colour of text
        c = c if isinstance(c, list) else [c]
    elif onsetPath is not None:
        if annotLabel and type(onsetTimeKeyword) == type(onsetLabelKeyword) and isinstance(onsetTimeKeyword, list):
            # check if length of lists is the same
            if len(onsetTimeKeyword) != len(onsetLabelKeyword):
                raise Exception('Length of onsetTimeKeyword and onsetLabelKeyword should match')
        if type(onsetTimeKeyword) == type(c) and isinstance(onsetTimeKeyword, list):
            # check if length of lists is the same
            if len(onsetTimeKeyword) != len(c):
                raise Exception('Length of onsetTimeKeyword and c should match')   
        timeCol = onsetTimeKeyword if isinstance(onsetTimeKeyword, list) else [onsetTimeKeyword]    # name of column with time readings
        labelCol = onsetLabelKeyword if isinstance(onsetLabelKeyword, list) else [onsetLabelKeyword]  # name of column to extract label of annotation from
        textColours = textColour if isinstance(textColour, list) else [textColour]
        c = c if isinstance(c, list) else [c]
        provided = readOnsetAnnotation(onsetPath, startTime, duration, onsetKeyword=timeCol)
        computed = None
    else:
        raise Exception('A cycle or onset path has to be provided for annotation')
    if computed is not None:
        for computedVal in computed:
            ax.axvline(computedVal - startTime, linestyle='--', c=c[0], alpha=0.4)
    if provided is not None:
        for i, providedListVal in enumerate(provided):
            firstLabel = True   # marker for first line being plotted; to prevent duplicates from occuring in the legend
            for _, providedVal in providedListVal.iterrows():
                ax.axvline((providedVal[timeCol[i]]) - startTime, linestyle='-', c=c[i], label=timeCol[i] if firstLabel and cyclePath is None else '', alpha=alpha)  # add label only for first line of onset for each keyword
                if firstLabel:  firstLabel = False
                if annotLabel:
                    ylims = ax.get_ylim()   # used to set label at 0.7 height of the plot
                    if isinstance(providedVal[labelCol[i]], str):
                        ax.annotate(f"{providedVal[labelCol[i]]}", (providedVal[timeCol[i]]-startTime, (ylims[1]-ylims[0])*y + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c=textColours[i])
                    else:
                        ax.annotate(f"{float(providedVal[labelCol[i]]):g}", (providedVal[timeCol[i]]-startTime, (ylims[1]-ylims[0])*y + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c=textColours[i])
    if onsetPath is not None and cyclePath is None:     # add legend only is onsets are given, i.e. legend is added
        ax.legend()
    return ax

def pitchCountour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, maxPitch=660, notes=None, tonic=220, timeStep=0.01, octaveJumpCost=0.9, veryAccurate=True, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', xticks=False, yticks=False, xlabel=True, ylabel=True, title='Pitch Contour (Cents)', annotLabel=True, cAnnot='purple', ylim=None, annotAlpha=0.8):
    '''Returns pitch contour (in cents) for the audio

    Uses `plotPitch` to plot pitch contour if ax is not None.

    Parameters:
        audio (np.ndarray): loaded audio time series
        sr (int): sample rate of audio time series/ to load the audio at
        audioPath (str): path to audio file; only needed if audio is None
        startTime (float): time to start reading audio file
        duration (float): duration of the audio file to read
        minPitch (float): minimum pitch to read for contour extraction
        maxPitch (float): maximum pitch to read for contour extraction
        notes (dict): list of note objects indicating notes present in the raga
        tonic (float): tonic of the audio
        timeStep (float): time steps in which audio is extracted
        octaveJumpCost (float): parameter passed to pitch detection function
        veryAccurate (bool): parameter passed to pitch detection function
        ax (matplotlib.axes.Axes): axis to plot the pitch contour in
        freqXlabels (float): time (in seconds) after which each x label occurs
        annotate (bool): if True, will annotate tala markings
        cyclePath (str): path to file with tala cycle annotations
        numDiv (int): number of divisions to put between each annotation marking
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetTimeKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        xticks (bool): if True, will plot xticklabels
        yticks (bool): if True, will plot yticklabels
        xlabel (bool): if True, will print xlabel
        ylabel (bool): if True will pring ylabel
        title (str): Title to add to the plot
        annotLabel: if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        cAnnot: input to the ax.annotate function for the colour (c) parameter
        ylim (tuple): (min, max) limits for the y axis; if None, will be directly interpreted from the data
        annotAlpha (float): controls opacity of the annotation lines

    Returns:
        ax (matplotlib.axes.Axes): plot of pitch contour if ax was not None
        (pitchvals, timevals): tuple with arrays of pitch values (in cents) and time stamps; returned if ax was None
    '''
    
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        # if audio is not given, load audio from audioPath
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration

    snd = parselmouth.Sound(audio, sr)
    pitch = snd.to_pitch_ac(time_step=timeStep, pitch_floor=minPitch, very_accurate=veryAccurate, octave_jump_cost=octaveJumpCost, pitch_ceiling=maxPitch)

    pitchvals = pitch.selected_array['frequency']
    pitchvals[pitchvals==0] = np.nan    # mark unvoiced regions as np.nan
    if tonic is None:   Exception('No tonic provided')
    pitchvals[~(np.isnan(pitchvals))] = 1200*np.log2(pitchvals[~(np.isnan(pitchvals))]/tonic)    #convert Hz to cents
    timevals = pitch.xs()
    if ax is None:
        warnings.warn('ax not provided; returning pitch and time values')
        return (pitchvals, timevals)
    else:
        # plot the contour
        return plotPitch(pitchvals, timevals, notes, ax, tonic, startTime, duration, freqXlabels, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, xticks=xticks, yticks=yticks, xlabel=xlabel, ylabel=ylabel, title=title, cAnnot=cAnnot, annotLabel=annotLabel, ylim=ylim, annotAlpha=annotAlpha)

def plotPitch(pitchvals=None, timevals=None, notes=None, ax=None, tonic=None, startTime=0, duration=None, freqXlabels=5, xticks=True, yticks=True, xlabel=True, ylabel=True, title='Pitch Contour (Cents)', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='purple', annotLabel=True, ylim=None, annotAlpha=0.8, yAnnot=0.7, sizeAnnot=10):
    '''Plots the pitch contour

    Parameters:
        pitchvals (np.ndarray): pitch values in cents
        timevals (np.ndarray): time values in seconds
        notes (dict): object for each note used for labelling y-axis
        ax (matplotlib.Axes.axes): axis object on which plot is to be plotted
        tonic (float): tonic (in Hz) of audio clip
        startTime (float): start time for x labels in the plot
        duration (float): duration of audio in the plot (used for x labels)
        freqXlabels (int): time (in seconds) after which each x label occurs
        annotate (bool): if true will mark annotations provided
        xticks (bool): if True, will print x tick labels
        yticks (bool): if True, will print y tick labels
        xlabel (bool): if True, will add label to x axis
        ylabel (bool): if True, will add label to y axis
        title (str): title to add to the plot
        annotate (bool): if True, will add beat annotations to the plot 
        cyclePath (bool): path to file with cycle annotations; used only if annotate is True
        numDiv (int): number of divisions to add between each marked cycle; used only if annotate is True
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        cAnnot: input to the ax.annotate function for the colour (c) parameter; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True
        ylim (tuple): (min, max) limits for the y axis; if None, will be directly interpreted from the data
        annotAlpha (float): controls opacity of the line drawn
        yAnnot (float): float value from [0, 1] indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.
        sizeAnnot (int): font size for annotated text
    Returns
        ax: plotted axis
    '''

    # Check that all required parameters are present
    if pitchvals is None:
        Exception('No pitch contour provided')
    if timevals is None:
        warnings.warn('No time values provided, assuming 0.01 s time steps in pitch contour')
        timevals = np.arange(0, len(pitchvals)*0.01, 0.01)
    if ax is None:
        Exception('ax parameter has to be provided')
    
    # duration = xvals[-1] + 1    # set duration as last x value + 1
    ax = sns.lineplot(x=timevals, y=pitchvals, ax=ax)
    ax.set(xlabel='Time Stamp (s)' if xlabel else '', 
    ylabel='Notes' if ylabel else '', 
    title=title, 
    xlim=(0, duration), 
    xticks=np.around(np.arange(math.ceil(startTime)-startTime, duration, freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
    xticklabels=np.around(np.arange(startTime, duration+startTime, freqXlabels) ).astype(int) if xticks else [])
    if notes is not None and yticks:
        # add yticks if needed
        ax.set(
        yticks=[x['cents'] for x in notes if (x['cents'] >= min(pitchvals[~(np.isnan(pitchvals))])) & (x['cents'] <= max(pitchvals[~(np.isnan(pitchvals))]))] if yticks else [], 
        yticklabels=[x['label'] for x in notes if (x['cents'] >= min(pitchvals[~(np.isnan(pitchvals))])) & (x['cents'] <= max(pitchvals[~(np.isnan(pitchvals))]))] if yticks else [])
    if ylim is not None:
        ax.set(ylim=ylim)

    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime, duration, ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha, y=yAnnot, size=sizeAnnot)
    return ax

def spectrogram(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, winSize=0.04, hopSize=0.01, n_fft=None, cmap='Blues', ax=None, amin=1e-5, freqXlabels=5, xticks=False, yticks=False, xlabel=True, ylabel=True, title='Spectrogram', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='purple', annotLabel=True):
    '''Plots spectrogram

    Parameters:
        audio (np.ndarray): loaded audio time series
        sr (int): sample rate that audio time series is loaded/ is to be loaded in
        audioPath (str): path to the audio file; only needed if audio is None
        startTime (float): time to start reading the audio at
        duration (float): duration of audio
        winSize (float): size of window for STFT in seconds
        hopSize (float): size of hop for STFT in seconds
        n_fft (int): DFT size
        cmap (matplotlib.colors.Colormap or str): colormap to use to plot spectrogram
        ax (plt.Axes.axes): axis to plot spectrogram in
        amin (float): controls the contrast of the spectrogram; passed into librosa.power_to_db function
        freqXlabels (float): time (in seconds) after which each x label occurs
        xticks (bool): if true, will print x labels
        yticks (bool): if true, will print y labels
        xlabel (bool): if true, will add an xlabel
        ylabel (bool): if true, will add a ylabel
        title (str): title for the plot
        annotate (bool): if True, will annotate either tala or onset markings; if both are provided, tala annotations will be marked
        cyclePath (str): path to file with tala cycle annotations
        numDiv (int): number of divisions to put between each tala annotation marking
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        cAnnot: input to the ax.annotate function for the colour (c) parameter; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
    '''
    if ax is None:
        Exception('ax parameter has to be provided')
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration
    
    # convert winSize and hopSize from seconds to samples
    winSize = int(np.ceil(sr*winSize))
    hopSize = int(np.ceil(sr*hopSize))
    if n_fft is None:
        n_fft = int(2**np.ceil(np.log2(winSize)))

    # STFT
    f,t,X = sig.stft(audio, fs=sr, window='hann', nperseg=winSize, noverlap=(winSize-hopSize), nfft=n_fft)
    X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=amin)

    specshow(X_dB, x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopSize, ax=ax, cmap=cmap)
    ax.set(ylabel='Frequency (Hz)' if ylabel else '', 
    xlabel='Time (s)' if xlabel else '', 
    title=title,
    xlim=(0, duration), 
    xticks=(np.arange(0, duration, freqXlabels)) if xticks else [], 
    xticklabels=(np.arange(startTime, duration+startTime, freqXlabels)) if xticks else [],
    ylim=(0, 5000),
    yticks=[0, 2e3, 4e3] if yticks else [], 
    yticklabels=['0', '2k', '4k'] if yticks else [])

    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime, duration, ax, c=cAnnot, annotLabel=annotLabel)
    return ax

def drawWave(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, xticks=False, yticks=True, xlabel=True, ylabel=True, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, cAnnot='purple', annotLabel=True, odf=False, winSize_odf=0.4, hopSize_odf=0.01, nFFT_odf=1024, source_odf='vocal', cOdf='black', title='Waveform'):
    '''Plots the wave plot of the audio

    Parameters:

        audio (np.ndarray): loaded audio time series
        sr (int): sample rate that audio time series is loaded/ is to be loaded in
        audioPath (str): path to the audio file
        startTime (float): time to start reading the audio at
        duration (float): duration of audio to load
        ax (plt.Axes.axes): axis to plot waveplot in
        xticks (bool): if True, will plot xticklabels
        yticks (bool): if True, will plot yticklabels
        xlabel (bool): if True, will add a x label
        ylabel (bool): if True will add a y label
        freqXlabels (float): time (in seconds) after which each x label occurs
        annotate (bool): if True, will annotate tala markings
        cyclePath (str): path to file with tala cycle annotations
        numDiv (int): number of divisions to put between each annotation marking
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        cAnnot: colour for the annotation marking; input to the ax.annotate function for the colour (c) parameter; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        odf (bool): if True, will plot the onset detection function over the wave form
        winSize_odf (float): window size in seconds, fed to the onset detection function; valid only if odf is true
        hopSize_odf (float): hop size in seconds, fed to the onset detection function; valid only if odf is true
        nFFT_odf (int): size of DFT used in onset detection function; valid only if odf is true
        source_odf (str): type of instrument - vocal or pakhawaj, fed to odf; valid only if odf is true
        cOdf: colour to plot onset detection function in; valid only if odf is true
        title (str): title of the plot
    '''
    if ax is None:
        Exception('ax parameter has to be provided')
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration

    waveplot(audio, sr, ax=ax)
    if odf:
        plotODF(audio=audio, sr=sr, startTime=0, duration=None, ax=ax, winSize_odf=winSize_odf, hopSize_odf=hopSize_odf, nFFT_odf=nFFT_odf, source_odf=source_odf, cOdf=cOdf, ylim=True)
    ax.set(xlabel='' if not xlabel else 'Time (s)', 
    ylabel = '' if not ylabel else 'Amplitude',
    xlim=(0, duration), 
    xticks=[] if not xticks else np.around(np.arange(0, duration, freqXlabels)),
    xticklabels=[] if not xticks else np.around(np.arange(startTime, duration+startTime, freqXlabels), 2),
    yticks=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
    yticklabels=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
    title=title)
    if annotate:
        ax = drawAnnotation(cyclePath=cyclePath, onsetPath=onsetPath, numDiv=numDiv, startTime=startTime, duration=duration, ax=ax, c=cAnnot, annotLabel=annotLabel)
    
    return ax

def plotODF(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, winSize_odf=0.4, hopSize_odf=0.01, nFFT_odf=1024, source_odf='vocal', cOdf='black', freqXlabels=5, ylim=True, xlabel=False, ylabel=False, xticks=False, yticks=False, title='Onset Detection Function'):
    '''
    Plots onset detection function if ax is provided. If not returns an a tuple with 2 arrays - onset detection function values and time stamps

    audio (np.ndarray): loaded audio time series
    sr (int): sample rate that audio time series is loaded/ is to be loaded in
    audioPath (str): path to the audio file
    startTime (float): time to start reading the audio at
    duration (float): duration of audio to load
    ax (plt.Axes.axes): axis to plot waveplot in
    winSize_odf (float): window size in seconds, fed to the onset detection function
    hopSize_odf (float): hop size in seconds, fed to the onset detection function
    nFFT_odf (int): size of DFT used in onset detection function
    source_odf (str): type of instrument - vocal or pakhawaj, fed to odf
    cOdf: colour to plot onset detection function in
    freqXlabels (float): time (in seconds) after which each x label occurs
    ylim (bool): if True, will reset the ylim to the range of the output of the ODF function; this is added because when the ODF is plotted over another plot, say the waveform, it is easier to see if the ylim is readjusted
    xticks (bool): if True, will plot xticklabels
    yticks (bool): if True, will plot yticklabels
    xlabel (bool): if True, will add a x label
    ylabel (bool): if True will add a y label
    title (str): title of the plot

    Returns
        ax (matplotlib.Axes.axes): if ax is not None, returns a plot
        (odf_vals, time_vals): if ax is None, returns a tuple with ODF values and time stamps.
    '''

    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration

    odf_vals, _, _ = getOnsetActivation(x=audio, audioPath=None, startTime=startTime, endTime=duration+startTime, fs=sr, winSize=winSize_odf, hopSize=hopSize_odf, nFFT=nFFT_odf, source=source_odf)
    
    # set time and odf values in variables
    time_vals = np.arange(0, duration, hopSize_odf)
    odf_vals = odf_vals[:-1]    # disregard the last frame of odf_vals since it is centered around the frame at time stamp 'duration'

    if ax is None:
        # if ax is None, return (odf_vals, time_vals)
        return (odf_vals, time_vals)
    else:
        ax.plot(time_vals, odf_vals, c=cOdf)     # plot odf_vals and consider odf_vals for all values except the last frame
        max_abs_val = max(abs(min(odf_vals)), abs(max(odf_vals)))   # find maximum value to set y limits to ensure symmetrical plot
        # set ax parameters only if they are not None
        ax.set(xlabel='' if not xlabel else 'Time (s)', 
        ylabel = '' if not ylabel else 'ODF',
        xlim=(0, duration), 
        xticks=[] if not xticks else np.around(np.arange(0, duration, freqXlabels)),
        xticklabels=[] if not xticks else np.around(np.arange(startTime, duration+startTime, freqXlabels), 2),
        yticks=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
        yticklabels=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
        ylim= ax.get_ylim() if not ylim else (-max_abs_val, max_abs_val),
        title=title) 
        return ax

def plotEnergy(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, xticks=False, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, cAnnot='purple', annotLabel=True, winSize_odf=0.4, hopSize_odf=0.01, nFFT_odf=1024, source_odf='vocal', cOdf='black'):
    '''
    For debugging puposes only - plots energy function used to calculate odf

    Parameters:
        audio: loaded audio time series
        sr: sample rate that audio time series is loaded/ is to be loaded in
        audioPath: path to the audio file
        startTime: time to start reading the audio at
        duration: duration of audio to load
        ax: axis to plot waveplot in
        xticks: if True, will plot xticklabels
        freqXlabels: time (in seconds) after which each x label occurs
        annotate: if True, will annotate tala markings
        cyclePath: path to file with tala cycle annotations
        numDiv: number of divisions to put between each annotation marking
        onsetPath: path to file with onset annotations; only considered if cyclePath is None
        cAnnot: colour for the annotation marking
        annotLabel: if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        odf: if True, will plot the onset detection function over the wave form
        winSize_odf: window size, fed to the onset detection function; valid only if odf is true
        hopSize_odf: hop size in seconds, fed to the onset detection function; valid only if odf is true
        nFFT_odf: size of DFT used in onset detection function; valid only if odf is true
        source_odf: type of instrument - vocal or pakhawaj, fed to odf; valid only if odf is true
        cOdf: colour to plot onset detection function in; valid only if odf is true
    '''
    if ax is None:
        Exception('ax parameter has to be provided')
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
        audio /= np.max(np.abs(audio))
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration
    ax.set(xlabel='' if not xticks else 'Time (s)', 
    xlim=(0, duration), 
    xticks=[] if not xticks else np.around(np.arange(0, duration, freqXlabels)),
    xticklabels=[] if not xticks else np.around(np.arange(startTime, duration+startTime, freqXlabels), 2),
    title='Energy Contour', 
    ylabel='dB')

    _, _, energy = getOnsetActivation(x=audio, audioPath=None, startTime=0, endTime=duration, fs=sr, winSize=winSize_odf, hopSize=hopSize_odf, nFFT=nFFT_odf, source=source_odf)
    ax.plot(np.arange(0, duration, hopSize_odf), energy[:-1], c=cOdf)
    if annotate:
        ax = drawAnnotation(cyclePath=cyclePath, onsetPath=onsetPath, numDiv=numDiv, startTime=startTime, duration=duration, ax=ax, c=cAnnot, annotLabel=annotLabel)
    return ax
    
def playAudio(audio=None, sr=16000, audioPath=None, startTime=0, duration=None):
    '''Plays relevant part of audio

    Parameters:
        audio: loaded audio sample
        sr: sample rate of audio; valid only if audio is not None
        audioPath: path to audio file
        startTime: time to start reading audio at
        duration: duration of the audio to load

    Returns:
        iPython.display.Audio object that plays the audio
    '''
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=duration)
    return Audio(audio, rate=sr)

def playAudioWClicks(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, onsetFile=None, onsetLabels=['Inst', 'Tabla'], destPath=None):
    '''Plays relevant part of audio along with clicks at timestamps in onsetTimes

    Parameters:
        audio (np.ndarray): loaded audio sample
        sr (float): sample rate of audio
        audioPath (str): path to audio file
        startTime (float): time to start reading audio at
        duration (float): duration of the audio to load
        onsetFile (str): file path to onset values
        onsetLabels (str): column names in onsetFile to mark with clicks
        destPath (str): path to save audio file at; if None, will not save any audio file

    Returns:
        iPython.display.Audio object that plays the audio
    '''

    if audio is None:
        audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio)
    onsetFileVals = pd.read_csv(onsetFile)
    onsetTimes = []
    for onsetLabel in onsetLabels:
        onsetTimes.append(onsetFileVals.loc[(onsetFileVals[onsetLabel] >= startTime) & (onsetFileVals[onsetLabel] <= startTime+duration), onsetLabel].values)
    clickTracks = [librosa.clicks(onsetTime-startTime, sr=sr, length=len(audio), click_freq=1000*(2*i+1)) for i, onsetTime in enumerate(onsetTimes)]
    audioWClicks = 0.8*audio  # add clicks to this variable
    for clickTrack in clickTracks:
        audioWClicks += 0.2/len(clickTracks)*clickTrack
    if destPath is not None:
        sf.write(destPath, audioWClicks, sr)
    return Audio(audioWClicks, rate=sr)

def playVideo(video=None, videoPath=None, startTime=0, duration=None, destPath='Data/Temp/VideoPart.mp4', videoOffset=0):
    '''Plays relevant part of audio

    Parameters:
        video (np.ndarray): loaded video sample
        videoPath (str): path to video file
        startTime (float): time to start reading the video from
        duration (float): duration of the video to load
        destPath (str): path to store shortened video
        videoOffset (float): number of seconds offset between video and audio; time in audio + videioOffset = time in video
    Returns:
        iPython.display.Video object that plays the video
    '''
    if video is None:
        if duration is None and startTime == 0:
            # play the entire video
            return Video(videoPath, embed=True)
        else:
            # store a shortened video in destPath
            vid = ffmpeg.input(videoPath)
            joined = ffmpeg.concat(
            vid.video.filter('trim', start=startTime+videoOffset, duration=duration).filter('setpts', 'PTS-STARTPTS'),
            vid.audio.filter('atrim', start=startTime+videoOffset, duration=duration).filter('asetpts', 'PTS-STARTPTS'),
            v=1,
            a=1
            ).node
            v3 = joined['v']
            a3 = joined['a']
            out = ffmpeg.output(v3, a3, destPath).overwrite_output()
            out.run()
            return Video(destPath, embed=True)
    else:
        return Video (data=video, embed=True)

def generateFig(noRows, figSize=(14, 7), heightRatios=None):
    '''Generates a matplotlib.pyplot.figure and axes to plot in

    Axes in the plot are stacked vertically in one column, with height of each axis determined by heightRatios

    Parameters:
        noRows: number of rows in the figure
        figSize: (width, height) in inches  of the figure
        heightRatios: list of the fraction of height that each axis should take; len(heightRatios) has to be equal to noRows

    Returns:
        fig: figure object
        axs: list of axes objects
    '''
    if len(heightRatios) != noRows:
        Exception("Length of heightRatios has to be equal to noRows")
    if heightRatios is None:
        # if heightRatios is None
        heightRatios = np.ones(noRows)
    fig = plt.figure(figsize=figSize)
    specs = fig.add_gridspec(noRows, 1, height_ratios = heightRatios)
    axs = [fig.add_subplot(specs[i, 0]) for i in range(noRows)]
    return fig, axs

def to_dB(x, C):
    '''Applies logarithmic (base 10) transformation
    
    Parameters:
        x: input signal
        C: scaling constant
    
    Returns
        log-scaled x
    '''
    return np.log10(1 + x*C)/(np.log10(1+C))

def subBandEner(X,fs,band):
    '''Computes spectral sub-band energy (suitable for vocal onset detection)

    Parameters:
        X: STFT of an audio signal x
        fs: sampling rate
        band: edge frequencies (in Hz) of the sub-band of interest
        
    Returns
        sbe: array with each value representing the magnitude STFT values in a short-time frame squared & summed over the sub-band
    '''

    binLow = int(np.ceil(band[0]*X.shape[0]/(fs/2)))
    binHi = int(np.ceil(band[1]*X.shape[0]/(fs/2)))
    sbe = np.sum(np.abs(X[binLow:binHi])**2, 0)
    return sbe

def biphasicDerivative(x, tHop, norm=1, rectify=1):
    '''Computes a biphasic derivative (See [1] for a detailed explanation of the algorithm)
    
    Parameters:
        x: input signal
        tHop: frame- or hop-length used to obtain input signal values (reciprocal of sampling rate of x)
        norm: if output is to be normalized
        rectify: if output is to be rectified to keep only positive values (sufficient for peak-picking)
    
    Returns
        x: after performing the biphasic derivative of input x (i.e, convolving with a biphasic derivative filter)

    [1] Rao, P., Vinutha, T.P. and Rohit, M.A., 2020. Structural Segmentation of Alap in Dhrupad Vocal Concerts. Transactions of the International Society for Music Information Retrieval, 3(1), pp.137–152. DOI: http://doi.org/10.5334/tismir.64
    '''

    n = np.arange(-0.1, 0.1, tHop)
    tau1 = 0.015  # = (1/(T_1*sqrt(2))) || -ve lobe width
    tau2 = 0.025  # = (1/(T_2*sqrt(2))) || +ve lobe width
    d1 = 0.02165  # -ve lobe position
    d2 = 0.005  # +ve lobe position
    A = np.exp(-pow((n-d1)/(np.sqrt(2)*tau1), 2))/(tau1*np.sqrt(2*np.pi))
    B = np.exp(-pow((n+d2)/(np.sqrt(2)*tau2), 2))/(tau2*np.sqrt(2*np.pi))
    biphasic = A-B
    x = np.convolve(x, biphasic, mode='same')
    x = -1*x

    if norm==1:
        x/=np.max(x)
        x-=np.mean(x)

    if rectify==1:
        x*=(x>0)
    return x

def getOnsetActivation(x=None, audioPath=None, startTime=0, endTime=None, fs=16000, winSize=0.4, hopSize=0.01, nFFT=1024, source='vocal'):
    '''Computes onset activation function

    Parameters:
        x: audio signal array
        audioPath: path to the audio file
        startTime: time to start reading the audio at
        endTime: time to stop reading audio at
        fs: sampling rate to read audio at
        winSize: window size in seconds for STFT
        hopSize: hop size in seconds for STFT
        nFFT: DFT size
        source: choice of instrument - vocal or pakhawaj

    Returns
        odf: the frame-wise onset activation function (at a sampling rate of 1/hopSize)
        onsets: time locations of detected onset peaks in the odf (peaks detected using peak picker from librosa)
    '''

    winSize = int(np.ceil(winSize*fs))
    hopSize = int(np.ceil(hopSize*fs))
    nFFT = int(2**(np.ceil(np.log2(winSize))))
    
    if x is not None:
        x = fadeIn(x,int(0.5*fs))
        x = fadeOut(x,int(0.5*fs))
        if endTime is not None:
            x = x[int(np.ceil(startTime*fs)):int(np.ceil(endTime*fs))]
        else:
            # if end time is None, consider the whole audio
            x = x[int(np.ceil(startTime*fs)):]
    elif audioPath is not None:
        x, _ = librosa.load(audioPath, sr=fs, offset=startTime, duration=endTime-startTime)
    else:
        print('Provide either the audio signal or path to the stored audio file on disk')
        raise

    X,_ = librosa.magphase(librosa.stft(x,win_length=winSize, hop_length=hopSize, n_fft=nFFT))

    if source=='vocal':
        sub_band = [600,2500]
        odf = subBandEner(X, fs, sub_band)
        odf = to_dB(odf, 100)
        # energy = odf.copy()
        odf = biphasicDerivative(odf, hopSize/fs, norm=1, rectify=1)

        onsets = librosa.onset.onset_detect(onset_envelope=odf.copy(), sr=fs, hop_length=hopSize, pre_max=4, post_max=4, pre_avg=6, post_avg=6, wait=50, delta=0.12)*hopSize/fs

    else:
        sub_band = [0,fs/2]
        odf = spectralFlux(X, fs, sub_band, aMin=1e-4, normalize=True)
        # energy = odf.copy()
        odf = biphasicDerivative(odf, hopSize, norm=1, rectify=1)

        onsets = librosa.onset.onset_detect(onset_envelope=odf, sr=fs, hop_length=hopSize, pre_max=1, post_max=1, pre_avg=1, post_avg=1, wait=10, delta=0.05)*hopSize/fs

    return odf, onsets #, energy

def spectralFlux(X, fs, band, aMin=1e-4, normalize=True):
    '''Computes 1st order rectified spectral flux (difference) of a given STFT input
    
    Parameters:
        X: input STFT matrix
        fs: sampling rate of audio signal
        band: frequency band over which to compute flux from STFT (sub-band spectral flux)
        aMin: lower threshold value to prevent log(0)
        normalize: whether to normalize output before returning

    Returns
        specFlux: array with frame-wise spectral flux values
    '''

    X = 20*np.log10(aMin+abs(X)/np.max(np.abs(X)))
    binLow = int(band[0]*X.shape[0]/(fs/2))
    binHi = int(band[1]*X.shape[0]/(fs/2))
    specFlux = np.array([0])
    for hop in range(1,X.shape[1]):
        diff = X[binLow:binHi,hop]-X[binLow:binHi,hop-1]
        diff = (diff + abs(diff))/2
        specFlux=np.append(specFlux,sum(diff))
    if normalize:
        specFlux/=max(specFlux)
    return specFlux
    
def compute_local_average(x, M, Fs=1):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        M: Determines size (2M+1*Fs) of local average
        Fs: Sampling rate

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    M = int(np.ceil(M * Fs))
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def spectral_flux_fmp(x, Fs=1, N=1024, W=640, H=80, gamma=100, M=20, norm=1, band=[]):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        M: Size (frames) of local average
        norm: Apply max norm (if norm==1)
        band: List of lower and upper spectral freq limits

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=W, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
	
	#if vocal-band SF
    if len(band)!=0: 
        band = np.array(band)*(N/2+1)/Fs
        Y = Y[int(band[0]):int(band[1]),:]

    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm == 1:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum

def fadeIn(x,length):
	fade_func = np.ones(len(x))
	fade_func[:length] = np.hanning(2*length)[:length]
	x*=fade_func
	return x

def fadeOut(x,length):
	fade_func = np.ones(len(x))
	fade_func[-length:] = np.hanning(2*length)[length:]
	x*=fade_func
	return x

def ACF_DFT_sal(signal, t_ACF_lag, t_ACF_frame, t_ACF_hop, fs):
    n_ACF_lag = int(t_ACF_lag*fs)
    n_ACF_frame = int(t_ACF_frame*fs)
    n_ACF_hop = int(t_ACF_hop*fs)
    signal = subsequences(signal, n_ACF_frame, n_ACF_hop)
    ACF = np.zeros((len(signal), n_ACF_lag))
    for i in range(len(ACF)):
        ACF[i][0] = np.dot(signal[i], signal[i])
        for j in range(1, n_ACF_lag):
            ACF[i][j] = np.dot(signal[i][:-j], signal[i][j:])
    DFT = (abs(np.fft.rfft(signal)))
    sal = np.zeros(len(ACF))
    for i in range(len(ACF)):
        sal[i] = max(ACF[i])
    for i in range(len(ACF)):
        if max(ACF[i])!=0:
            ACF[i] = ACF[i]/max(ACF[i])
        if max(DFT[i])!=0:
            DFT[i] = DFT[i]/max(DFT[i])
    return (ACF, DFT, sal)

def subsequences(signal, frame_length, hop_length):
    shape = (int(1 + (len(signal) - frame_length)/hop_length), frame_length)
    strides = (hop_length*signal.strides[0], signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

def tempo_period_comb_filter(ACF, fs, norm=1):
    L = np.shape(ACF)[1]
    min_lag = 10
#    max_lag = L/2    #For madhya laya
    max_lag = L     # For Vil bandish
#    max_lag = 66    # For Drut gats
    N_peaks = 11    # 11 for Madhya laya & 9 for Drut gat  
    
 
    window = zeros((L, L))
    for j in range(min_lag, max_lag):
        C = j*np.arange(1, N_peaks)
        D = np.concatenate((C, C+1, C-1, C+2, C-2, C+3, C-3))
        D = D[D<L]
        norm_factor = len(D)
        if norm == 1:
            window[j][D] = 1.0/norm_factor
        else:
            window[j][D] = 1.0
            
    tempo_candidates = np.dot(ACF, transpose(window))
    
#    re_weight = zeros(L)
#    re_weight[min_lag:max_lag] = linspace(1, 0.5, max_lag-min_lag)
#    tempo_candidates = tempo_candidates*re_weight
    tempo_lag = np.argmax(tempo_candidates, axis=1)/float(fs)
    return (window, tempo_candidates, tempo_lag)

def viterbi_tempo_rhythm(tempo_candidates, fs, transition_penalty):
    T = np.shape(tempo_candidates)[0]
    L = np.shape(tempo_candidates)[1]

    p1=transition_penalty

    cost=ones((T,L//2))*1e8
    m=zeros((T,L//2))

#    cost[:,0]=1000
#    m[0][1]=argmax(tempo_candidates[0])
#    for j in range(1,L):
#        cost[1][j]=abs(60*fs/j-60*fs/m[0][1])/tempo_candidates[1][j]
#        m[1][j]=m[0][1]/fs

    for i in range(1,T):
        for j in range(1,L//2):
            cost[i][j]=cost[i-1][1]+p1*abs(60.0*fs/j-60.0*fs)-tempo_candidates[i][j]
            for k in range(2,L//2):
                if cost[i][j]>cost[i-1][k]+p1*abs(60.0*fs/j-60.0*fs/k)-tempo_candidates[i][j]:
                    cost[i][j]=cost[i-1][k]+p1*abs(60.0*fs/j-60.0*fs/k)-tempo_candidates[i][j]
                    m[i][j]=int(k)
                    
    tempo_period=zeros(T)
    tempo_period[T-1]=argmin(cost[T-1,1:])/float(fs)
    t=int(m[T-1,argmin(cost[T-1,1:])])
    i=T-2
    while(i>=0):
        tempo_period[i]=t/float(fs)
        t=int(m[i][t])
        i=i-1
    return tempo_period

def plot_matrix(X, Fs=1, Fs_F=1, T_coef=None, F_coef=None, xlabel='Time (seconds)', ylabel='Frequency (Hz)', title='',
                dpi=72, colorbar=True, colorbar_aspect=20.0, ax=None, figsize=(6, 3), **kwargs):
    """Plot a matrix, e.g. a spectrogram or a tempogram
    
    Notebook: B/B_PythonVisualization.ipynb in [2])
 
    Args:
        X: The matrix
        Fs: Sample rate for axis 1
        Fs_F: Sample rate for axis 0
        T_coef: Time coeffients. If None, will be computed, based on Fs.
        F_coef: Frequency coeffients. If None, will be computed, based on Fs_F.
        xlabel: Label for x axis
        ylabel: Label for y axis
        title: Title for plot
        dpi: Dots per inch
        colorbar: Create a colorbar.
        colorbar_aspect: Aspect used for colorbar, in case only a single axes is used.
        ax: Either (1.) a list of two axes (first used for matrix, second for colorbar), or (2.) a list with a single
            axes (used for matrix), or (3.) None (an axes will be created).
        figsize: Width, height in inches
        **kwargs: Keyword arguments for matplotlib.pyplot.imshow

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = [ax]
    if T_coef is None:
        T_coef = np.arange(X.shape[1]) / Fs
    if F_coef is None:
        F_coef = np.arange(X.shape[0]) / Fs_F

    if 'extent' not in kwargs:
        x_ext1 = (T_coef[1] - T_coef[0]) / 2
        x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
        y_ext1 = (F_coef[1] - F_coef[0]) / 2
        y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
        kwargs['extent'] = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2, F_coef[0] - y_ext1, F_coef[-1] + y_ext2]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray_r'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'

    im = ax[0].imshow(X, **kwargs)

    if len(ax) == 2 and colorbar:
        plt.colorbar(im, cax=ax[1])
    elif len(ax) == 2 and not colorbar:
        ax[1].set_axis_off()
    elif len(ax) == 1 and colorbar:
        plt.sca(ax[0])
        plt.colorbar(im, aspect=colorbar_aspect)

    ax[0].set_xlabel(xlabel, fontsize=14)
    ax[0].set_ylabel(ylabel, fontsize=14)
    ax[0].set_title(title, fontsize=18)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, im

def drawSyllablesBols(filePath, ax, y=None, timeOffset=50e-3, size=10, textColor='blue'):
    data_df = pd.read_csv(filePath)
    if y is None: y = 0
    for i in range(data_df.shape[0]):
        try:
            text = data_df.iloc[i]['Syllable']
        except KeyError:
            text = data_df.iloc[i]['Bol']
        x = data_df.iloc[i]['Time'] + timeOffset
        ax.annotate(text, (x, y), bbox=dict(facecolor='white', edgecolor='grey', linewidth=0.5), c=textColor, fontsize=size)
    return ax

def intensityContour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, timeStep=0.01, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', xticks=False, yticks=False, xlabel=True, ylabel=True, title='Intensity Contour', cAnnot='red', annotLabel=True, annotAlpha=0.8):
    '''Calculates the intensity contour for an audio clip. Used in fig 10.

    Parameters:
        audio (np.array): loaded audio time series
        sr (int): sample rate of audio time series/ to load the audio at
        audioPath (str): path to audio file; only needed if audio is None
        startTime (float): time to start reading audio file
        duration (float): duration of the audio file to read
        minPitch (float): minimum pitch to read for contour extraction
        timeStep (float): time steps in which audio is extracted
        ax (plt.Axes.axis): axis to plot the pitch contour in
        freqXlabels (int): time (in seconds) after which each x label occurs
        annotate (bool): if True, will annotate tala markings
        cyclePath (str): path to file with tala cycle annotations
        numDiv (int): number of divisions to put between each annotation marking
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetTimeKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        xticks (bool): if True, will plot xticklabels
        yticks (bool): if True, will plot yticklabels
        xlabel (bool): if True, will add an x label
        ylabel (bool): if True, will add a y label
        title (str): title of the plot
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        cAnnot (str): color of the annotation
        annotAlpha: controls opacity of the annotation lines

    Returns:
        ax: plot of pitch contour if ax was not None
        (intensity_vals, time_vals): returns intensity values and time values if ax is None
    
    '''
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot
    if audio is None:
        # if audio is not given, load audio from audioPath
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration = duration)
    snd = parselmouth.Sound(audio, sr)
    intensity = snd.to_intensity(time_step=timeStep, minimum_pitch=minPitch)
    intensity_vals = intensity.values[0]
    time_vals = intensity.xs()
    
    if ax is None:
        # if ax is None, return intensity and time values
        return (intensity_vals, time_vals)
    else:
        # else plot the contour
        return plotIntensity(intensity_vals=intensity_vals, time_vals=time_vals, ax=ax, startTime=startTime, duration=duration, freqXlabels=freqXlabels, xticks=xticks, yticks=yticks, xlabel=xlabel, ylabel=ylabel, title=title, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, cAnnot=cAnnot, annotLabel=annotLabel, annotAlpha=annotAlpha)

def plotIntensity(intensity_vals=None, time_vals=None, ax=None, startTime=0, duration=None, freqXlabels=5, xticks=True, yticks=True, xlabel=True, ylabel=True, title='Intensity Contour', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='red', annotLabel=True, annotAlpha=0.8):
    '''Function to plot a computed intensity contour from `intensityContour` function. Used in fig 10.

    Parameters:
        intensity_vals (np.array): intensity value from `intensityContour`
        time_vals (np.array): time steps corresponding to the intensity values from `intensityContour`
        ax (plt.Axes.axis): axis object on which plot is to be plotted
        startTime (float): start time for x labels in the plot
        duration (float): duration of audio in the plot (used for x labels)
        freqXlabels (int): time (in seconds) after which each x label occurs
        xticks (bool): if True, will print x tick labels
        yticks (bool): if True, will print y tick labels
        xlabel (bool): if True, will print the x label
        ylabel (bool): if True, will print the y label
        title (str): The title of the plot
        annotate (bool): if True will mark annotations provided
        cyclePath (str): path to file with cycle annotations; used only if annotate is True
        numDiv (int): number of divisions to add between each marked cycle; used only if annotate is True
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        cAnnot (str): colour to draw annotation lines in; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True
        annotAlpha (float): controls opacity of the line drawn
    Returns
        ax: plotted axis
    
    '''
    if intensity_vals is None or time_vals is None:
        Exception('No intensity contour and/or time values provided')
    if ax is None:
        Exception('ax parameter has to be provided')
    
    ax = sns.lineplot(x=time_vals, y=intensity_vals, ax=ax, color=cAnnot);
    ax.set(xlabel='Time Stamp (s)' if xlabel else '', 
    ylabel='Intensity (dB)' if ylabel else '', 
    title=title, 
    xlim=(0, duration), 
    xticks=(np.arange(0, duration, freqXlabels)), 
    xticklabels=(np.arange(startTime, duration+startTime, freqXlabels) )if xticks else [])
    if not yticks:
        ax.set(yticklabels=[])
    if annotate:
        ax = drawAnnotation(cyclePath=cyclePath, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, numDiv=numDiv, startTime=startTime, duration=duration, ax=ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha)
    return ax

def plot_hand(annotationFile=None, startTime=0, duration=None, freqXLabels=5, vidFps=25, ax=None, annotate=False, cyclePath=None, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', numDiv=0, cAnnot='yellow', annotLabel=False, annotAlpha=0.8, xticks=False, yticks=False, xlabel=True, ylabel=True, title='Wrist Position Vs. Time', vidOffset=0, lWristCol='LWrist', rWristCol='RWrist', wristAxis='y'):
    '''Function to show hand movement. Used in fig 10.
    
    Parameters:
        annotationFile (str): file path to openpose annotations
        startTime (float): start time for x labels in the plot
        duration (float): duration of audio in the plot (used for x labels)
        freqXlabels (int): time (in seconds) after which each x label occurs
        vidFps (float): fps of the video
        ax (plt.Axes.axis): axis object on which plot is to be plotted
        annotate (bool): if True will mark annotations provided
        cyclePath (str): path to file with cycle annotations; used only if annotate is True
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        numDiv (int): number of divisions to add between each marked cycle; used only if annotate is True
        cAnnot (str): colour to draw annotation lines in; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True
        annotAlpha (float): controls opacity of the line drawn
        xticks (bool): if True, will print x tick labels
        yticks (bool): if True, will print y tick labels
        xlabel (bool): if True, will add the x label
        ylabel (bool): if True, will add the y label
        title (str): title of the plot
        videoOffset (float): number of seconds offset between video and audio; time in audio + videioOffset = time in video
        lWristCol (str): name of the column with left wrist data in annotationFile
        rWristCol (str): name of the column with right wrist data in annotationFile
        wristAxis (str): Level 2 header denoting axis along which movement is plotted
        
    Returns
        ax: plotted axis

    '''
    startTime = math.floor(startTime) + np.around(vidOffset)   # set start time to an integer, for better readability on the x axis of the plot
    duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot
    movements = pd.read_csv(annotationFile, header=[0, 1])
    lWrist = movements[lWristCol][wristAxis].values[startTime*vidFps:(startTime+duration)*vidFps]
    rWrist = movements[rWristCol][wristAxis].values[startTime*vidFps:(startTime+duration)*vidFps]
    xvals = np.linspace(0, duration, vidFps*duration, endpoint=False)
    ax.plot(xvals, lWrist, label='Left Wrist')
    ax.plot(xvals, rWrist, label='Right Wrist')
    ax.set(xlabel='Time Stamp (s)' if xlabel else '', 
    ylabel='Wrist Position' if ylabel else '', 
    title=title, 
    xlim=(0, duration), 
    xticks=(np.arange(0, duration, freqXLabels)), 
    xticklabels=(np.arange(startTime, duration+startTime, freqXLabels) )if xticks else [],
    )
    if not yticks:
        ax.set(yticklabels=[])
    ax.invert_yaxis()
    ax.legend()
    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime-np.around(vidOffset), duration, ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha)
    return ax

def annotateInteraction(axs, keywords, cs, interactionFile, startTime, duration):
    '''Adds interaction annotation to the axes given. Used in fig 3.

    Parameters:
        axs: list of axs to add annotation to
        keywords: keyword corresponding to each axis. If len(keywords) = len(axs) + 1, the last keyword is plotted in all axes
        cs: list of colours associated with each keyword
        interactionFile: path to csv file with the annotation of the interactions
        startTime: time to start reading the audio
        duration: length of audio to consider

    Returns
        axs: list of axes with annotation
    '''

    annotations = pd.read_csv(interactionFile, header=None)
    annotations.columns = ['Type', 'Start Time', 'End Time', 'Duration', 'Label']
    annotations = annotations.loc[((annotations['Start Time'] >= startTime) & (annotations['Start Time'] <= startTime+duration)) &
                                ((annotations['End Time'] >= startTime) & (annotations['End Time'] <= startTime+duration))
                                ]
    for i, keyword in enumerate(keywords):
        if i < len(axs):
            # keyword corresponds to a particular axis
            for _, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
                rand = np.random.random()# random vertical displacement for the label
                lims = axs[i].get_ylim()
                axs[i].annotate('', xy=(annotation['Start Time'] - startTime, rand*(lims[1] - lims[0] - 100) + lims[0] + 50), xytext=(annotation['End Time'] - startTime, rand*(lims[1] - lims[0] - 100) + lims[0] + 50), arrowprops={'headlength': 0.4, 'headwidth': 0.2, 'width': 3, 'ec': cs[i], 'fc': cs[i]})
                axs[i].annotate(annotation['Label'], (annotation['Start Time']-startTime+annotation['Duration']/2, rand*(lims[1] - lims[0] - 100) + lims[0] + 150), ha='center')
        else:
            # keyword corresponds to all axes
            for ax in axs:
               for _, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
                rand = np.random.random()# random vertical displacement for the label
                ax.annotate('', xy=(annotation['Start Time'] - startTime, rand*(lims[1] - lims[0] - 100) + lims[0] + 50), xytext=(annotation['End Time'] - startTime, rand*(lims[1] - lims[0] - 100) + lims[0] + 50), arrowprops={'headlength': 0.4, 'headwidth': 0.2, 'width': 3, 'ec':cs[i], 'fc': cs[i]})
                ax.annotate(annotation['Label'], (annotation['Start Time']-startTime+annotation['Duration']/2, rand*(lims[1] - lims[0] - 100) + lims[0] + 150), ha='center') 
    return axs

def drawHandTap(ax, handTaps, c='purple'):
    '''Plots the hand taps as vertical lines on the axis ax. Used in fig 9.
    
    Parameters:
        ax (plt.Axes.axis): axis to add hand taps to
        handTaps (np.array): array of hand tap timestamps
        c (str): color for plot
    '''
    for handTap in handTaps:
        ax.axvline(handTap, linestyle='--', c=c, alpha=0.6)
    return ax

def generateVideoWSquares(vid_path, timeStamps, dest_path='Data/Temp/vidWSquares.mp4', vid_size=(720, 576)):
    '''Function to genrate a video with rectangles for each tap. Used in fig 9.
    
    Parameters:
        vid_path (str): path to the original video
        timeStamps (list): list of time stamps with the following values for each time stamp - [time, keyword, [pos1, pos2], color]
        dest_path (str): file path to save video with clicks
        vid_size ((int, int)): video size to generate

    Returns
        None
    '''

    cap_vid = cv2.VideoCapture(vid_path)
    fps = cap_vid.get(cv2.CAP_PROP_FPS)
    framesToDraw = defaultdict(list)   # dictionary with frame numbers as keys and properties of square box to draw as list of values
    for timeRow in timeStamps:
        framesToDraw[int(np.around(timeRow[0]*fps))] = timeRow[1:]
    output = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*"XVID"), fps, vid_size)
    i = 0
    # generate video
    while(cap_vid.isOpened()):
        ret, frame = cap_vid.read()
        if ret == True:
            i+=1
            if i in framesToDraw.keys():
                frame = cv2.rectangle(frame, framesToDraw[i][1][0], framesToDraw[i][1][1], tuple([int(x) for x in framesToDraw[i][2]][::-1]), 3)    # converting color from BGR to RGB
            output.write(frame)
        else:
            # all frames are read
            break
    cap_vid.release()
    output.release()

def combineAudioVideo(vid_path='Data/Temp/vidWSquares.mp4', audio_path='audioWClicks.wav', dest_path='Data/Temp/FinalVid.mp4'):
    '''Function to combine audio and video into a single file. Used in fig 9.

    Parameters:
        vid_path (str): file path to the video file with squares
        audio_path (str): file path to the audio file with clicks
        dest_path (str): file path to store the combined file at

    Returns
        None

    '''
    
    vid_file = ffmpeg.input(vid_path)
    audio_file = ffmpeg.input(audio_path)
    (
        ffmpeg
        .concat(vid_file.video, audio_file.audio, v=1, a=1)
        .output(dest_path)
        .overwrite_output()
        .run()
    )
    print('Video saved at ' + dest_path)

def generateVideo(annotationFile, onsetKeywords, vidPath='Data/Temp/VS_Shree_1235_1321.mp4', tempFolder='Data/Temp/', pos=None, cs=None):
    '''Function to generate video with squares and clicks corresponding to hand taps. Used in fig 9.
    
    Parameters:
        annotationFile (str): file path to the annotation file with hand tap timestamps
        onsetKeywords (list): list of column names to read from annotationFile
        vidPath (str): file path to original file
        tempFolder (str): file path to temporary directory to store files in
        pos (list): list of [pos1, pos2] -> 2 opposite corners of the box for each keyword 
        cs (list): list of [R, G, B] colours used for each keyword
    
    Returns
        None
    '''
    annotations = pd.read_csv(annotationFile)
    timeStamps = []
    for i, keyword in enumerate(onsetKeywords):
        for timeVal in annotations[keyword].values[~np.isnan(annotations[keyword].values)]:
            timeStamps.append([timeVal, keyword, pos[i], cs[i]])
    timeStamps.sort(key=lambda x: x[0])

    # generate video 
    generateVideoWSquares(vid_path=vidPath, timeStamps=timeStamps, dest_path=os.path.join(tempFolder, 'vidWSquares.mp4'))

    # generate audio
    playAudioWClicks(audioPath=vidPath, onsetFile=annotationFile, onsetLabels=onsetKeywords, destPath=os.path.join(tempFolder, 'audioWClicks.wav'))

    # combine audio and video
    combineAudioVideo(vid_path=os.path.join(tempFolder, 'vidWSquares.mp4'), audio_path=os.path.join(tempFolder, 'audioWClicks.wav'), dest_path=os.path.join(tempFolder, 'finalVid.mp4'))

'''
References

[1] Rao, P., Vinutha, T.P. and Rohit, M.A., 2020. Structural Segmentation of Alap in Dhrupad Vocal Concerts. 
    Transactions of the International Society for Music Information Retrieval, 3(1), pp.137–152. DOI: http://doi.org/10.5334/tismir.64
[2] Meinard Müller and Frank Zalkow: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing. 
    Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
'''
