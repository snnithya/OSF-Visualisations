from IPython.core.display import Video
from librosa.core.audio import get_duration
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import scipy.signal as sig
from librosa.display import waveplot, specshow
from IPython.display import Audio
from IPython.core.display import Video 
import parselmouth
import math
import soundfile as sf
import ffmpeg
sns.set_theme()

def readCycleAnnotation(cyclePath, numDiv, startTime, duration):
    '''Function to read cycle annotation and add divisions in the middle if required.

    Parameters:
        cyclePath: path to the cycle annotation file
        numDiv: number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime: start time of audio being analysed
        duration: duration of the audio to be analysed

    Returns:
        provided: a numpy array of annotations from the file
        computed: a numpy array of division between annotations
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

def drawAnnotation(cyclePath=None, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', numDiv=0, startTime=0, duration=None, ax=None, annotLabel=True, c='purple', alpha=0.8):
    '''Draws annotations on ax

    Parameters
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

    Returns
        ax (plt.Axes.axis): axis that has been plotted in
    '''
    if cyclePath is not None:
        provided, computed = readCycleAnnotation(cyclePath, numDiv, startTime, duration)
        timeCol = ['Time']    # name of column with time readings
        labelCol = ['Cycle']  # name of column to extract label of annotation from
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
        c = c if isinstance(c, list) else [c]
        provided = readOnsetAnnotation(onsetPath, startTime, duration, onsetTimeKeyword)
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
                    ax.annotate(f"{float(providedVal[labelCol[i]]):g}", (providedVal[timeCol[i]]-startTime, (ylims[1]-ylims[0])*0.7 + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c='white')
    ax.legend()
    return ax

def pitchCountour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, maxPitch=660, notes=None, tonic=220, timeStep=0.01, octaveJumpCost=0.9, veryAccurate=True, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', xticks=False, yticks=False, annotLabel=True, cAnnot='purple', ylim=None, annotAlpha=0.8):
    '''Returns pitch contour for the audio

    Uses `plotPitch` to plot pitch contour.

    Parameters
        audio: loaded audio time series
        sr: sample rate of audio time series/ to load the audio at
        audioPath: path to audio file; only needed if audio is None
        startTime: time to start reading audio file
        duration: duration of the audio file to read
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
        onsetPath: path to file with onset annotations; only considered if cyclePath is None
        onsetTimeKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        xticks: if True, will plot xticklabels
        yticks: if True, will plot yticklabels
        annotLabel: if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        cAnnot: color of the annotation
        ylim: (min, max) limits for the y axis; if None, will be directly interpreted from the data
        annotAlpha: controls opacity of the annotation lines

    Returns:
        ax: plot of pitch contour if ax was not None
        
    '''
    
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        # if audio is not given, load audio from audioPath
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot

    snd = parselmouth.Sound(audio, sr)
    pitch = snd.to_pitch_ac(time_step=timeStep, pitch_floor=minPitch, very_accurate=veryAccurate, octave_jump_cost=octaveJumpCost, pitch_ceiling=maxPitch)
    
    # if ax is None, raise error
    if ax is None:
        Exception('ax parameter has to be provided')
    # plot the contour
    return plotPitch(pitch, notes, ax, tonic, startTime, duration, freqXlabels, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, xticks=xticks, yticks=yticks, cAnnot=cAnnot, annotLabel=annotLabel, ylim=ylim, annotAlpha=annotAlpha)

def plotPitch(pitch=None, notes=None, ax=None, tonic=None, startTime=0, duration=None, freqXlabels=5, xticks=True, yticks=True, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='purple', annotLabel=True, ylim=None, annotAlpha=0.8):
    '''Converts the pitch contour from Hz to Cents, and plots it

    Parameters
        pitch: pitch object from `pitchCountour`
        notes: object for each note used for labelling y-axis
        ax: axis object on which plot is to be plotted
        tonic: tonic (in Hz) of audio clip
        startTime: start time for x labels in the plot
        duration: duration of audio in the plot (used for x labels)
        freqXlabels: time (in seconds) after which each x label occurs
        annotate: if true will mark annotations provided
        xticks: if True, will print x tick labels
        yticks: if True, will print y tick labels
        annotate: if True, will add beat annotations to the plot 
        cyclePath: path to file with cycle annotations; used only if annotate is True
        numDiv: number of divisions to add between each marked cycle; used only if annotate is True
        onsetPath: path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        cAnnot: colour to draw annotation lines in; used only if annotate is True
        annotLabel: if True, will print annotation label along with line; used only if annotate is True
        ylim: (min, max) limits for the y axis; if None, will be directly interpreted from the data
        annotAlpha (float): controls opacity of the line drawn
    Returns
        ax: plotted axis
    '''

    # Check that all required parameters are present
    if pitch is None:
        Exception('No pitch contour provided')
    if tonic is None:
        Exception('No tonic provided')
    if ax is None:
        Exception('ax parameter has to be provided')
    yvals = pitch.selected_array['frequency']
    yvals[yvals==0] = np.nan    # mark unvoiced regions as np.nan
    yvals[~(np.isnan(yvals))] = 1200*np.log2(yvals[~(np.isnan(yvals))]/tonic)   #convert Hz to cents
    xvals = pitch.xs()
    # duration = xvals[-1] + 1    # set duration as last x value + 1
    ax = sns.lineplot(x=xvals, y=yvals, ax=ax)
    ax.set(xlabel='Time Stamp (s)' if xticks else '', 
    ylabel='Notes' if yticks else '', 
    title='Pitch Contour (in Cents)', 
    xlim=(0, duration), 
    xticks=(np.arange(0, duration, freqXlabels)), 
    xticklabels=(np.arange(startTime, duration+startTime, freqXlabels) )if xticks else [])
    if notes is not None and yticks:
        # add yticks if needed
        ax.set(
        yticks=[x['cents'] for x in notes if (x['cents'] >= min(yvals[~(np.isnan(yvals))])) & (x['cents'] <= max(yvals[~(np.isnan(yvals))]))] if yticks else [], 
        yticklabels=[x['label'] for x in notes if (x['cents'] >= min(yvals[~(np.isnan(yvals))])) & (x['cents'] <= max(yvals[~(np.isnan(yvals))]))] if yticks else [])
    if ylim is not None:
        ax.set(ylim=ylim)

    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime, duration, ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha)
    return ax

def spectrogram(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, cmap='Blues', ax=None, amin=1e-5, freqXlabels=5, xticks=False, yticks=False, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='purple', annotLabel=True, title='Spectrogram'):
    '''Plots spectrogram

    Parameters
        audio: loaded audio time series
        sr: sample rate that audio time series is loaded/ is to be loaded in
        audioPath: path to the audio file; only needed if audio is None
        startTime: time to start reading the audio at
        duration: duration of audio
        cmap: colormap to use to plot spectrogram
        ax: axis to plot spectrogram in
        amin: controls the contrast of the spectrogram; passed into librosa.power_to_db function
        freqXlabels: time (in seconds) after which each x label occurs
        xticks: if true, will print x labels
        yticks: if true, will print y labels
        annotate: if True, will annotate either tala or onset markings; if both are provided, tala annotations will be marked
        cyclePath: path to file with tala cycle annotations
        numDiv: number of divisions to put between each tala annotation marking
        onsetPath: path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        cAnnot: colour for the annotation marking
        annotLabel: if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
    '''
    if ax is None:
        Exception('ax parameter has to be provided')
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot
    
    # stft params
    winsize = int(np.ceil(sr*40e-3))
    hopsize = int(np.ceil(sr*10e-3))
    nfft = int(2**np.ceil(np.log2(winsize)))

    # STFT
    f,t,X = sig.stft(audio, fs=sr, window='hann', nperseg=winsize, noverlap=(winsize-hopsize), nfft=nfft)
    X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=amin)

    specshow(X_dB, x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopsize, ax=ax, cmap=cmap)
    ax.set(ylabel='Frequency (Hz)' if yticks else '', 
    xlabel='Time (s)' if xticks else '', 
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

def drawWave(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, xticks=False, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, cAnnot='purple', annotLabel=True):
    '''Plots the wave plot of the audio

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
    '''
    if ax is None:
        Exception('ax parameter has to be provided')
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot
    
    
    waveplot(audio, sr, ax=ax)
    ax.set(xlabel='' if not xticks else 'Time (s)', 
    xlim=(0, duration), 
    xticks=[] if not xticks else np.around(np.arange(0, duration, freqXlabels)),
    xticklabels=[] if not xticks else np.around(np.arange(startTime, duration+startTime, freqXlabels), 2),
    title='Waveplot')
    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, numDiv, startTime, duration, ax, c=cAnnot, annotLabel=annotLabel)
    return ax

def playAudio(audio=None, sr=16000, audioPath=None, startTime=0, duration=None):
    '''Plays relevant part of audio

    Parameters
        audio: loaded audio sample
        sr: sample rate of audio
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

    Parameters
        audio (np.array): loaded audio sample
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

    Parameters
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
    if heightRatios is None:
        # if heightRatios is None
        heightRatios = np.ones(noRows)
    fig = plt.figure(figsize=figSize)
    specs = fig.add_gridspec(noRows, 1, height_ratios = heightRatios)
    axs = [fig.add_subplot(specs[i, 0]) for i in range(noRows)]
    return fig, axs