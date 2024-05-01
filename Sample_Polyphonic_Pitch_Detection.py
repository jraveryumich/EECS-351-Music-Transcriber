"""
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Performs Polyphonic Pitch Detection using a custom pitch detection algorithm given as the two functions
             below 'remove_harmonics' and 'polyphonic_pitch' on a sample audio file. The default audio file is the
             included 'C_chord.wav' file which is a CEG chord held as a whole note.

Outputs: plot of final fourier transform of kept frequencies and prints these frequencies and their corresponding
         musical notes.

Known Limitations: This is not a machine learning algorithm, it's all math. So, it can be broken fairly easily with
                   short, impulse-like notes and at high frequencies. However, it is a good example of removing
                   extra harmonics from chords.

Python External Dependencies:
- numpy
- scipy
- librosa
- matplotlib
"""

import numpy as np
from scipy.signal import find_peaks
import librosa

"""
Polyphonic Pitch Detection Algorithm

Initially developed in MATLAB, but translated to Python for integration
"""


def remove_harmonics(Y: np.ndarray, cutoff=0.05, mul2_cutoff=0.1, sum_cutoff=7) -> np.ndarray:
    """
    Removes additional harmonics from a fourier transform of pitches. This algorithm works great on simple whole-note
    chords, but can be broken down with shorter note chords or multiple high frequency notes (above the 5th octave)

    :param Y: magnitude of fft of audio file
    :param cutoff: remove frequencies below this magnitude threshold
    :param mul2_cutoff: variance in harmonic frequencies - default of 0.1 should be fine
    :param sum_cutoff: variance in added harmonic frequencies - default of 7 should be fine
    :return: array containing only the "real" notes from the signal
    """

    # contains the local peaks (with the correct peak values)
    peaks = find_peaks(Y, height=cutoff)[0]
    for i in range(len(Y)):
        if i not in peaks:
            Y[i] = 0  # set all non-peaks to 0
    maxY = Y

    # list of indices to delete after loop
    indicesToDelete = np.ones_like(maxY)

    length = len(maxY)

    # Look through all of the frequency peaks and determine which ones to keep
    for i in range(1, length):
        if maxY[i] != 0:
            for j in range(1, length):
                if maxY[j] != 0 and j != i:
                    # remove multiple of 2 (octave harmonic)
                    isMul2 = abs(2 - (j / i)) < mul2_cutoff
                    if isMul2 and maxY[j] < maxY[i] * 2.5:  # * 2.5 to remove more harmonics, / 2 to detect more notes
                        indicesToDelete[j] = 0

                    # check for sum of previous notes
                    # AND you need to check for the sum of the previous notes with each harmonic
                    # so don't delete the harmonic before you check
                    for k in range(1, length):
                        if maxY[k] != 0 and k != i and k != j:
                            # remove the sum of two frequencies that would produce a fake note
                            isSum = abs(k - (i + j)) < sum_cutoff
                            if isSum and maxY[k] < maxY[i] and maxY[k] < maxY[j]:
                                indicesToDelete[k] = 0

    maxY = np.multiply(maxY, indicesToDelete)
    return maxY


def polyphonic_pitch(y: np.ndarray, fs) -> (np.ndarray, np.ndarray):
    """
    Header function for the 'remove_harmonics' function. Takes the fft of a signal and performs polyphonic pitch
    detection on that signal

    :param y: sampled signal
    :param fs: sampling frequency y was sampled at
    :return: tuple of (signal pitches, and the fourier transform data)
    """

    # compute fft
    Y = abs(np.fft.fft(y))
    half = round(len(Y) / 2.0)
    Y = Y[0:half] / len(Y)  # only need half the fft signal

    # remove harmonics
    newY = remove_harmonics(Y, fs * 0.1 / len(Y) * max(Y))

    # set output variables
    ft = newY
    pitches: np.ndarray = np.nonzero(newY)[0]
    pitches = np.multiply(pitches, fs / len(y))  # scale pitches based on length of signal and fs

    return pitches, ft


if __name__ == '__main__':
    """
    Uses the above polyphonic pitch function to loosely detect pitches on waveforms.
    """

    # Audio File to Run pitch detection on
    SONG_NAME = 'C_chord.wav'

    # sample the file
    y, fs = librosa.load(SONG_NAME, sr=44100)

    # perform polyphonic pitch detection
    pitches, ft = polyphonic_pitch(y, fs)

    # Plots the fourier transform of the output
    import matplotlib.pyplot as plt
    plt.plot(ft)
    plt.xscale('log')
    plt.show()

    # Print the pitches corresponding the notes
    print("Pitches (Hz)")
    print("\t" + str(pitches))

    # Prints the notes these frequencies correspond to
    print("\nNotes")
    print("\t" + str(librosa.hz_to_note(pitches)))

    # Sample Output
    """
    Pitches (Hz)
    [261.  329.  391.5]

    Notes
    ['C4' 'E4' 'G4']
    """
