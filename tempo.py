"""
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Function that calculates the tempo (bpm) of a song
"""

import librosa
from librosa import onset, beat

import matplotlib.pyplot as plt

import scipy


def calculate_tempo(file: str):
    y, sr = librosa.load(file, duration=30)
    onset_env = onset.onset_strength(y=y, sr=sr)
    tempo: float = beat.tempo(onset_envelope=onset_env, sr=sr)

    # N = 10
    # Fc = 500
    # # provide them to firwin
    # h = scipy.signal.firwin(numtaps=N, cutoff=Fc, nyq=sr / 2)
    # # 'x' is the time-series data you are filtering
    # y = scipy.signal.lfilter(h, 1.0, onset_env)
    #
    # # y = scipy.signal.medfilt(onset_env, 7)
    #
    # plt.plot(y)
    # plt.show()

    return tempo


if __name__ == '__main__':
    """
    Example use case with the Piano_C_Major.wav file
    """

    SONG_FILE: str = "Piano_C_Major.wav"
    # SONG_FILE: str = "output_files/dmitri/stems/original.wav"

    print("Calculating Tempo of: " + SONG_FILE)

    tempo = calculate_tempo(SONG_FILE)
    print("Tempo: " + str(tempo))
