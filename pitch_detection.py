"""
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Function that runs a MATLAB script using the "pitch" function.
"""


# Runs Matlab "pitch" function - used on stemss
import matlab.engine
import matplotlib.pyplot as plt


m_eng = matlab.engine.start_matlab()


def MATLAB_Pitches(filename: str, showPlot: bool = False):
    """
    MATLAB script that detects the pitches in an wave file.

    :param filename: path to file
    :param showPlot: plot the pitches
    :return: vector of pitches in Hz, position of pitch in seconds
    """
    pitches, s = m_eng.pitch_detection(filename, nargout=2)

    if showPlot:
        plt.plot(pitches)
        plt.show()
    return pitches, s


if __name__ == '__main__':
    """
    Example use case with the test.wav file    
    """

    MATLAB_Pitches("test.wav", True)
