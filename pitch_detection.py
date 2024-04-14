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


def plot_pitches(s, pitches):
    """
    Generates a pitch plot with notes on the y-axis.
    :param s: seconds output from MATLAB_Pitches
    :param pitches: pitches output from MATLAB_Pitches
    """
    import librosa

    """
    Change the "range(4,5)" to include the range of octaves you wish to plot
        (for example, C4 to C5 is default) 
    """

    note_freqs = []
    note_names = []

    for i in range(4, 5):
        note_freqs.extend(
            librosa.note_to_hz([
                f'C{i}',
                # f'C#{i}',
                f'D{i}',
                # f'D#{i}',
                f'E{i}',
                f'F{i}',
                # f'F#{i}',
                f'G{i}',
                # f'G#{i}',
                f'A{i}',
                # f'A#{i}',
                f'B{i}',
                f'C{i+1}'
            ])
        )

        note_names.extend([
            f'C{i}',
            # f'C#{i}',
            f'D{i}',
            # f'D#{i}',
            f'E{i}',
            f'F{i}',
            # f'F#{i}',
            f'G{i}',
            # f'G#{i}',
            f'A{i}',
            # f'A#{i}',
            f'B{i}',
            f'C{i+1}'
        ])
    plt.plot(s, pitches)
    plt.yticks(note_freqs, note_names, fontsize=40, rotation=90)
    plt.xticks(fontsize=40)
    plt.title("Other Pitches", fontsize=60)
    plt.xlabel("Time (s)", fontsize=60)
    plt.show()


if __name__ == '__main__':
    """
    Example use case with the Piano_C_Major.wav file    
    """

    pitches, s = MATLAB_Pitches("Piano_C_Major.wav", False)

    plot_pitches(s, pitches)
