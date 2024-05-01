"""
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Uses the Basic_Pitch Machine Learning Algorithm to transcribe songs

Outputs: outputs a MIDI file that can be loaded into a music program such as MuseScore

WARNINGS: if you get lots of red output text, as long as no error occurs, this is fine to ignore


https://en.wikipedia.org/wiki/General_MIDI
https://www.ccarh.org/courses/253/handout/gminstruments/
"""

import os
import pretty_midi
from basic_pitch.inference import predict


def Transcribe(FILES: list[str], instrumentTypes: list[int], instrumentThresholds: list[float], saveName: str, Output_Folder='') -> str:
    """
    Header Function to use the 'predict' function from Basic_Pitch machine learning algorithm

    :param FILES: list of audio files to combine into one MIDI file
    :param instrumentTypes: list of MIDI instrument types
    :param instrumentThresholds: list of thresholds for each file for Basic_Pitch
    :param saveName: output file name
    :returns: path to midi file
    """
    if Output_Folder == '':
        Output_Folder = os.path.dirname(os.path.realpath(FILES[0]))

    midis = []
    for i in range(len(FILES)):
        model_output, midi_data, note_events = predict(FILES[i], frame_threshold=instrumentThresholds[i])
        midis.append(midi_data)

    midi = pretty_midi.PrettyMIDI()

    for i in range(len(midis)):
        midis[i].instruments[0].program = instrumentTypes[i]
        midi.instruments.append(midis[i].instruments[0])

    path = os.path.join(Output_Folder, saveName + '.mid')
    midi.write(path)
    return path


if __name__ == '__main__':
    # https://en.wikipedia.org/wiki/General_MIDI

    # Audio files to combine in MIDI file
    files: list[str] = ["chords.wav", "C-Major.wav"]

    # Declare what Instruments you are using
    # 1 is a piano sound
    # other sounds/types can found in the wiki link
    instrument_types: list[int] = []

    # makes all instruments pianos - comment-out this line and set the parameters manually to change
    instrument_types = [1] * len(files)

    # Threshold parameter for Basic_Pitch algorithm
    thresholds: list[float] = []

    # set all to 0.5 for general thresholding
    thresholds = [0.5] * len(files)

    # name of output file in directory
    outputFileName = 'sample_transcription'

    # Transcribe these signals to MIDI file format
    output_path = Transcribe(files, instrument_types, thresholds, outputFileName)

    print("Output File: " + outputFileName)

    # Sample Output
    """
    Predicting MIDI for chords.wav...
    Predicting MIDI for C-Major.wav...
    Output File: sample_transcription
    """
