"""
DUPLICATE OF demo.py - REFERENCED AS AN IMPORT IN main.py

Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Allows testing of the polyphonic algorithm with any song. Creates
             a MIDI file which can be open in a music program, such as MuseScore

Outputs: list of detected notes, note positions, note durations.
         Creates a MIDI File with name: 'polyphonic_test_output.mid'

Known Limitations: This is not a machine learning algorithm, it's all math. So, it can be broken fairly easily with
                   short, impulse-like notes and at high frequencies. However, it is a good example of removing
                   extra harmonics from chords.

How to Use: Select a song to perform the transcription. Any of the provided example .wav files can be used!

https://en.wikipedia.org/wiki/General_MIDI
https://www.ccarh.org/courses/253/handout/gminstruments/

Python External Dependencies:
- numpy
- scipy
- librosa
- pretty_midi
"""

import sys, os
import librosa
import numpy as np
import tkinter
from tkinter import filedialog

import pretty_midi

from Sample_Polyphonic_Pitch_Detection import polyphonic_pitch


def Transcribe(FILES: list[str], instrumentTypes: list[int] = None, saveName: str = '', verbose=False, Output_Folder='') -> str:
    """
    Transcribes a list of audio signals to MIDI using the custom polyphonic pitch detection function

    :param FILES: list of audio files to combine into one MIDI file
    :param instrumentTypes: list of MIDI instrument types
    :param saveName: output file name
    :param verbose: print verbose output
    :param Output_Folder: The folder to output the MIDI file
    :returns: path to midi file named 'score.mid' by default
    """

    # get the folder containing the files assuming they all originate from the same folder
    if Output_Folder == '':
        Output_Folder = os.path.dirname(os.path.realpath(FILES[0]))

    # create a MIDI object
    midi: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()

    # set all instruments to piano if not specified
    if not instrumentTypes:
        instrumentTypes = [1] * len(FILES)

    # generate a track for each file given
    for i in range(len(FILES)):
        # handy function to get name of file from path
        instrument_name = "".join(os.path.basename(FILES[i]).split(".")[:-1])

        isDrum = False
        if instrument_name == "drums":
            isDrum = True

        # generate the MIDI for this file
        midi_instrument = GenerateInstrument(FILES[i], instrumentTypes[i], instrument_name, isDrum, verbose)
        midi.instruments.append(midi_instrument)

    # apply tempo
    midi.get_tempo_changes()

    # write the MIDI object
    path = os.path.join(Output_Folder, 'score.mid' if saveName == '' else saveName + ".mid")
    midi.write(path)
    return path


def GenerateInstrument(filepath: str, instrument_type=1, instrument_name='piano', is_drum=False, verbose=False) -> pretty_midi.Instrument:
    """
    Use the custom polyphonic pitch detection algorithm to create a MIDI instrument

    :param filepath: path to audio signal .wav
    :param instrument_type: MIDI instrument type
    :param instrument_name: name of instrument
    :param is_drum: is this instrument drums?
    :param verbose: Print verbose output
    :return: MIDI
    """
    y, sr = librosa.load(filepath, sr=44100, mono=True)

    if verbose: print("loaded: " + filepath)

    # calculate the onsets from the sampled data
    #   - this is the list points at which a new note starts
    if verbose: print("calculating onset")
    onset = librosa.onset.onset_detect(y=y, sr=sr, units='samples')
    onset = np.multiply((onset > 1), onset)

    # append the ending to the onset frames
    note_ranges = np.append(onset, len(y))

    # the first onset is always 1536 samples behind
    note_ranges[0] -= 1536

    pitches = []
    note_lengths = []

    # runs polyphonic pitch detection on each note by windowing the note using the onsets
    #   - gives a status bar as this can take a while for longer songs
    if verbose: print("\nperforming polyphonic pitch detection on: " + filepath)
    for i in range(len(note_ranges) - 1):
        # create a window
        center = ((note_ranges[i + 1] - note_ranges[i]) / 2 + note_ranges[i])
        windlength = 0.8 * (note_ranges[i + 1] - note_ranges[i])

        note_lengths.append((note_ranges[i + 1] - note_ranges[i]))

        start = center - windlength / 2
        end = center + windlength / 2

        start = round(start)
        end = round(end)

        x = y[start:end]

        # zero pad x
        zero_length = int(sr / 4)
        if len(x) < zero_length:
            x = librosa.util.pad_center(x, size=zero_length, mode='constant')
        elif len(x) > sr:  # hard cap on length of a note to 1 second
            x = librosa.util.fix_length(x, size=int(sr))

        # run the algorithm
        [m_pitches, ft] = polyphonic_pitch(x, sr)

        pitches.append(m_pitches.tolist())

        # status bar
        print_percent_done(i, len(note_ranges) - 1)

    print("\n")  # fixes formatting after status bar

    # Create a MIDI instrument
    instrument = pretty_midi.Instrument(instrument_type, is_drum, instrument_name)

    # convert the frequencies to musical notes
    notes_list = []
    for notes in pitches:
        if notes:  # as long as it's not empty
            notes_list.append(librosa.hz_to_note(notes).tolist())

    # time that note start (in seconds)
    note_times = [round(x / sr * 8) / 8 for x in note_ranges]       # 8 means resolution of 16th notes
    # time duration of note (in seconds)
    note_durations = [round(x / sr * 8) / 8 for x in note_lengths]  # 8 means resolution of 16th notes

    # remove duplicate notes
    uniq_notes_list = []
    for notes in notes_list:
        newlist = []
        for note in notes:  # ignore highlight
            if note not in newlist:
                newlist.append(note)
        uniq_notes_list.append(newlist)

    if verbose: print(uniq_notes_list)
    if verbose: print(note_times)
    if verbose: print(note_durations)

    # add all the notes to the MIDI data
    for i in range(len(uniq_notes_list)):
        for n in uniq_notes_list[i]:
            pitch = round(librosa.note_to_midi(n))
            if pitch < 0 or pitch > 127: continue
            time = note_times[i]
            duration = note_durations[i]

            note = pretty_midi.Note(100, pitch, time, time + duration)
            instrument.notes.append(note)

    return instrument


def print_percent_done(index, total, bar_len=50, title=''):
    """
    Function that gives updates during long songs
    """
    percent_done = (index+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    sys.stdout.write(f'\r\t⏳ {title}: [{done_str}{togo_str}] {percent_done}% done')
    sys.stdout.flush()


if __name__ == '__main__':
    """
    Uses the polyphonic pitch function to loosely detect pitches on waveforms.
    Creates a MIDI file that can be loaded into a music program, such as Musescore, and read as sheet music.
    
    Name of MIDI file: "polyphonic_test_output.mid"
    """

    # file stuff
    print("Window Open")
    tkinter.Tk().withdraw()
    filepath = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select song to transcribe",
        filetypes=(('', '*.m4a'), ('', '*.wav'), ('', '*.mp3'))  # specify others as needed
    )
    if str(filepath) == '':
        print("\nNo File Selected")
        quit()

    path = Transcribe([filepath], [1], "polyphonic_test_output", False)
    print("Output file: " + path)
