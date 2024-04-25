"""
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Uses Music21 to transcribe songs
"""
import os
from basic_pitch.inference import predict

import music21
import tkinter
from tkinter import filedialog
from pydub import AudioSegment  # converts formats


# old - don't use
# def __transcribe(STEM_FILE: str):  #, pitches, s):
#     img = Image.open("Empty_Score.png")
#     plt.imshow(img)
#
#     def freq_to_note(freq):
#         notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
#
#         note_number = 12 * math.log2(freq / 440) + 49
#         note_number = round(note_number)
#
#         note = (note_number - 1) % len(notes)
#         note = notes[note]
#
#         octave = (note_number + 8) // len(notes)
#
#         return note, octave
#
#     # 12 between spaces
#     # 6 between steps
#     note_positions = {
#         ("C", 5): 153.5,
#         ("B", 4): 159.5,
#         ("A", 4): 165.5,
#         ("G", 4): 171.5,
#         ("F", 4): 177.5,
#         ("E", 4): 183.5,
#         ("D", 4): 189.5,
#         ("C", 4): 195.5,
#     }
#
#     f, sr = librosa.load(STEM_FILE, sr=44100)
#     onset = librosa.onset.onset_strength(y=f, sr=sr)
#     onset_frames = librosa.onset.onset_detect(onset_envelope=onset, sr=sr)
#     song_time = len(f) / sr
#
#     # make sure you get the note after it starts playing
#     for i in range(len(onset_frames)):
#         onset_frames[i] += 2
#
#     f0 = librosa.yin(y=f, fmin=50, fmax=4000, sr=sr)
#     f0_sr = len(f0) / song_time
#
#     note_durations = []
#
#     # see how long each note is held
#     # note is held until the pitch changes
#     for start in onset_frames:
#         start_note = freq_to_note(f0[start])
#         end = -1
#         for i in range(start, len(f0)):
#             if freq_to_note(f0[i]) != start_note:
#                 end = i
#                 break
#
#         note_duration = (end - start) / f0_sr
#         note_durations.append(note_duration)
#
#     bpm = 120
#     bps = bpm / 60
#     note_lengths = []  # 1 = quarter note, 2 = half note, etc.
#     for i in range(len(note_durations) - 1):  # ignore last value
#         note_lengths.append(1 / math.floor(bps / note_durations[i]))
#
#     # note_locations = []
#     # for i in range(len(note_lengths)):
#     #     note_locations.append(note_lengths[i])
#     #
#     # sum = note_locations[0]
#     # for i in range(1, len(note_locations)):
#     #     sum += note_locations[i]
#     #     note_locations[i] = sum
#
#     # print(note_locations)
#     #
#     # for i in range(len(note_locations)):
#     #     note_locations[i] -= note_lengths[i]
#
#     print(note_lengths)
#     # print(note_locations)
#
#     x_dot, y_dot = [], []
#     x_dash, y_dash = [], []
#     quarter_note_length = 360
#     first_bar_offset = 220
#     for i in range(len(onset_frames) - 1):  # for all notes
#         x_dot.append(i * 80 + first_bar_offset)
#         t = onset_frames[i]
#         note, octave = freq_to_note(f0[t])
#         print(note, octave)
#         y_dot.append(note_positions.get((note, octave)))
#     for i in range(len(onset_frames) - 2):
#         # determine if there is a rest here or not
#         time_to_next_note = onset_frames[i+1] - onset_frames[i]
#         rest_length = 1 / math.floor((time_to_next_note - note_lengths[i]) / 10)
#         if rest_length > 0:
#             x_dash.append((i+1) * 80 + first_bar_offset - rest_length * quarter_note_length)
#             y_dash.append(note_positions.get(("A", 4)))
#             print("rest for: " + str(rest_length))
#
#     size = 0.3
#     plt.scatter(x_dot, y_dot, s=size, color='black')
#     plt.plot(x_dash, y_dash,
#              np.zeros_like(x_dash), np.zeros_like(y_dash),
#              color='black')
#
#     SONG_PATH, SONG_FORMAT = os.path.splitext(STEM_FILE)
#     plt.savefig(SONG_PATH + ".png", dpi=500)


# old - don't use
# def __generate_MIDI(STEM_FILE: str):
#     y, sr = librosa.load(STEM_FILE, sr=44100)
#     onset = librosa.onset.onset_strength(y=y, sr=sr)
#     onset_frames = librosa.onset.onset_detect(onset_envelope=onset, sr=sr)
#     song_time = len(y) / sr
#
#     # make sure you get the note after it starts playing
#     for i in range(len(onset_frames)):
#         onset_frames[i] += 2
#
#     f0 = librosa.yin(y=y, fmin=50, fmax=4000, sr=sr)
#     f0_sr = len(f0) / song_time
#
#     note_names = []
#     bpm = 120
#     bps = bpm / 60
#     note_durations = []  # 0.25 = quarter note, 0.5 = half note, etc.
#     note_times = []  # beat when note starts
#
#     # see how long each note is held
#     # note is held until the pitch changes
#     for start in onset_frames:
#         start_note = librosa.hz_to_note(f0[start])
#         end = -1
#         for i in range(start, len(f0)):
#             if librosa.hz_to_note(f0[i]) != start_note:
#                 end = i
#                 break
#
#         if end != -1:
#             length = (end - start) / f0_sr
#             note_type = math.floor(bps / length)
#             if note_type <= 64:  # avoid notes less than 64th
#                 note_durations.append(1 / note_type)
#                 note_names.append(start_note)
#                 note_times.append(round((start - 2) / 44.1))  # sampling frequency / 1000
#
#     print(note_durations)
#     print(note_names)
#     print(note_times)
#
#     # create MIDI object
#     mf = MIDIFile(1)  # only 1 track
#     track = 0  # the only track
#
#     time = 0  # start at beginning
#     mf.addTrackName(track, time, "Sample Track")
#     mf.addTempo(track, time, 120)
#
#     # add some notes
#     channel = 0
#     volume = 100
#
#     for i in range(len(note_names)):
#         pitch = librosa.note_to_midi(note_names[i])
#         time = note_times[i]
#         duration = note_durations[i]
#         mf.addNote(track, channel, pitch, time, duration, volume)
#
#     pitch = 1
#     time = 0
#     duration = 1
#     mf.addNote(track, channel, pitch, time, duration, volume)
#
#     pitch = 64  # E4
#     time = 2
#     duration = 1
#     mf.addNote(track, channel, pitch, time, duration, volume)
#
#     pitch = 67  # G4
#     time = 4
#     duration = 1
#     mf.addNote(track, channel, pitch, time, duration, volume)
#
#     # write it to disk
#     with open("output.mid", 'wb') as outf:
#         mf.writeFile(outf)
#
#     music21.environment.set('musescoreDirectPNGPath', 'C:\\Program Files\\Musescore 4\\bin\\Musescore4.exe')
#     parsed = music21.converter.parse("output.mid")
#     parsed.show('musicxml.png')


def transcribeWithMuseScore(FILE: str):
    FOLDER = os.path.dirname(os.path.realpath(FILE))
    NAME = os.path.basename(FILE)
    model_output, midi_data, note_events = predict(FILE)

    SONG_NAME = NAME.split('.')[:-1]
    SONG_NAME = ''.join(SONG_NAME)
    midi_data.write(os.path.join(FOLDER, SONG_NAME + '.mid'))

    # music21.environment.set('musescoreDirectPNGPath', 'C:\\Program Files\\Musescore 4\\bin\\Musescore4.exe')
    # parsed = music21.converter.parse(SONG_NAME + '.mid')
    # parsed.show('musicxml.png')


if __name__ == '__main__':
    """
    Example use case with the Piano_C_Major.wav file
    """

    # from pitch_detection import MATLAB_Pitches
    # pitches, s = MATLAB_Pitches("Piano_C_Major.wav")
    # pitches, s = MATLAB_Pitches("C-Major_Up.wav")
    # pitches, s = MATLAB_Pitches("output_files/sax-alto-a-major-scale/stems/original.wav")

    # SONG_FILE: str = "Piano_C_Major.wav"
    # SONG_FILE: str = "C-Major_Up.wav"
    # SONG_FILE: str = 'chords.wav'

    # File Stuff
    print("Window Open")
    tkinter.Tk().withdraw()
    filepath = filedialog.askopenfilename(
        initialdir='Music',
        title="Select song to transcribe",
        filetypes=(('', '*.m4a'), ('', '*.wav'), ('', '*.mp3'))  # specify others as needed
    )

    if str(filepath) == '':
        print("\nNo File Selected")
        quit()

    transcribeWithMuseScore(filepath)
