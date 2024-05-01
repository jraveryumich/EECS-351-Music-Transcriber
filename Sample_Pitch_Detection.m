%{
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Performs signal-note pitch detection on a song. The default
             audio file includes is a C Major Scale.

Outputs: Plot of the pitches vs. time

Known Limitations: This does not correctly detect more than one note being
                   played at the same time as this is very difficult to
                   implement.

Requires: Audio Toolbox
%}

SONG_NAME = 'C-Major.wav';
[pitches, s] = pitch_detection(SONG_NAME);

figure(1);
plot(s, pitches)
title("Pitches");
ylabel("Frequencies");
xlabel("Time (s)");