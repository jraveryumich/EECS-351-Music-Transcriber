%{
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Uses custom polyphonic detection function on two chords. First
             chord is a CEG single octave chord and second is a CEGCEG
             double octave chord showing harmonic detection and removal.

Outputs: Plot of Frequencies before and after pitch detection. The red
         vertical lines are true notes being played.

Known Limitations: This is not a machine learning algorithm, it's all math.
                   So, it can be broken fairly easily with short,
                   impulse-like notes and at high frequencies. However, it
                   is a good example of removing extra harmonics from
                   chords.
%}

clear; clc;

[y1, fs1] = audioread("test.wav");
[y2, fs2] = audioread("C_chord_double.wav");


fmin = 5;
fmax = 10000;

% Notes/Octaves in C chord
C4 = 261.6256;
E4 = 329.6276;
G4 = 391.9954;

C5 = 523.2511;
E5 = 659.2551;
G5 = 783.9909;

fmin = fmin * 2;
fmax = fmax * 2;

% Perform Fast Fourier Transform
L1 = length(y1);
f1 = fs1*(1:(L1/2))/L1;
f1 = f1(fmin:fmax);
Y1 = abs(fft(y1));
half1 = round(length(Y1)/2);
Y1 = Y1(1:half1, 1) / length(Y1);

L2 = length(y2);
f2 = fs2*(1:(L2/2))/L2;
f2 = f2(fmin:fmax);
Y2 = abs(fft(y2));
half2 = round(length(Y2)/2);
Y2 = Y2(1:half2, 1) / length(Y2);

% remove harmonics
newY1 = remove_harmonics(Y1, 0.1 * fs1 / length(Y1) * max(Y1), 0.1, 7);
newY2 = remove_harmonics(Y2, 0.1 * fs2 / length(Y2) * max(Y2), 0.1, 7);

% Generate Plot
figure(1);
tiledlayout(2, 2);
nexttile;
hold on;
semilogx(f1, Y1(fmin:fmax));
xline([C4, E4, G4], '--r');
% xline([C5, E5, G5], '--r');
xlim([10, 2000]);
ylim([0, 0.06]);
title("original - simple chord (CEG)");
hold off;

nexttile;
hold on;
semilogx(f1, newY1(fmin:fmax));
xline([C4, E4, G4], '--r');
% xline([C5, E5, G5], '--r');
xlim([10, 2000]);
ylim([0, 0.06]);
title("no harmonics - simple chord (CEG)");
hold off;

nexttile;
hold on;
semilogx(f2, Y2(fmin:fmax));
xline([C4, E4, G4], '--r');
xline([C5, E5, G5], '--r');
xlim([10, 2000]);
ylim([0, 0.06]);
title("original - stacked chord (CEGCEG)");
hold off;

nexttile;
hold on;
semilogx(f2, newY2(fmin:fmax));
xline([C4, E4, G4], '--r');
xline([C5, E5, G5], '--r');
xlim([10, 2000]);
ylim([0, 0.06]);
title("no harmonics - stacked chord (CEGCEG)");
hold off;
