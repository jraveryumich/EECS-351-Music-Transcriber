%{
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Detects pitches from music with mono audio

Requires: Audio Toolbox

Reference:
https://matlab.mathworks.com/

%}

% Filename path relative to project path
function [pitches, s] = pitch_detection(filename)
    [song,fs] = audioread(filename);


    %sound(song,fs)
    
    %Makes the stereo audio mono
    lv = ~sum(song == 0, 2);                       
    mono = sum(song, 2);                             
    mono(lv) = mono(lv)/2;
    
    song = mono;
    
    %power of the song
    %remove the lower power areas to reduce noise
    song_ones = [abs(song) > 0.01];
    song = [song_ones .* song];
    
    %Process the song
    method = "SRH";
    range = [1, 800]; % hertz
    winDur = 0.1; % seconds, increase for lower res
    overlapDur = 0.06; % seconds
    medFiltLength = 10; % frames
    
    winLength = round(winDur*fs);
    overlapLength = round(overlapDur*fs);
    
    [pitches, s] = pitch(song,fs, ...
        Method=method, ...
        Range=range, ...
        WindowLength=winLength, ...
        OverlapLength=overlapLength, ...
        MedianFilterLength=medFiltLength);

    s = s/fs;
    
    % Remove stray notes
    for i = 3:length(pitches) - 2
        left2 = pitches(i-2);
        left1 = pitches(i-1);
        right1 = pitches(i+1);
        right2 = pitches(i+2);
        meanleft = (left2 + left1) / 2;
        meanright = (right2 + right1) / 2;
        if (abs(pitches(i) - meanleft) > 4 && abs(pitches(i) - meanright) > 4)
            pitches(i) = pitches(i-1);
        end
    end
end








