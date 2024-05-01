%{
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Description: Removes harmonics from a chord or multiple notes played at
             same time. If harmonics have greater power than the original,
             then it leaves it - probably a real note being played.

Outputs: FFT of note frequencies without harmonics.

Known Limitations: This is not a machine learning algorithm, it's all math.
                   So, it can be broken fairly easily with short,
                   impulse-like notes and at high frequencies. However, it
                   is a good example of removing extra harmonics from
                   chords.
%}

function y = remove_harmonics(Y, cutoff, mul2_cutoff, sum_cutoff)
    % Suggested Values
    % cutoff = 0.05
    % mul2_cutoff = 0.1
    % sum_cutoff = 7

    % contains the local peaks (with the correct peak values)
    maxY = islocalmax(Y) .* Y;
    maxY = (maxY > cutoff) .* maxY;

    % because harmonics have to be SO difficult
    indicesToDelete = ones(length(maxY), 1);

    for i = 1:length(maxY)
        if (maxY(i) ~= 0)
            for j = 1:length(maxY)
                if (maxY(j) ~= 0 && j ~= i)
                    % check for multiple of 2
                    isMul2 = abs(2 - (j / i)) < mul2_cutoff;
                    if (isMul2 && maxY(j) < maxY(i))
                        % maxY(j) = 0;
                        indicesToDelete(j) = 0;
                    % else
                    %     % check for sum of previous notes
                    %     % adding to the harmonic
                    %     for r = 1:length(maxY)
                    %         if (maxY(r) ~= 0 && r ~= i && r ~= j)
                    %             isSum = abs(j - (i + r)) < sum_cutoff;
                    %             if (isSum && maxY(j) < maxY(r) * 1.8)
                    %                 indicesToDelete(j) = 0;
                    %             end
                    %         end
                    %     end
                    end

                    % check for sum of previous notes
                    % (yes this is a thing)

                    % AND you need to check for the sum of the previous
                    % notes with each harmonic
                    % so...don't delete the harmonic before you check
                    % (yes this is also a thing)
                    for k = 1:length(maxY)
                        if (maxY(k) ~= 0 && k ~= i && k ~= j)
                            isSum = abs(k - (i + j)) < sum_cutoff;
                            if (isSum && maxY(k) < maxY(i) && maxY(k) < maxY(j))
                                % maxY(k) = 0;
                                indicesToDelete(k) = 0;
                            end
                        end
                    end
                end
            end
        end
    end

    maxY = maxY .* indicesToDelete;
    y = maxY(:, 1);
end
