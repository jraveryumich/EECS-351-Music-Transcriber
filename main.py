"""
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

Referenced this tutorial: https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html

*********************************************************************************
To Use:
- Run Script
- Select a song on your device from the window popup
    - supports most formats ex: mp3, m4a, wav (will error if format is wrong)
        - Can add more formats as needed
    - "cancel" will stop the program
    - "select" starts the process
- Wait for program to finish
    - Progress updates are printed in terminal
- On finish, the output folder will open with the stems and spectrogram plots
*********************************************************************************
"""

import os.path
import urllib.error

import tkinter
from tkinter import filedialog
import platform

import librosa.onset
import numpy
from librosa import beat
from librosa import onset
import torch, torchaudio
import torchaudio.prototype.transforms

import matplotlib.pyplot as plt

from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS  # demucs model
from torchaudio.transforms import Fade

from pydub import AudioSegment  # converts formats

import time

from transcribe import transcribeWithMuseScore

OUTPUT_FOLDER = 'output_files'
PRODUCE_SPECTROGRAM = False


def construct_pipeline():
    """
        Uses the torchaudio.models.HDemucs model trained on MUSDB18-HQ and additional internal extra training data.
        This specific model is suited for higher sample rates, around 44.1 kHz and has an nfft value of 4096 with
        a depth of 6 in the model implementation.
    """
    delay = 5
    max_retries = 3
    for _ in range(max_retries):
        try:
            bundle = HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            _sample_rate = bundle.sample_rate
            print(f"\tSample Rate: {_sample_rate}")
            break
        except urllib.error.URLError:
            time.sleep(delay)
            delay *= 2
    else:
        raise Exception(f"Connection failed after {max_retries} attempts")

    return device, model


def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
    """
        Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

        Args:
            segment (int): segment length in seconds
            device (torch.device, str, or None): if provided, device on which to
                execute the computation, otherwise `mix.device` is assumed.
                When `device` is different from `mix.device`, only local computations will
                be on `device`, while the entire tracks will be stored on `mix.device`.
            model:
            mix:
            overlap:
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)
    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    print("\tSuccessfully Separated Sources")
    return final


def plot_spectrogram(stft, title="Spectrogram"):
    """
    Plots a spectrogram of the fourier transform.
    """
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    fig, axis = plt.subplots(1, 1)
    axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, title.lower() + '.png'))
    plt.close(fig)


def output_results(predicted_source: torch.Tensor, source: str, predicted_source_spec: torch.Tensor = None):
    """
    Outputs the results of the model as the 4 stems and 4 spectrograms.
    :param predicted_source: entire stem
    :param predicted_source_spec: stem spec
    :param source: name of stem (string)
    :return: none
    """

    torchaudio.save(os.path.join(STEMS_FOLDER, source + '.wav'), predicted_source, sample_rate=sample_rate,
                    bits_per_sample=16, encoding='PCM_S')

    if PRODUCE_SPECTROGRAM and (predicted_source_spec is not None):
        plot_spectrogram(stft(predicted_source_spec)[0], f"Spectrogram - {source}")


if __name__ == '__main__':
    startTime = time.time()

    # Necessities
    print("PyTorch Info")
    print(f"\tVersion: {torch.__version__}")
    print(f"\tAudio Version: {torchaudio.__version__}")
    print(f"\tCuda Available: {str(torch.cuda.is_available())}")
    if torch.cuda.is_available():
        print(f"\tUsing GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # File Stuff
    print("Window Open")
    tkinter.Tk().withdraw()
    filepath = filedialog.askopenfilename(
        initialdir='Music',
        title="Select Song To Split",
        filetypes=(('', '*.m4a'), ('', '*.wav'), ('', '*.mp3'))  # specify others as needed
    )

    if str(filepath) == '':
        print("\nNo File Selected")
        quit()

    SONG_NAME, SONG_FORMAT = os.path.splitext(filepath)
    SONG_NAME = str(SONG_NAME).split('/')[-1]

    OUTPUT_SONG_FOLDER = os.path.join(OUTPUT_FOLDER, SONG_NAME)
    PLOTS_FOLDER = os.path.join(OUTPUT_SONG_FOLDER, 'plots')
    STEMS_FOLDER = os.path.join(OUTPUT_SONG_FOLDER, 'stems')
    if not os.path.exists(STEMS_FOLDER):
        os.makedirs(STEMS_FOLDER)
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)

    # Convert audio to wav if needed
    song = AudioSegment.from_file(filepath, format=str(SONG_FORMAT).split(".")[1])

    SONG_FILE = os.path.join(STEMS_FOLDER, 'original.wav')
    song.export(SONG_FILE, format='wav')

    print("\nConstructing Pipeline")
    device, model = construct_pipeline()

    # Run the model and store the separate source files in a directory
    print("\n------------------------------------\n")
    print(f"Performing Separation Using DEMUCS On: {SONG_NAME}")

    waveform, sample_rate = torchaudio.load(SONG_FILE)
    waveform = waveform.to(device)
    mixture = waveform

    # parameters
    segment: int = 10
    overlap = 0.1

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    # demucs is run here
    sources = separate_sources(model, waveform[None], device=device, segment=segment, overlap=overlap)[0]
    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))

    drums = audios["drums"].cpu()
    bass = audios["bass"].cpu()
    vocals = audios["vocals"].cpu()
    other = audios["other"].cpu()

    # Separate Track
    """
    The default set of pretrained weights that has been loaded has 4 sources
    that it is separated into: drums, bass, other, and vocals in that order. They
    have been stored into the dict "audios" and therefore can be accessed there. For 
    the four sources, there is a separate cell for each that will create the audio, the
    spectrogram graph, and calculate the SDR score. SDR is the signal-to-distortion ratio,
    essentially a representation to the "quality" of an audio track.
    """

    N_FFT = 4096
    N_HOP = 4
    stft = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=N_HOP, power=None)

    # Audio Segmenting and Processing
    """
    Below is the processing steps and segmenting a few seconds of the tracks to feed into the
    spectrogram.
    """
    print("\tSegmenting Audio")

    # segment_start = 34.398
    # segment_end = 45.224
    segment_start = 0
    segment_end = 3

    frame_start = int(segment_start * sample_rate)
    frame_end = int(segment_end * sample_rate)

    drums_spec = audios["drums"][:, frame_start:frame_end].cpu()
    bass_spec = audios["bass"][:, frame_start:frame_end].cpu()
    vocals_spec = audios["vocals"][:, frame_start:frame_end].cpu()
    other_spec = audios["other"][:, frame_start:frame_end].cpu()
    mix_spec = mixture[:, frame_start:frame_end].cpu()
    # mix_spec = mixture.cpu()

    # Spectrograms and Audio
    # print("\tCreating Spectrograms")

    # Mixture Clip
    # plot_spectrogram(stft(mix_spec)[0], "Spectrogram - Mixture")

    print("\n------------------------------------\n")
    print("Creating Output Files")
    print("\tDrums")
    output_results(drums, "drums", drums_spec)
    print("\tBass")
    output_results(bass, "bass", bass_spec)
    print("\tVocals")
    output_results(vocals, "vocals", vocals_spec)
    print("\tOther")
    output_results(other, "other", other_spec)

    print("\nGenerating Transcription")
    transcribeWithMuseScore(STEMS_FOLDER, "vocals.wav")
    transcribeWithMuseScore(STEMS_FOLDER, "bass.wav")
    transcribeWithMuseScore(STEMS_FOLDER, "drums.wav")
    transcribeWithMuseScore(STEMS_FOLDER, "other.wav")

    # Pitch Detection
    # print("\nPitch Detection")
    #
    # print("\tVocal Pitches")
    # vocals_pitches, vocals_s = MATLAB_Pitches(f"{STEMS_FOLDER}/vocals.wav")
    #
    # print("\tBass Pitches")
    # bass_pitches, bass_s = MATLAB_Pitches(f"{STEMS_FOLDER}/bass.wav")
    #
    # print("\tDrums Pitches")
    # drums_pitches, drums_s = MATLAB_Pitches(f"{STEMS_FOLDER}/drums.wav")
    #
    # print("\tOther Pitches")
    # other_pitches, other_s = MATLAB_Pitches(f"{STEMS_FOLDER}/other.wav")

    # Plot all pitches together on subplots
    # print("\nGenerating Pitch Plots")
    # fig, axs = plt.subplots(4, 1)
    # axs[0].plot(vocals_s, vocals_pitches)
    # axs[0].set_title("Vocals", fontsize=20)
    # axs[1].plot(bass_s, bass_pitches)
    # axs[1].set_title("Bass", fontsize=20)
    # axs[2].plot(drums_s, drums_pitches)
    # axs[2].set_title("Drums", fontsize=20)
    # axs[3].plot(other_s, other_pitches)
    # axs[3].set_title("Other", fontsize=20)

    # plt.tight_layout()
    # plt.savefig(os.path.join(PLOTS_FOLDER, 'pitches.png'))
    # plt.close(fig)

    """
    Uncomment the code below to output a plot with the y-axis labeled with notes
    """
    # plot_pitches(other_s, other_pitches)

    # Tempo and BPM
    # print("\nCalculating BPM")
    # tempo = calculate_tempo(SONG_FILE)
    # print("Tempo: " + str(tempo))

    # Music21
    # print("\nTranscribing Stems")
    # stem = "bass"
    # os.path.join(STEMS_FOLDER, stem + '.wav')

    path = os.path.join(os.path.abspath(os.getcwd()), OUTPUT_SONG_FOLDER)

    # Opens output folder - different on different OS's
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macos
        import subprocess
        subprocess.Popen(["open", path])

    print(f"\nComplete in {round(time.time() - startTime, 2)} s")
    print("Output Files Here:")
    print(f'\t{path}')
