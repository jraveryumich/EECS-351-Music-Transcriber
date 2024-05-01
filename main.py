"""
Created for EECS 351 Final Project - Music Transcriber

Authors: Jacob Avery, Ethan Regan, Jae Un Pae

High-Level Description: Compiles all findings from this project into one 'main' script. It takes a song or audio signal
                        runs the Hybrid Demucs machine learning algorithm on the signal, splitting the song into
                        'vocals', 'drums', 'bass', and 'other' tracks. Spectrograms of each of these splits are created
                        and the splits are converted to .wav files and outputted to the 'output_files' folder. Then,
                        pitch detection using a custom polyphonic pitch detection algorithm is run on the separated
                        tracks and this information is stored in a MIDI file in the output folder. Finally, once the
                        program completes, the generated MIDI file can be opened in a music notation
                        software - we recommend MuseScore.

Referenced this tutorial: https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html

*********************************************************************************
To Use:
- Run Script
- Select a song on your device from the window popup
    - We have provided sample songs to run, but feel free to run any song through
        the algorithm
    - supports most formats ex: mp3, m4a, wav (will error if format is wrong)
        - Can add more formats as needed
    - "cancel" will stop the program
    - "select" starts the process
- Wait for program to finish
    - Progress updates are printed in terminal
- On finish, the output folder will open with the generated output files
*********************************************************************************

* Reference the files titled "Sample_..." for more in depth summaries of functionally
    and samples of how each aspect of the project runs
"""

import os.path
import urllib.error

import tkinter
from tkinter import filedialog
import platform
import torch, torchaudio
import torchaudio.prototype.transforms

import matplotlib.pyplot as plt

from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS  # hybrid demucs model
from torchaudio.transforms import Fade

from pydub import AudioSegment  # converts formats

import time


# General Variables
OUTPUT_FOLDER = 'output_files'
PRODUCE_SPECTROGRAM = True
TRANSCIBE_USING_POLYPHONIC_ALGORITHM = True  # set to False to use Basic_Pitch ML Algorithm


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

    path = os.path.join(STEMS_FOLDER, source + '.wav')
    torchaudio.save(path, predicted_source, sample_rate=sample_rate,
                    bits_per_sample=16, encoding='PCM_S')

    if PRODUCE_SPECTROGRAM and (predicted_source_spec is not None):
        plot_spectrogram(stft(predicted_source_spec)[0], f"Spectrogram - {source}")

    return path


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
    TRANSCRIPTIONS_FOLDER = os.path.join(OUTPUT_SONG_FOLDER, 'transcriptions')
    if not os.path.exists(STEMS_FOLDER):
        os.makedirs(STEMS_FOLDER)
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)
    if not os.path.exists(TRANSCRIPTIONS_FOLDER):
        os.makedirs(TRANSCRIPTIONS_FOLDER)

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
    N_FFT = 4096
    N_HOP = 4
    stft = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=N_HOP, power=None)

    # Audio Segmenting and Processing
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
    print("\tCreating Spectrograms")
    # Mixture Clip
    plot_spectrogram(stft(mix_spec)[0], "Spectrogram - Mixture")

    print("\n------------------------------------\n")
    print("Creating Output Files")
    print("\tDrums")
    drums_path = output_results(drums, "drums", drums_spec)
    print("\tBass")
    bass_path = output_results(bass, "bass", bass_spec)
    print("\tVocals")
    vocals_path = output_results(vocals, "vocals", vocals_spec)
    print("\tOther")
    other_path = output_results(other, "other", other_spec)

    # Pitch Detection
    print("\nPitch Detection")

    # Transcribe using custom algorithm or an ML - both have benefits and drawbacks
    if TRANSCIBE_USING_POLYPHONIC_ALGORITHM:
        from Sample_Polyphonic_Transcription import Transcribe
        print("\tTranscribing Using Polyphonic Pitch Detection\n")

        print("\tVocals")
        Transcribe([vocals_path], [22], "vocals", False, TRANSCRIPTIONS_FOLDER)
        print("\tOther")
        Transcribe([other_path], [0], "other", False, TRANSCRIPTIONS_FOLDER)
        print("\tBass")
        Transcribe([bass_path], [33], "bass", False, TRANSCRIPTIONS_FOLDER)
        print("\tAll Parts")
        Transcribe([vocals_path, other_path, bass_path], [22, 0, 33], "score", False, TRANSCRIPTIONS_FOLDER)
    else:
        from Sample_Basic_Pitch import Transcribe
        print("\tTranscribing Using Basic Pitch\n")

        print("\tVocals")
        Transcribe([vocals_path], [22], [0.4], "vocals", TRANSCRIPTIONS_FOLDER)
        print("\tOther")
        Transcribe([other_path], [0], [0.5], "other", TRANSCRIPTIONS_FOLDER)
        print("\tBass")
        Transcribe([bass_path], [33], [0.3], "bass", TRANSCRIPTIONS_FOLDER)
        print("\tAll Parts")
        Transcribe([vocals_path, other_path, bass_path], [22, 0, 33], [0.4, 0.5, 0.3], "score", TRANSCRIPTIONS_FOLDER)

    path = os.path.join(os.path.abspath(os.getcwd()), OUTPUT_SONG_FOLDER)

    # Opens output folder - different on different OS's
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # MAC OS
        import subprocess
        subprocess.Popen(["open", path])

    print(f"\nComplete in {round(time.time() - startTime, 2)} s")
    print("Output Files Here:")
    print(f'\t{path}')
