# Music Transcriber

_Updated on 5/1/2024_

### Overview
This project was created as a final project for EECS 351 at the University of Michigan.
The goal is to separate a song into multiple "stems" of vocals, drums, bass, and other. Then,
the system will transcribe the stems into notation using MIDI files to create sheet music.

[Wix Website](https://jravery.wixsite.com/music-transcriber)

It uses [Hybrid Demucs](https://github.com/facebookresearch/demucs), a source separation model based on the U-Net 
convolutional network inspired by Wave U-Net. This project followed and expanded upon 
[this](https://github.com/pytorch/audio/blob/main/examples/tutorials/hybrid_demucs_tutorial.py) PyTorch tuturial.


### Demo Requirements
There are only a few dependencies that need to be installed to run the demo.py script and the various sample scripts
we have included.

To install dependencies: run in terminal ```pip install -r requirements.txt```

Run the demo script in the terminal with ```python demo.py```

### Requirements to run all scripts
To run all the python scripts, please install the following dependencies in order. Note: we ran into many errors while
installing these dependencies the first time around - specifically PyTorch and Hybrid Demucs. If you run into
trouble, the only file that uses these is ```main.py```, so the rest of the sample files should work.

**_Requirements.txt:_**
- ```pip install -r requirements.txt```

**_PyTorch and Hybrid Demucs:_**
- Windows with Cuda: ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

- Windows without Cuda: ```pip install torch torchvision torchaudio```

- Mac: ```pip install torch torchvision torchaudio```

**_Basic Pitch:_**
- ```pip install basic_pitch```

**_MATLAB:_**
- This project includes MATLAB scripts that can be run outside of Python
- The only necessary toolbox is the Audio Toolbox

**_MAC:_**
- Delete ffmpeg.exe and ffprobe.exe files in the zip folder and follow "ffmpeg or ffprobe 
error" in Troubleshooting below

**_Troubleshooting_** \
Various solutions to issues we ran into - mostly Mac OS related
- MATLAB error: make sure you have **MATLAB 2024a** or newer installed
  - try ```pip install matlabengine```
- MAC: Need MATLAB version and python version to match, as in they would both
have to be apple silicon or intel versions.
  - VSCode: Create a new virtual environment venv using the correct python interpreter
- If having troubles with python 3.9 or lower, download python 3.11 to update python
- use ```python --version``` to check version of python
  - if version does not immediately change, check ```python3 --version```
  - then use ```alias python=python3``` to update ```python``` version to match
- ffmpeg or ffprobe error:
  - Download ffmpeg.exe and ffprobe.exe from https://ffbinaries.com/downloads and extract in Downloads
  - Run this is terminal
```
sudo cp Downloads/ffmpeg /usr/local/bin/
sudo chmod 755 /usr/local/bin/ffmpeg
ffmpeg

sudo cp Downloads/ffprobe /usr/local/bin/
sudo chmod 755 /usr/local/bin/ffprobe
ffprobe
```
- URL Error: Macintosh HD > Applications > Python3.11 folder (or whatever version of python you're using) > double click on "Install Certificates.command" file.

### How to Use Song Splitter and Music Transcriber

- Download Song Splitter zip and extract to a project folder
- Run in terminal ```python main.py``` for the culmination of the project
  - Run ```python demo.py``` for a short demo of the pitch detection to MIDI algorithm
- On startup, the program will check package requirements and open a folder if successful.
- Select the song/audio file you wish to use.
- The program will run and provide step-by-step updates in the terminal.
- Once the program is finished, it will open the outputs folder containing the output files:
  - Spectrograms
  - Stem Waveforms
  - MIDI Files

