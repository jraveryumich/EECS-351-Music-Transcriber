# Music Transcriber

_Updated on 4/14/2024_

### Overview
This project was created as a final project for EECS 351 at the University of Michigan.
The goal is to separate a song into multiple "stems" of vocals, drums, bass, and other. Then,
the system will transcribe the stems into notation with the ultimate goal of creating sheet music.

It uses [Hybrid Demucs](https://github.com/facebookresearch/demucs), a source separation model based on the U-Net 
convolutional network inspired by Wave U-Net. This project followed and expanded upon 
[this](https://github.com/pytorch/audio/blob/main/examples/tutorials/hybrid_demucs_tutorial.py) PyTorch tuturial.


### Requirements

**_PyTorch:_**
- Windows with Cuda: ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

- Windows without Cuda: ```pip install torch torchvision torchaudio```

- Mac: ```pip install torch torchvision torchaudio```

**_Requirements.txt:_**
- ```pip install -r requirements.txt```

**_MATLAB:_**
- This project runs MATLAB scripts from python
- To do this, it may be required to download the latest version of MATLAB
  - As of Winter 2024, MATLAB 2024a is required.

**_MAC:_**
- Delete ffmpeg.exe and ffprobe.exe files and follow "ffmpeg or ffprobe 
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
- If on Windows, Double Click "Transcriber.bat" to run
- If on MAC, run in terminal: ```python main.py``` 
  - To create desktop shortcut: Right Click "Transcriber.bat" and Send To -> Desktop (Create Shortcut)
- Windows defender may flag the Transcriber.bat as a "harmful" file to run. If this is the case, select "more info" and run away.
- On startup, the program will check package requirements and open a folder if successful.
- Select the song/audio file you wish to use.
- The program will run and provide step-by-step updates in the terminal.
- Once the program is finished, it will open the outputs folder containing the spectrogram plots and stems created.

