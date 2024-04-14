# Music Transcriber

### Overview
Uses [Hybrid Demucs](https://github.com/facebookresearch/demucs), a source separation model based on the U-Net 
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

**_TroubleShooting_**
- If the ```librosa``` package does not download correctly, it is not completely necessary
for the program to run. It is used for plotting notes on the y-axis and is already
commented-out.
- MATLAB error: make sure you have **MATLAB 2024a** or newer installed
  - try ```pip install matlabengine```


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

