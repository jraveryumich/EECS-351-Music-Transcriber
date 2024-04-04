# Music Transcriber

### Overview
Uses Hybrid [Demucs](https://github.com/facebookresearch/demucs), a source separation model based on the U-Net convolutional network inspired by Wave U-Net. This project followed and expanded upon [this](https://github.com/pytorch/audio/blob/main/examples/tutorials/hybrid_demucs_tutorial.py) PyTorch tuturial.


###Requirements
_**PyTorch:**_
- Windows with Cuda: ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

- Windows without Cuda: ```pip3 install torch torchvision torchaudio```

- Mac: ```pip3 install torch torchvision torchaudio```


### How to Use Song Splitter

- Download Song Splitter zip and extract to a project folder
- Double Click "Splitter.bat" to run. 
  - To create desktop shortcut: Right Click "Splitter.bat" and Send To -> Desktop (Create Shortcut)
- On startup, the program will check package requirements and open a folder if successful.
- Select the song/audio file you wish to use.
- The program will run and provide step-by-step updates in the terminal.
- Once the program is finished it will open the outputs folder containing the spectrogram plots and stems created.

### TODO:
