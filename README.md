# Overview
This repository contains code to train a network that takes in a stream of speech audio and outputs predicted face and head movements in the form of [ARKit blendshapes](https://arkit-face-blendshapes.com/).

You can find an interactive demo [here](https://huggingface.co/spaces/cherrvak/voice2motion-demo), and pretrained checkpoints [here](https://huggingface.co/collections/cherrvak/voice2motion-67a264c301694197145921cd).

# Setup

This codebase requires a CUDA-enabled GPU.

1. Create a Python virtual env (I used Python 3.10.15)

2. [Install torch, torchaudio and torchvision](https://pytorch.org/get-started/locally/) (I used torch==2.5.1, torchaudio=2.5.1, torchvision=0.20.1)

3. [Install pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). I recommend using prebuilt wheels, like so:
```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html
```
but replacing the url with your CUDA version. You may have to install `iopath` first.

4. Install the rest of the dependencies: `pip install -r requirements.txt`

5. Run `python -m src.scripts.download_assets`

6. If you now run `python -m src.scripts.test_setup`, you should see a video of pretrained checkpoint results at `render_output/test_setup.mp4`.

# Dataset

## Videos
You can use any set of videos where each video contains a single speaker with a visible face.

I intended to train on [AVSpeech](https://looking-to-listen.github.io/avspeech/download.html), but got IP banned from Youtube halfway through acquiring the dataset. If you want to replicate my training setup, untar `assets/avspeech_jank_mix.csv.tar.gz`. Otherwise, download the test and train csvs from the AVSpeech download page. Once you've acquired the CSVs, run `python -m src.scripts.download_avspeech --csv your/csv/path --outdir your/video/dir`. Try not to get IP banned.

I also incorporated the "train" split of TalkingHead-1KH into my train set and used the "val" split for validation. If also want to download these videos, check out [this repository](https://github.com/ksadov/TalkingHead-1KH).

## Facial landmarks
Once you have a directory of videos, run `python -m src.scripts.preprocess_dataset --video_dir your/video/path --output_dir your/data/dir` to extract blendshape and head angle data.

If you want to check your dataset for clips that might trip up the dataloader, run `python -m src.scripts.find_bad_clips --data_dir your/data/dir`

# Training
Edit the contents of `configs/example.json` to fit your use case and run `python -m src.scripts.train --config configs/example.json`.

# ONNX export
`python -m src.scripts.export_onnx --checkpoint_path your/checkpoint/path`. Note that you'll also want an ONNX version of the Hubert audio feature extractor, which you can download [here](https://huggingface.co/Xenova/hubert-base-ls960/blame/6d40b9586c1c5106a6f7da2d6e465175afeb0fbb/onnx/model.onnx).

# Inference
`python -m src.scripts.infer --checkpoint_path your/checkpoint/path --file_path your/audio.wav` will simulate running Pytorch inference on streamed audio. To use ONNX for inference, use `python -m src.scripts.infer --checkpoint_path your/checkpoint/path --file_path your/audio.wav`.

# Known issues
- Segmentation faults will randomly occur during training and halt the run. I think it has something to do with saving videos, and the probability of a failure decreases if you keep `num_video_examples` in the training config low and perform fewer vals per epoch.

# Credits
The mesh in `reference_mesh` was modified from the mesh provided in https://github.com/mynalabsai/blendshapes-visualization.
