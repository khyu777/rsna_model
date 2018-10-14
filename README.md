# Mask RCNN for RSNA Pneumonia Challenge
This is code to train and use Mask RCNN for the RSNA Penumonia. Follow the steps below to use it.

## How to train a model
These steps use Docker.

1. Install [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)
1. Download Kaggle data. Use the [official client](https://github.com/Kaggle/kaggle-api).
1. Download pretrained models from the matterport's [github release page](https://github.com/matterport/Mask_RCNN/releases). Coco weights are [here](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5).
1. Create a config file. You can see a sample in `sample_config.ini`.
1. Run `python train.py` to start training.

## TODO
- Eval script.
