# DPT-season-depth

## 1. DPT
This repository is a refinement of the DPT network on the Season dataset.
```
Vision Transformers for Dense Prediction
René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
```
For the official repository of DPT, see [code](https://github.com/isl-org/DPT).
## 2. Season Depth Challenge
It's an open-source monocular challenge that runs through February-May 2022. Please refer to the [website](http://seasondepth-challenge.org/index/index.html#introduction) for more detailed information.

# Requirements
The code was tested with Python 3.7, PyTorch 1.8.0, OpenCV 4.5.1, and timm 0.4.5

# For Train
1. We added other datasets to the official DPT [pre-training weights](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt) to continue training, and finally trained 60 epochs in the Season Dataset.

# For Test on Season Dataset
DPT weight on Season Dataset in [here]().

# Licences:
DPT(MIT License):  https://github.com/isl-org/DPT#license
