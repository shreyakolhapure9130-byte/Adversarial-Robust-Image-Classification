# Reproducing the Best Leaderboard Result (0.649957)

This repository contains the code and configuration necessary to recreate our highest-scoring model for the Adversarial ML: Robustness task. 

## 1. Prerequisites
Ensure you have the following dependencies installed in your Python environment:
* `torch`
* `torchvision`
* `numpy`

A CUDA-enabled GPU is strongly recommended to reproduce the training within a reasonable timeframe.

## 2. Data Setup
1. Download the training dataset (`train.npz`).
2. Place `train.npz` in the same directory as the training script. 
3. If running locally or on a cluster, ensure the `DRIVE_DATA_PATH` variable within the script points to the correct location of `train.npz`.

## 3. Training the Model
To train the robust ResNet18 classifier from scratch, execute the training script:

```bash
python task_template.py
