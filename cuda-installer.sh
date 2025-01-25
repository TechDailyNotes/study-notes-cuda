#!/bin/bash

sudo apt update && sudo apt upgrade -y && sudo apt autoremove
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run
nvcc --version
# Check whether Nvidia driver is installed.
# `SMI` refers to System Management Interface which monitors GPUs.
nvidia-smi
