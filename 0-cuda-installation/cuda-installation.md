# CUDA Installation

[Official Tutorial](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
[Official Tutorial for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## Check Hardware

Run the following CLI to ensure your machine has a CUDA-compatible Nvidia GPU.

```shell
lspci | grep -i nvidia
```

## Install Driver

[Official Tutorial for Linux](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installation)

The following tutorial uses Linux Ubuntu 22.04 LTS as an example.e

### Step 1: Set Environment Variables

Set up `version`, `distro`, `arch`, `arch_ext` based on your machine specifications and the [driver version](https://www.nvidia.com/en-us/drivers/).

```shell
# Get the target driver version based on the GPU version.
export version=550.144.03
export distro=ubuntu2204
export arch=x86_64
export arch_ext=amd64
```

### Step 2: Install Driver

[Official Tutorial for Ubuntu](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu)

## Use AWS EC2

Deep Learning OSS Nvidia Driver AMI only supports the following instances.

```shell
Supported EC2 instances: G4dn, G5, G6, Gr6, G6e, P4, P4de, P5, P5e
```
