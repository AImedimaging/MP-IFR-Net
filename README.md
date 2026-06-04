# IFR-Net

Implicit Frequency Representation Network for Sparse-View CT Reconstruction

## Overview

IFR-Net is an implicit neural representation framework that directly learns the complex-valued Fourier spectrum of CT images.

Instead of reconstructing images from sparse-view projections in the image domain, IFR-Net models the mapping

(kx, ky) → F(kx, ky)

where F(kx, ky) denotes the complex Fourier coefficient at a given frequency coordinate.

The learned frequency spectrum is then transformed back into the image domain through inverse Fourier transform.

## Features

* Implicit frequency-domain representation
* Complex-valued neural fitting
* Fourier feature encoding
* Sparse-view CT reconstruction
* Frequency-aware loss function
* GPU acceleration with PyTorch

## Network Architecture

Frequency Coordinate
↓
Fourier Encoding
↓
MLP Backbone
↓
Complex-valued Output
↓
Inverse FFT
↓
CT Reconstruction

## Installation

### Clone Repository

git clone https://github.com/AImedimaging/MP-IFR-Net.git

cd IFR-Net

### Create Environment

conda create -n ifrnet python=3.10

conda activate ifrnet

### Install Dependencies

pip install -r requirements.txt

## Dataset

The training dataset is stored in HDF5 format.

Required fields:

tri_inputs
tri_truths
val_inputs
val_truths

## Training

python scripts/train.py

## Reconstruction

python scripts/recon.py

## Results

The network learns a continuous representation of the Fourier spectrum and reconstructs CT images through inverse Fourier transformation.

