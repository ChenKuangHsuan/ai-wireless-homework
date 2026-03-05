# Exercise 2.15

This repository implements two neural network models (CsiNet and CS-CsiNet) for Channel State Information (CSI) compression and reconstruction in wireless communication systems. Both models support indoor/outdoor wireless environments, adjustable compression rates, and provide complete training & inference pipelines with performance evaluation and visualization.

## Model Overview
##### 1. CsiNet
An end-to-end residual-based autoencoder for CSI compression/reconstruction:
- Encoder: Conv2D → Flatten → Dense (adaptive CSI feature compression)
- Decoder: Dense → Reshape → Stacked Residual Blocks → Conv2D (CSI reconstruction)
- Key advantage: Fully trainable, no pre-defined compression matrix required

##### 2. CS-CsiNet
A compressed sensing enhanced version of CsiNet:
- Encoder: Fixed random projection matrix (non-trainable CS projection)
- Decoder: Residual-based structure (trainable for CSI reconstruction)
- Key advantage: Lightweight encoder (no training needed)

##### Core Features
- Support indoor/outdoor environment switching
- Configurable compression rates (1/4, 1/16, 1/32, 1/64) via `encoded_dim`
- Automatic performance evaluation (NMSE in dB, correlation coefficient)
- Inference time measurement for practical deployment
- CSI reconstruction visualization (original vs reconstructed amplitude)
- Training/validation loss recording (CSV) and TensorBoard support

## File Structure & Description
| File Name               | Function                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| CsiNet_train.py        | CsiNet training pipeline: model building, training, evaluation, saving   |
| CsiNet_onlytest.py     | CsiNet inference only: load pre-trained model, reconstruct CSI, evaluate |
| CS-CsiNet_train.py     | CS-CsiNet training pipeline: train decoder with fixed CS encoder         |
| CS-CsiNet_onlytest.py  | CS-CsiNet inference only: load decoder, reconstruct CSI from compressed data |

## Key Parameters (Modify at script top)
| Parameter       | Default | Description                                                  |
|-----------------|---------|--------------------------------------------------------------|
| envir           | indoor  | Wireless environment (indoor/outdoor)                        |
| img_height/width| 32/32   | CSI matrix spatial/frequency dimensions                      |
| img_channels    | 2       | CSI channels (real + imaginary parts)                        |
| residual_num    | 2       | Number of residual blocks in decoder                         |
| encoded_dim     | 512     | Compression dimension (512=1/4, 128=1/16, 64=1/32, 32=1/64)  |

## Environment Requirements
- Python 2.7/3.x
- TensorFlow 1.x (compatible with 2.x with minor adjustments)
- Keras, NumPy, SciPy, Matplotlib

Install dependencies:
pip install tensorflow keras numpy scipy matplotlib

## Dataset Preparation
Place MATLAB-formatted CSI datasets in a "data/" directory:
- Training/validation/test CSI: DATA_Htrainin.mat, DATA_Hvalin.mat, DATA_Htestin.mat (indoor)
- Frequency-domain CSI for evaluation: DATA_HtestFin_all.mat (indoor)
- Random projection matrices (CS-CsiNet): A512.mat, A128.mat, A64.mat, A32.mat

## Quick Start

##### Step 1: Create Directories
mkdir -p data result saved_model

##### Step 2: Model Training

###### Train CsiNet
python CsiNet_train.py

###### Train CS-CsiNet
python CS-CsiNet_train.py

##### Step 3: Model Inference
Copy pre-trained model files from "result/" to "saved_model/" first
###### Run CsiNet inference
python CsiNet_onlytest.py

###### Run CS-CsiNet inference
python CS-CsiNet_onlytest.py

## Performance Metrics
##### 1. Normalized Mean Square Error (NMSE, dB)
NMSE (dB) = 10*log10(E[|CSI_orig - CSI_rec|²]/E[|CSI_orig|²])
- Smaller (more negative) = better reconstruction

##### 2. Correlation Coefficient (rho)
- Range: [0,1], closer to 1 = higher similarity between original/reconstructed CSI

##### * Important Notes
1. Inference scripts must use the same "envir" and "encoded_dim" as the pre-trained model
2. CS-CsiNet requires matching random projection matrix (A{encoded_dim}.mat)
3. Remove "tf.reset_default_graph()" for TensorFlow 2.x compatibility
4. Code uses "channels_first" data format (do not modify without adjusting network)
5. Default training epochs: 1000 (adjust based on dataset/hardware)

## Output Directories
##### result/ (Training Only)
- trainloss_*.csv: Batch-wise training loss
- valloss_*.csv: Epoch-wise validation loss
- decoded_*.csv: Reconstructed CSI data
- rho_*.csv: Correlation coefficient per test sample
- model_*.json: Model architecture
- model_*.h5: Model weights
- TensorBoard_*/: Training logs

##### saved_model/ (Inference Only)
- model_*.json: Pre-trained model architecture
- model_*.h5: Pre-trained model weights