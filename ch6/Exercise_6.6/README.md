# Exercise 6.6 (FedAvg Implementation under Packet Loss)

This repository contains an implementation of the FedAvg (Federated Averaging) algorithm with simulated packet loss, designed to analyze how unreliable communication affects federated learning performance on the MNIST dataset.

## Usage

### Run the script:

```bash
python exercise_6.6.py
```

## Overview

The experiment includes:
- A CNN model architecture for MNIST classification
- Non-IID data partitioning using Dirichlet distribution
- Simulation of varying packet loss rates
- Federated training with partial client participation
- Visualization of accuracy and loss over communication rounds

## Configuration Parameters

The main configuration parameters are:

```python
NUM_CLIENTS = 100            # Total number of clients
COMM_ROUNDS = 100            # Number of global communication rounds
LOCAL_EPOCHS = 5             # Local training epochs per client
BATCH_SIZE = 32              # Batch size for training
ALPHA = 0.5                  # Dirichlet concentration parameter for non-IID partitioning
LOSS_RATES = [0.01, 0.05, 0.10]  # Packet loss probabilities to simulate
```

## Key Components

### Model Architecture

A convolutional neural network with:
- Two convolutional layers (32 and 64 filters)
- Two fully connected layers (512 neurons and 10 output classes)
- ReLU activation and max pooling
- LogSoftmax output for NLL loss

### Data Partitioning

The implementation uses a Dirichlet distribution (parameterized by α) to create non-IID data partitions among clients. Lower α values create more heterogeneous distributions.

### FedAvg with Packet Loss

In each communication round:
- 10% of clients are randomly selected for training
- Local model updates are subject to simulated packet loss
- The global model aggregates only the successfully received updates

### Evaluation

The global model is evaluated after every round using the full MNIST test set. Accuracy and loss are recorded to analyze training convergence under different loss conditions.

## Results

The implementation generates a plot showing test accuracy and loss across communication rounds for different packet loss rates. This helps visualize how communication reliability impacts learning performance.

The results are saved as:

```
packet_loss_impact.png
```
