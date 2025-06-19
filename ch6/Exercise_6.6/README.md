# Exercise 6.6: FedAvg Implementation under Packet Loss

This repository provides an implementation of the **FedAvg** (Federated Averaging) algorithm with simulated packet loss to analyze how unreliable communication affects federated learning performance on the MNIST dataset.

## Table of Contents

- [Overview](#overview)
- [Configuration Parameters](#configuration-parameters)
- [Key Components](#key-components)
  - [Model Architecture](#model-architecture)
  - [Data Partitioning](#data-partitioning)
  - [FedAvg with Packet Loss](#fedavg-with-packet-loss)
- [Running the Experiments](#running-the-experiments)
- [Results](#results)

## Overview

This implementation includes the following key features:

- A CNN model for MNIST classification.
- Non-IID data partitioning using the Dirichlet distribution to simulate real-world federated learning scenarios.
- Simulation of varying packet loss rates to analyze the impact of unreliable communication on federated learning.
- Federated training with partial client participation, where a random subset of clients is selected in each communication round.
- Visualization of training accuracy and loss over communication rounds.

## Configuration Parameters

The following configuration parameters are used for training:

```python
NUM_CLIENTS = 100                # Total number of clients
COMM_ROUNDS = 100                # Number of communication rounds
LOCAL_EPOCHS = 5                 # Number of local training epochs per client
BATCH_SIZE = 32                  # Batch size for training
ALPHA = 0.5                      # Dirichlet concentration parameter for non-IID partitioning
LOSS_RATES = [0.01, 0.05, 0.10]  # Packet loss probabilities to simulate
```

## Key Components
### Model Architecture
The model used for MNIST classification is a CNN with the following architecture:

- Two convolutional layers with 32 and 64 filters, respectively.
- Two fully connected layers with 512 neurons and 10 output classes (for the 10 MNIST digits).
- ReLU activation functions and max-pooling layers.
- LogSoftmax output for negative log likelihood loss.

### Data Partitioning
To simulate non-IID data distributions, the implementation uses a Dirichlet distribution with a concentration parameter (α). This creates heterogeneous data splits among the clients. Lower values of α result in more heterogeneous (non-IID) distributions.

### FedAvg with Packet Loss
In each communication round:
- 10% of clients are randomly selected to participate in the round.
- Local model updates are subject to simulated packet loss based on predefined probabilities.
- The global model aggregates only the successfully received updates from the clients.

### Evaluation
The global model is evaluated after every communication round using the full MNIST test set. The test accuracy and loss are tracked and visualized over the communication rounds to analyze the convergence of the model under varying packet loss conditions.

## Running the Experiments
To run the experiments, follow these steps:
1. Initialize a global model: A global model is created at the start of the experiment.
2. Partition the MNIST data: The MNIST dataset is partitioned among clients in a non-IID manner using the Dirichlet distribution.
3. Run federated learning: Federated learning is executed for each packet loss rate.
4. Track and visualize accuracy and loss: The model's test accuracy and loss are tracked and visualized over communication rounds.

To execute the code, run the following command:

```python
python exercise_6.6.py
```

## Results
After running the experiments, the implementation generates a plot that shows test accuracy and loss over communication rounds for different packet loss rates. This visualization helps assess the impact of communication reliability on the convergence and performance of the model.

- Loss rates = 0.01, 0.05, 0.10: Simulating different packet loss probabilities.

The results are saved as an image file named 'packet_loss_impact.png' in the current directory.





