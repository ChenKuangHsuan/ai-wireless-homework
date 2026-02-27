"""
Exercise 6.6: FedAvg Implementation under Packet Loss

This script simulates federated learning on a non-i.i.d. MNIST dataset
(Dirichlet split, α=0.5). It compares the convergence behavior across
different packet-loss rates (1%, 5%, 10%).

TODO:
1. Complete the `unreliable_channel` function to randomly drop client updates
   based on the specified loss rate.
2. Complete the `client_train` function. You need to implement the standard
   local training loop optimizing the NLLLoss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from collections import defaultdict
import copy


# ────────────────────────  Experiment Configuration  ─────────────────────── #
NUM_CLIENTS   = 100                 # Total simulated clients
COMM_ROUNDS   = 100                 # Global communication rounds
LOCAL_EPOCHS  = 5                   # Local epochs per round
BATCH_SIZE    = 32                  # Mini-batch size for client training
ALPHA         = 0.5                 # Dirichlet concentration (smaller ⇒ more skew)
LOSS_RATES    = [0.01, 0.05, 0.10]  # Packet-loss probabilities to test
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────  Model Definition  ─────────────────────────────── #
class MNISTCNN(nn.Module):
    """ Two-conv CNN for 28×28 MNIST digits (10 classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


# ────────────────────────  Data Partitioning  ────────────────────────────── #
def dirichlet_split(dataset, num_clients, alpha):
    """
    Partition `dataset` into `num_clients` subsets via Dirichlet sampling.

    Args:
        dataset (torch.utils.data.Dataset): Full dataset.
        num_clients (int): Number of client subsets.
        alpha (float): Dirichlet concentration; lower ⇒ more unbalanced.

    Returns:
        list[Subset]: List of dataset subsets for each client.
    """
    labels        = np.array(dataset.targets)
    all_indices   = np.arange(len(dataset))
    class_indices = [all_indices[labels == c] for c in range(10)]

    client_indices = [[] for _ in range(num_clients)]
    for c_idx in class_indices:
        np.random.shuffle(c_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        for cid, idx_split in enumerate(np.split(c_idx, splits)):
            client_indices[cid].extend(idx_split)

    return [Subset(dataset, idxs) for idxs in client_indices]


# ────────────────────────  Channel Simulation (TODO)  ────────────────────── #
def unreliable_channel(updates, loss_rate):
    """
    Simulate an unreliable network by randomly discarding client updates.

    TODO:
        • For each element in `updates`, drop it with probability `loss_rate`.
        • You can EITHER
            – return a list where dropped items are replaced by `None`, OR
            – filter out the dropped items entirely.
        • Make sure to document the choice you make.

    Args:
        updates   (list[dict]): State-dicts produced by selected clients.
        loss_rate (float)     : Packet-loss probability in the range [0, 1].

    Returns:
        list[dict | None]: List of updates after applying packet loss.
    """
    # ─── YOUR CODE HERE ──────────────────────────────────────────────────── #
    # Remove "pass" and "raise NotImplementedError" statements once implemented.
    pass
    # ─────────────────────────────────────────────────────────────────────── #

    raise NotImplementedError


# ────────────────────────  Local Training (TODO)  ────────────────────────── #
def client_train(model, loader):
    """
    Perform LOCAL_EPOCHS of SGD on a single client.

    TODO:
        • Forward pass to obtain log-probabilities.
        • Compute negative-log-likelihood (NLL) loss.
        • Back-propagate and update model parameters.
        • Return the trained model’s state-dict.

    Args:
        model  (nn.Module)  : Client model initialised with global weights.
        loader (DataLoader) : Client-side data loader.

    Returns:
        dict: Updated model weights after local training.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # ─── YOUR CODE HERE ──────────────────────────────────────────── #
            # 1. Zero existing gradients
            # 2. Forward pass
            # 3. Compute NLL loss
            # 4. Backward + parameter update
            # Remove "pass" and "raise NotImplementedError" statements once implemented.
            pass
            # ─────────────────────────────────────────────────────────────── #

    raise NotImplementedError


# ────────────────────────  Evaluation  ───────────────────────────────────── #
def evaluate(model, test_loader):
    """Compute test accuracy (%) and NLL loss."""
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss_sum += F.nll_loss(out, y, reduction="sum").item()
            correct  += out.argmax(dim=1).eq(y).sum().item()
    n = len(test_loader.dataset)
    return 100.0 * correct / n, loss_sum / n


# ────────────────────────  FedAvg Experiment  ────────────────────────────── #
def run_fedavg_experiment(loss_rate):
    """
    Execute FedAvg with a specified packet-loss probability.

    Args:
        loss_rate (float): Packet-loss probability to emulate.

    Returns:
        tuple[list, list]: Accuracy history and loss history.
    """
    # Data preparation
    transform    = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data   = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_data    = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader  = DataLoader(test_data, batch_size=1000)

    client_sets  = dirichlet_split(train_data, NUM_CLIENTS, ALPHA)
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_sets]

    global_model = MNISTCNN().to(DEVICE)
    acc_hist, loss_hist = [], []

    for rnd in range(COMM_ROUNDS):
        # Sample 10 % clients
        selected = np.random.choice(NUM_CLIENTS, size=NUM_CLIENTS // 10, replace=False)

        # Local training
        client_updates = []
        for cid in selected:
            local_model = copy.deepcopy(global_model)
            update = client_train(local_model, client_loaders[cid])
            client_updates.append(update)

        # Packet loss
        valid_updates = [u for u in unreliable_channel(client_updates, loss_rate) if u]

        # FedAvg aggregation
        if valid_updates:
            new_state = defaultdict(torch.Tensor)
            for k in valid_updates[0]:
                new_state[k] = torch.stack([u[k] for u in valid_updates]).mean(0)
            global_model.load_state_dict(new_state)

        # Evaluation
        acc, loss = evaluate(global_model, test_loader)
        acc_hist.append(acc)
        loss_hist.append(loss)

        print(f"Round {rnd + 1:02d}, Loss Rate {loss_rate:.0%}: "
              f"Acc {acc:.2f} %, Loss {loss:.4f}, "
              f"Received {len(valid_updates)}/{len(selected)} updates")

    return acc_hist, loss_hist


# ──────────  ──────────────  Main Execution  ───────────────────────────── #
if __name__ == "__main__":
    results = {}
    for rate in LOSS_RATES:
        print(f"\n=== Experiment with Packet-Loss Rate {rate:.0%} ===")
        results[rate] = run_fedavg_experiment(rate)

    # Visualization
    plt.figure(figsize=(12, 5))

    # Accuracy curves
    plt.subplot(1, 2, 1)
    for rate, (acc, _) in results.items():
        plt.plot(acc, label=f"Loss Rate {rate:.0%}")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.title("FedAvg Accuracy under Packet Loss")
    plt.legend()
    plt.grid(True)

    # Loss curves
    plt.subplot(1, 2, 2)
    for rate, (_, loss) in results.items():
        plt.plot(loss, label=f"Loss Rate {rate:.0%}")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test NLL Loss")
    plt.title("FedAvg Loss under Packet Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("packet_loss_impact.png", dpi=300)
    plt.show()