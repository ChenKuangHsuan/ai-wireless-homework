"""
Exercise 4.15: Task-Oriented Semantic Communication for Image Classification

This script adapts the DeepJSCC system for an image classification task over 
an AWGN channel using the CIFAR10 dataset.

TODO:
1. Design the network to output classification logits alongside the reconstructed image.
2. Design the loss function for the classification task.
3. Calculate classification accuracy.
4. Compare performance (PSNR and Accuracy) with the model from Exercise 4.14.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Parameters
BATCH_SIZE = 128
EPOCHS = 5  
LEARNING_RATE = 1e-3
TRAIN_SNRS = [1, 7, 20]  # dB
TEST_SNRS = list(range(0, 22, 2))  # [0, 2, 4, ..., 20] dB
NUM_CLASSES = 10 # CIFAR10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, out_channels=16):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
    def forward(self, x):
        return self.net(x)


class AWGNChannel(nn.Module):
    def __init__(self):
        super(AWGNChannel, self).__init__()

    def forward(self, x, snr_db):
        batch_size = x.shape[0]
        power = torch.mean(x**2, dim=[1, 2, 3], keepdim=True)
        x_norm = x / torch.sqrt(power + 1e-8)
        
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = math.sqrt(1.0 / snr_linear)
        noise = torch.randn_like(x_norm) * noise_std
        return x_norm + noise


class TaskOrientedDecoder(nn.Module):
    def __init__(self, in_channels=16, num_classes=10):
        super(TaskOrientedDecoder, self).__init__()
        
        # ─── YOUR CODE HERE ──────────────────────────────────────────── #
        # 1. Design the reconstruction branch (similar to Ex 4.14)
        self.reconstruction_net = nn.Sequential(
            # Add Transposed Convolutions here
        )
        
        # 2. Design the classification branch (e.g., CNN to Fully Connected)
        # Input to this branch is the received noisy channel symbols (in_channels, 8, 8)
        self.classification_net = nn.Sequential(
            # Add layers to output `num_classes` logits
        )
        pass
        # ─────────────────────────────────────────────────────────────── #

    def forward(self, x):
        # ─── YOUR CODE HERE ──────────────────────────────────────────── #
        # Return both the reconstructed image and the classification logits
        reconstructed_img = x # Replace with your reconstruction output
        logits = x # Replace with your classification output
        pass
        # ─────────────────────────────────────────────────────────────── #
        return reconstructed_img, logits


class DeepJSCC_Classification(nn.Module):
    def __init__(self, channel_dim=16, num_classes=10):
        super(DeepJSCC_Classification, self).__init__()
        self.encoder = Encoder(out_channels=channel_dim)
        self.channel = AWGNChannel()
        self.decoder = TaskOrientedDecoder(in_channels=channel_dim, num_classes=num_classes)
        
    def forward(self, x, snr_db):
        encoded = self.encoder(x)
        received = self.channel(encoded, snr_db)
        reconstructed, logits = self.decoder(received)
        return reconstructed, logits


def calculate_psnr(mse):
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)


def train_model(train_loader, snr_db):
    print(f"\n--- Training Task-Oriented Network at SNR = {snr_db} dB ---")
    model = DeepJSCC_Classification(channel_dim=16, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ─── YOUR CODE HERE ──────────────────────────────────────────── #
    # Design the loss function for the classification task.
    # Hint: You might want to use nn.CrossEntropyLoss(). You can also 
    # optionally combine it with nn.MSELoss() to maintain image quality.
    
    criterion_cls = None
    criterion_mse = None
    pass
    # ─────────────────────────────────────────────────────────────── #
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            reconstructed, logits = model(data, snr_db)
            
            # ─── YOUR CODE HERE ──────────────────────────────────────────── #
            # Calculate the total loss
            loss = None
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Calculate classification accuracy during training
            # delta = N_correct / N
            # correct += ...
            # total += targets.size(0)
            pass
            # ─────────────────────────────────────────────────────────────── #

        if loss is not None:
            avg_loss = total_loss / len(train_loader)
            acc = 100. * correct / total if total > 0 else 0
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%")
        
    return model


def evaluate_model(model, test_loader, snr_db):
    model.eval()
    mse_criterion = nn.MSELoss(reduction='mean')
    
    total_mse = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            reconstructed, logits = model(data, snr_db)
            
            # Reconstruction metric (PSNR)
            if reconstructed is not None and reconstructed.shape == data.shape:
                total_mse += mse_criterion(reconstructed, data).item()
            
            # ─── YOUR CODE HERE ──────────────────────────────────────────── #
            # Classification metric (Accuracy)
            # Calculate the number of correctly predicted samples
            
            pass
            # ─────────────────────────────────────────────────────────────── #
            
    avg_mse = total_mse / len(test_loader) if len(test_loader) > 0 else 1.0
    psnr = calculate_psnr(avg_mse)
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    return psnr, accuracy


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    trained_models = {}
    
    for snr in TRAIN_SNRS:
        model = train_model(train_loader, snr)
        trained_models[f"Train SNR {snr}dB"] = model

    results_psnr = {name: [] for name in trained_models.keys()}
    results_acc = {name: [] for name in trained_models.keys()}
    
    print("\n--- Evaluating Models over Test SNRs ---")
    for test_snr in TEST_SNRS:
        print(f"Testing at SNR = {test_snr} dB...")
        for name, model in trained_models.items():
            psnr, acc = evaluate_model(model, test_loader, test_snr)
            results_psnr[name].append(psnr)
            results_acc[name].append(acc)
            
    # Print numerical results
    print("\n--- Final Performance Results (Accuracy & PSNR) ---")
    print(f"{'Test SNR':<10} | " + " | ".join([f"{name} Acc / PSNR" for name in trained_models.keys()]))
    
    for i, test_snr in enumerate(TEST_SNRS):
        print(f"{test_snr:<10} | ", end="")
        for name in trained_models.keys():
            acc_val = results_acc[name][i]
            psnr_val = results_psnr[name][i]
            print(f"{acc_val:>6.2f}% / {psnr_val:>6.2f}dB | ", end="")
        print()

    # Plotting Accuracy
    plt.figure(figsize=(10, 5))
    markers = ['o', 's', '^']
    for (name, acc_list), marker in zip(results_acc.items(), markers):
        acc_list_clean = [a for a in acc_list if a != 0.0]
        if acc_list_clean:
            plt.plot(TEST_SNRS[:len(acc_list_clean)], acc_list_clean, marker=marker, linewidth=2, label=name)
        
    plt.title('Task-Oriented Semantic Comm: Classification Accuracy vs Test SNR')
    plt.xlabel('Test SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.xticks(TEST_SNRS)
    
    if not os.path.exists('../exp'):
        os.makedirs('../exp')
    plt.savefig('../exp/accuracy_vs_snr.png')
    print("\nPlot saved to '../exp/accuracy_vs_snr.png'")

if __name__ == '__main__':
    main()
    print("end")