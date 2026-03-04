"""
Exercise 4.14: Semantic Communication System for Images over an AWGN Channel

This script sets up a CNN-based Deep Joint Source-Channel Coding (DeepJSCC) 
system to transmit CIFAR10 images over an AWGN channel.

TODO:
Complete the semantic communication system implementation:
1. Design a novel CNN-based structure for the Encoder and Decoder.
2. Implement the AWGN channel (power normalization + noise addition).
3. Derive and define the new loss function.
4. Calculate the transmission ratio k/n.
5. Calculate the PSNR metric.
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, out_channels=16):
        super(Encoder, self).__init__()
        # ─── YOUR CODE HERE ──────────────────────────────────────────── #
        # Design a novel network structure based on the CNN for the encoder.
        # Input shape: (Batch, 3, 32, 32)
        
        self.net = nn.Sequential(
            
        )
        pass
        # ─────────────────────────────────────────────────────────────── #

    def forward(self, x):
        return self.net(x)


class AWGNChannel(nn.Module):
    def __init__(self):
        super(AWGNChannel, self).__init__()

    def forward(self, x, snr_db):
        # ─── YOUR CODE HERE ──────────────────────────────────────────── #
        # 1. Perform power normalization to meet average power constraint P = 1.
        # 2. Calculate noise variance based on snr_db and add Gaussian noise.
        
        x_noisy = x # Replace this with your normalized and noisy signal
        pass
        # ─────────────────────────────────────────────────────────────── #
        return x_noisy


class Decoder(nn.Module):
    def __init__(self, in_channels=16):
        super(Decoder, self).__init__()
        # ─── YOUR CODE HERE ──────────────────────────────────────────── #
        # Design a novel network structure based on the CNN for the decoder.
        # Ensure the final output matches the original image shape and pixel range [0, 1].
        
        self.net = nn.Sequential(
            
        )
        pass
        # ─────────────────────────────────────────────────────────────── #

    def forward(self, x):
        return self.net(x)


class DeepJSCC(nn.Module):
    def __init__(self, channel_dim=16):
        super(DeepJSCC, self).__init__()
        self.encoder = Encoder(out_channels=channel_dim)
        self.channel = AWGNChannel()
        self.decoder = Decoder(in_channels=channel_dim)
        
    def forward(self, x, snr_db):
        encoded = self.encoder(x)
        received = self.channel(encoded, snr_db)
        decoded = self.decoder(received)
        return decoded, encoded.shape


def calculate_psnr(mse):
    # ─── YOUR CODE HERE ──────────────────────────────────────────── #
    # Calculate the PSNR metric according to (4.40).
    # Assume the max pixel value is 1.0.
    
    psnr = 0.0 # Replace with your calculation
    pass
    # ─────────────────────────────────────────────────────────────── #
    return psnr


def train_model(train_loader, snr_db):
    print(f"\n--- Training Network at SNR = {snr_db} dB ---")
    model = DeepJSCC(channel_dim=16).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ─── YOUR CODE HERE ──────────────────────────────────────────── #
    # Derive and define the new loss function.
    criterion = None 
    pass
    # ─────────────────────────────────────────────────────────────── #
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            reconstructed, encoded_shape = model(data, snr_db)
            
            # ─── YOUR CODE HERE ──────────────────────────────────────────── #
            # Calculate the loss
            loss = None
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pass
            # ─────────────────────────────────────────────────────────────── #
            
            if batch_idx == 0 and epoch == 0:
                # ─── YOUR CODE HERE ──────────────────────────────────────────── #
                # Calculate the transmission ratio k/n
                # Input original dimension (n) vs Channel dimension (k)
                
                n = 0
                k = 0
                if n != 0:
                    print(f"[Info] Input dim (n): {n}, Channel dim (k): {k}")
                    print(f"[Info] Transmission ratio k/n: {k}/{n} = {k/n:.4f}")
                pass
                # ─────────────────────────────────────────────────────────────── #

        if criterion is not None and loss is not None:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
        
    return model


def evaluate_model(model, test_loader, snr_db):
    model.eval()
    
    # Use MSE for PSNR calculation
    criterion = nn.MSELoss(reduction='mean')
    total_mse = 0.0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(DEVICE)
            reconstructed, _ = model(data, snr_db)
            total_mse += criterion(reconstructed, data).item()
            
    avg_mse = total_mse / len(test_loader)
    return calculate_psnr(avg_mse)


def main():
    # Load CIFAR10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    trained_models = {}
    
    # Train the network with different SNRs (1, 7, 20 dB)
    for snr in TRAIN_SNRS:
        model = train_model(train_loader, snr)
        trained_models[f"Train SNR {snr}dB"] = model

    # Evaluate performance over [0, 20] dB with steps of 2
    results = {name: [] for name in trained_models.keys()}
    
    print("\n--- Evaluating Models over Test SNRs ---")
    for test_snr in TEST_SNRS:
        print(f"Testing at SNR = {test_snr} dB...")
        for name, model in trained_models.items():
            psnr = evaluate_model(model, test_loader, test_snr)
            results[name].append(psnr)
            
    # Print numerical results
    print("\n--- Final PSNR Results ---")
    print(f"{'Test SNR (dB)':<15}", end="")
    for name in trained_models.keys():
        print(f" | {name:<15}", end="")
    print()
    
    for i, test_snr in enumerate(TEST_SNRS):
        print(f"{test_snr:<15}", end="")
        for name in trained_models.keys():
            print(f" | {results[name][i]:<15.2f}", end="")
        print()

    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']
    for (name, psnr_list), marker in zip(results.items(), markers):
        # Filter out inf values if PSNR wasn't implemented yet
        psnr_list_clean = [p for p in psnr_list if p != 0.0] 
        if psnr_list_clean:
            plt.plot(TEST_SNRS[:len(psnr_list_clean)], psnr_list_clean, marker=marker, linewidth=2, label=name)
        
    plt.title('Performance of Semantic Communication System over AWGN Channel')
    plt.xlabel('Test SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.xticks(TEST_SNRS)
    
    if not os.path.exists('../exp'):
        os.makedirs('../exp')
    plt.savefig('../exp/psnr_vs_snr.png')
    print("\nPlot saved to '../exp/psnr_vs_snr.png'")
    # plt.show()

if __name__ == '__main__':
    main()
    print("end")