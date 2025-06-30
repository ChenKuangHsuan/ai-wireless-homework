import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm  # Import tqdm for progress bars
import math

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate JSCC model with different SNRs")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device for training (cpu or cuda)')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10', help='Dataset to use (cifar10 or cifar100)')
    parser.add_argument('--k', type=int, default=128, help='Dimension of latent space (k)')
    parser.add_argument('--n', type=int, default=3072, help='Dimension of input image (n), for CIFAR-10 and CIFAR-100 this is 3072')
    return parser.parse_args()

# Set device
def set_device(device_choice):
    return torch.device(device_choice if torch.cuda.is_available() and device_choice == 'cuda' else 'cpu')

# Load CIFAR-10 or CIFAR-100 dataset
def load_data(batch_size, dataset_name):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if dataset_name == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# Define the AWGN channel function
def awgn_channel(x, snr_dB):
    snr_linear = 10 ** (snr_dB / 10)
    signal_power = torch.mean(x ** 2)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * torch.sqrt(noise_power / 2)
    return x + noise

# PSNR Calculation
def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # If MSE is zero, PSNR is infinite (perfect reconstruction)
    max_pixel = 1.0  # For normalized images, the max pixel value is 1
    psnr = 10 * torch.log10(max_pixel**2 / mse)
    return psnr

# Encoder: Convolutional Neural Network
class Encoder(nn.Module):
    def __init__(self, k):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # Output dimension k (latent space representation)
        self.fc = nn.Linear(256 * 4 * 4, k) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc(x)) 
        return x

# Decoder: Deconvolutional Neural Network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(128, 256 * 4 * 4)  
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = x.view(x.size(0), 256, 4, 4)
        x = torch.relu(self.deconv1(x))  
        x = torch.relu(self.deconv2(x)) 
        x = torch.sigmoid(self.deconv3(x)) 
        return x

# Complete JSCC model combining Encoder and Decoder
class JSCC_Model(nn.Module):
    def __init__(self, k):
        super(JSCC_Model, self).__init__()
        self.encoder = Encoder(k)
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)  # Get latent representation
        decoded = self.decoder(encoded)  # Decode to reconstruct image
        return decoded

# MSE Loss Function
def mse_loss(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2)

# Training loop with tqdm for progress bar
def train(model, trainloader, snr_dB, epochs, optimizer, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for data in progress_bar:
            inputs, _ = data
            inputs = inputs.to(device)

            noisy_inputs = awgn_channel(inputs, snr_dB)  # Add noise

            optimizer.zero_grad()
            outputs = model(noisy_inputs)  # Forward pass
            loss = mse_loss(inputs, outputs)  # MSE loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / len(trainloader))

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

# Test the model and calculate test loss
def test(model, testloader, snr_dB, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="Testing", leave=False)
        for data in progress_bar:
            inputs, _ = data
            inputs = inputs.to(device)

            noisy_inputs = awgn_channel(inputs, snr_dB)
            outputs = model(noisy_inputs)

            loss = mse_loss(inputs, outputs)
            total_loss += loss.item()

            psnr = calculate_psnr(inputs, outputs)
            total_psnr += psnr.item()

            progress_bar.set_postfix(loss=total_loss / len(testloader))

    avg_test_loss = total_loss / len(testloader)
    avg_psnr = total_psnr / len(testloader)
    print(f"Test Loss: {avg_test_loss}")
    print(f"Average PSNR: {avg_psnr}")

def main():
    # Parse command line arguments
    args = parse_args()

    # Set the device for training (CPU or GPU)
    device = set_device(args.device)

    # Load CIFAR-10 or CIFAR-100 data
    trainloader, testloader = load_data(args.batch_size, args.dataset)

    # Initialize the model
    model = JSCC_Model(args.k).to(device)

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model for different SNRs
    for snr in [1, 7, 20]:
        print(f"Training with SNR = {snr} dB")
        train(model, trainloader, snr, args.epochs, optimizer, device)

    # Test the model and calculate test loss for a range of SNRs
    for snr in range(0, 21, 2):
        print(f"Evaluating performance at SNR = {snr} dB")
        test(model, testloader, snr, device)

if __name__ == '__main__':
    main()