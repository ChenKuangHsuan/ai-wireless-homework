# Exercise 4.14: Semantic Communication System for Images over AWGN Channel

This repository provides the starter code for Exercise 4.14. Your task is to implement a Deep Joint Source-Channel Coding (DeepJSCC) semantic communication network based on CNNs, transmit images over an AWGN channel, and evaluate its performance on the CIFAR10 dataset.

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `exercise_4.14_starter.py` and complete the `# YOUR CODE HERE` blocks. You need to:<br>  • Design a novel CNN-based structure for both Encoder and Decoder.<br>  • Implement the AWGN channel with power normalization.<br>  • Define the new loss function.<br>  • Calculate the transmission ratio k/n.<br>  • Calculate the PSNR metric according to (4.40). |
| **Run** | Execute: `python exercise_4.14_starter.py`<br>*(Note: Ensure you have installed the `torch`, `torchvision`, and `matplotlib` libraries first).* |
| **Observe** | The script will train the network at SNRs of 1 dB, 7 dB, and 20 dB. After training, it will evaluate their performance over the SNR range of [0, 20] dB with steps of 2 dB. Observe the final PSNR vs. SNR plot saved in the `../exp` directory. |

## Files
| File | Purpose |
|------|---------|
| `exercise_4.14_starter.py` | Starter script (with TODOs). |