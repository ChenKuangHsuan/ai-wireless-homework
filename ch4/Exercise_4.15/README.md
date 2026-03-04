# Exercise 4.15: Task-Oriented Semantic Communication for Image Classification

This repository provides the starter code for Exercise 4.15. While Exercise 4.14 focused on image reconstruction (measured by PSNR), this exercise shifts the focus to a downstream task: **Image Classification**. 

Your goal is to modify the semantic communication system to output class predictions, design a suitable loss function, and compare its classification and reconstruction performance against the pure reconstruction model.

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Network Architecture** | Open `exercise_4.15_starter.py`. Modify the Decoder (or add a new module) to output **both** the reconstructed image and the classification logits (10 classes for CIFAR10). |
| **Loss Function** | Design the loss function for the classification task. You will likely need `nn.CrossEntropyLoss()`. You can also experiment with a joint loss function (e.g., combining MSE and Cross-Entropy) to maintain some PSNR performance. |
| **Accuracy Metric** | Implement the calculation for classification accuracy: $\delta = N_{correct}/N$. |
| **Train & Compare** | Run the code and observe the PSNR and Accuracy metrics. Compare these results with the outputs from Exercise 4.14. You should notice a trade-off between pure reconstruction quality and task performance. |

## Files
| File | Purpose |
|------|---------|
| `exercise_4.15_starter.py` | Starter script (with TODOs) for task-oriented training. |