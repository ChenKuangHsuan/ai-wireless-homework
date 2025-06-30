import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------- Model Components ----------
class SemanticEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 64, 8, 8)
        return self.decoder(x)


class AWGNChannel(nn.Module):
    def __init__(self, snr_dB=10):
        super().__init__()
        self.snr_dB = snr_dB

    def forward(self, x):
        snr = 10 ** (self.snr_dB / 10.0)
        x_power = x.pow(2).mean()
        noise_power = x_power / snr
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise


class SemanticDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class JointSemanticCommSystem(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10, snr_dB=10):
        super().__init__()
        self.encoder = SemanticEncoder(latent_dim)
        self.channel = AWGNChannel(snr_dB)
        self.decoder_cls = SemanticDecoder(latent_dim, num_classes)
        self.decoder_img = ImageDecoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        noisy = self.channel(latent)
        cls_out = self.decoder_cls(noisy)
        rec_img = self.decoder_img(noisy)
        return cls_out, rec_img


# ---------- Evaluation ----------
def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2, reduction='none')
    mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)
    psnr = 10 * torch.log10((max_val ** 2) / (mse_per_sample + 1e-8))
    return psnr.mean().item()


def evaluate(model, dataloader, device):
    model.eval()
    correct, total, total_psnr = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            out_cls, out_rec = model(images)
            _, predicted = out_cls.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            total_psnr += compute_psnr(out_rec, images) * images.size(0)
    return correct / total, total_psnr / total


# ---------- Training ----------
def run_experiment(lambda_cls=1.0, lambda_rec=0.0, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_loader = DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
        batch_size=128, shuffle=False)

    model = JointSemanticCommSystem().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_cls = nn.CrossEntropyLoss()
    loss_rec = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out_cls, out_rec = model(images)
            l_cls = loss_cls(out_cls, labels)
            l_rec = loss_rec(out_rec, images)
            loss = lambda_cls * l_cls + lambda_rec * l_rec
            loss.backward()
            optimizer.step()

    acc, psnr = evaluate(model, test_loader, device)
    return acc, psnr


# ---------- Run & Plot ----------
if __name__ == "__main__":
    acc_cls, psnr_cls = run_experiment(lambda_cls=1.0, lambda_rec=0.0)
    acc_joint, psnr_joint = run_experiment(lambda_cls=1.0, lambda_rec=1.0)

    labels = ['Classification Only', 'Joint Loss']
    acc = [acc_cls, acc_joint]
    psnr = [psnr_cls, psnr_joint]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar([x - 0.2 for x in range(2)], acc, width=0.4, label='Accuracy', color='tab:blue')
    ax2.bar([x + 0.2 for x in range(2)], psnr, width=0.4, label='PSNR', color='tab:green')

    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('PSNR (dB)')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels)
    ax1.set_title('Comparison: Accuracy vs PSNR')
    plt.tight_layout()
    plt.show()
