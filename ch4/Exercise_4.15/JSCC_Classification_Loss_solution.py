import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
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


# ---------- Training Script ----------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_class = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' else torchvision.datasets.CIFAR100
    num_classes = 10 if args.dataset == 'cifar10' else 100

    train_loader = torch.utils.data.DataLoader(
        dataset_class(root='./data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset_class(root='./data', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False)

    # Model and Optimizer
    model = JointSemanticCommSystem(args.latent_dim, num_classes, args.snr_dB).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_cls = nn.CrossEntropyLoss()
    loss_rec = nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            out_cls, out_rec = model(images)

            l_cls = loss_cls(out_cls, labels)
            l_rec = loss_rec(out_rec, images)
            loss = args.lambda_cls * l_cls + args.lambda_rec * l_rec

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = out_cls.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'Loss': f"{total_loss/(total/args.batch_size):.4f}",
                'Acc': f"{100.*correct/total:.2f}%"
            })

        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out_cls, _ = model(images)
            _, predicted = out_cls.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100. * correct / total:.2f}%")


# ---------- Argument Parser ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Communication System Trainer")

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--snr-dB", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-rec", type=float, default=1.0)

    args = parser.parse_args()
    train(args)
