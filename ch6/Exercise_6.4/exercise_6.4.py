import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, Dataset, Subset

# Hyperparameters
NUM_CLIENTS = 20
NUM_ROUNDS = 100
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
ALPHA = 0.5
MU_VALUES = [0, 0.01, 0.1]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000)


# Non-i.i.d. partition using Dirichlet
def partition_data(data, alpha):
    labels = np.array(data.targets)
    idxs = np.arange(len(data))
    class_idxs = [idxs[labels == i] for i in range(10)]

    client_data = [[] for _ in range(NUM_CLIENTS)]
    for c in class_idxs:
        np.random.shuffle(c)
        proportions = np.random.dirichlet(np.repeat(alpha, NUM_CLIENTS))
        proportions = (np.cumsum(proportions) * len(c)).astype(int)[:-1]
        splits = np.split(c, proportions)
        for i, split in enumerate(splits):
            client_data[i].extend(split)

    return [Subset(data, client_data[i]) for i in range(NUM_CLIENTS)]


# Local training with FedProx
def local_train(model, global_model, train_loader, mu):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)

            # FedProx proximal term
            prox_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                prox_term += ((w - w_t.detach()) ** 2).sum()
            loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()


# Evaluate global model
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


# Main simulation
results = {}

for mu in MU_VALUES:
    client_datasets = partition_data(train_data, ALPHA)
    global_model = CNN().to(DEVICE)
    acc_list = []

    for round in range(NUM_ROUNDS):
        local_models = []
        for client_idx in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            train_loader = DataLoader(client_datasets[client_idx], batch_size=BATCH_SIZE, shuffle=True)
            local_train(local_model, global_model, train_loader, mu)
            local_models.append(local_model)

        # Aggregate
        global_dict = global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.stack([local_models[i].state_dict()[key].float() for i in range(NUM_CLIENTS)], 0).mean(0)
        global_model.load_state_dict(global_dict)

        # Evaluate
        acc = evaluate(global_model)
        acc_list.append(acc)
        print(f"Round {round + 1}/{NUM_ROUNDS}, μ={mu}, Accuracy: {acc:.4f}")

    results[mu] = acc_list


# Plotting
plt.figure(figsize=(10, 6))
for mu in MU_VALUES:
    plt.plot(results[mu], label=f'μ={mu}')
plt.xlabel('Communication Round')
plt.ylabel('Test Accuracy')
plt.title('FedProx Performance with Different Proximal Terms')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('fedprox_results.png')
plt.show()
