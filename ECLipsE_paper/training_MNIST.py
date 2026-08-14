import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.io import savemat
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


WIDTHS = [100, 200, 300, 400]
BATCH_SIZE = 64
NUM_EPOCHS = 10

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "datasets"
MNIST_DIR = DATASETS_DIR / "MNIST"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class NeuralNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def create_data_loaders():
    train_data = MNIST(
        root=DATASETS_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_data = MNIST(
        root=DATASETS_DIR,
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train_model(width, train_loader):
    model = NeuralNet(width)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Width {width}, Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

    return model


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def export_mat(model, width, output_dir=MNIST_DIR):
    weights = []
    for layer in model.linear_relu_stack:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().cpu().double().numpy())

    weight_cells = np.empty((1, len(weights)), dtype=object)
    for index, weight in enumerate(weights):
        weight_cells[0, index] = weight

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lyr3n{width}test1.mat"
    savemat(output_path, {"weights": weight_cells})
    print(f"Exported {output_path}")


def train_width(width, train_loader, test_loader):
    model = train_model(width, train_loader)
    accuracy = evaluate_model(model, test_loader)
    print(f"Accuracy of width-{width} model on the test set: {accuracy:.2f}%")
    export_mat(model, width)


def train_all(widths=WIDTHS):
    train_loader, test_loader = create_data_loaders()
    for width in widths:
        train_width(width, train_loader, test_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("width", nargs="?", type=int, choices=WIDTHS)
    args = parser.parse_args()
    train_all(WIDTHS if args.width is None else [args.width])


if __name__ == "__main__":
    main()
