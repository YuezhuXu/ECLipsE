import argparse
import random
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 77
EPOCHS = 50
BATCH = 128
WIDTHS = (128, 128, 128)
ACT = "elu"
LR = 1e-3
WD = 1e-4

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
DATA_DIR = PAPER_DIR / "datasets"
TRAINED_DIR = DATA_DIR / "MNIST" / "trained_NN"


def set_seed(s=0):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, in_dim=784, widths=(256, 256, 256), num_classes=10, act="silu"):
        super().__init__()
        activation = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "elu": nn.ELU,
        }.get(act, nn.ReLU)
        layers = []
        last = in_dim
        for width in widths:
            layers += [nn.Linear(last, width), activation()]
            last = width
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def jacobian_frobenius_sqr(model, x, n_proj=1):
    x = x.detach()
    x.requires_grad_(True)
    logits = model(x)
    total = 0.0
    for _ in range(n_proj):
        v = torch.empty_like(logits).uniform_(-1, 1).sign()
        vjp = torch.autograd.grad(
            outputs=logits,
            inputs=x,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]
        total = total + vjp.flatten(1).pow(2).sum(dim=1).mean()
    return total / n_proj


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    ok = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ok += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return ok / total


def train(model, train_loader, test_loader, device, lambda_jr=0.0, n_proj=1, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best = 0.0
    checkpoint = None
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            cross_entropy = F.cross_entropy(logits, y)
            loss = cross_entropy + (
                lambda_jr * jacobian_frobenius_sqr(model, x, n_proj=n_proj)
                if lambda_jr > 0
                else cross_entropy * 0
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.size(0)
        test_accuracy = accuracy(model, test_loader, device)
        if test_accuracy > best:
            best = test_accuracy
            checkpoint = {"model": model.state_dict()}
        print(
            f"E{epoch:02d}  loss={total / len(train_loader.dataset):.4f}  "
            f"acc={test_accuracy * 100:.2f}%  {(time.time() - start):.1f}s"
        )
        if test_accuracy > 0.98:
            break

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    return model, best


def save_mlp_to_mat(model, output_path):
    weights = []
    biases = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach().cpu().double().numpy()
            bias = layer.bias.detach().cpu().double().numpy().reshape(1, -1)
            weights.append(weight)
            biases.append(bias)

    weight_cells = np.empty((1, len(weights)), dtype=object)
    bias_cells = np.empty((1, len(biases)), dtype=object)
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        weight_cells[0, index] = weight
        bias_cells[0, index] = bias

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output_path, {"weights": weight_cells, "biases": bias_cells})


def create_data_loaders():
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_model(model_name, train_loader, test_loader, device):
    lambda_jr = 0.0 if model_name == "base" else 1
    model = MLP(784, WIDTHS, 10, ACT).to(device)
    label = "Baseline" if model_name == "base" else "JR"
    print(f"{label} params: {count_params(model) / 1e6:.2f}M")
    model, clean_accuracy = train(
        model,
        train_loader,
        test_loader,
        device,
        lambda_jr=lambda_jr,
        n_proj=1,
        epochs=EPOCHS,
    )
    print(f"{label} clean={clean_accuracy * 100:.2f}%")
    return model


def export_model(model, model_name, output_dir=TRAINED_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mlp_to_mat(model, output_dir / f"mnist_{model_name}.mat")
    torch.save(model.state_dict(), output_dir / f"mnist_{model_name}.pt")


def train_all(model_names=("base", "jr")):
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = create_data_loaders()
    for model_name in model_names:
        model = train_model(model_name, train_loader, test_loader, device)
        export_model(model, model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", choices=("base", "jr", "all"), default="all")
    args = parser.parse_args()
    train_all(("base", "jr") if args.model == "all" else (args.model,))


if __name__ == "__main__":
    main()
