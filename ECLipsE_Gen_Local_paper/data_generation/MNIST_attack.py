import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from training_mnist_jacreg import ACT, WIDTHS, MLP, set_seed


BATCH = 256
PGD_STEPS = 40
EPS_LIST = [
    1 / 2,
    1 / 4,
    1 / 8,
    1 / 16,
    1 / 32,
    1 / 64,
    1 / 128,
    1 / 256,
]
MAX_BATCHES = None
SEED = 77

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
DATA_DIR = PAPER_DIR / "datasets"
TRAINED_DIR = DATA_DIR / "MNIST" / "trained_NN"
RESULTS_DIR = DATA_DIR / "MNIST" / "results"
MODEL_PATHS = [
    TRAINED_DIR / "mnist_base.pt",
    TRAINED_DIR / "mnist_jr.pt",
]


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def pgd_l2_attack(model, x, y, eps=1.0, steps=40, alpha=None):
    model.eval()
    if alpha is None:
        alpha = eps / 10
    x0 = x.clone().detach()
    delta = torch.zeros_like(x).normal_(0, 1e-6)
    direction = delta.flatten(1)
    direction = direction / (direction.norm(p=2, dim=1, keepdim=True) + 1e-12) * eps
    delta = direction.view_as(delta).detach().requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(model((x0 + delta).clamp(0, 1)), y)
        gradient = torch.autograd.grad(loss, delta)[0].flatten(1)
        gradient = gradient / (gradient.norm(p=2, dim=1, keepdim=True) + 1e-12)
        delta = (delta.flatten(1) + alpha * gradient).view_as(delta)
        direction = delta.flatten(1)
        norm = direction.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        direction = direction * (eps / norm).clamp(max=1.0)
        delta = direction.view_as(delta).detach().requires_grad_(True)
    return (x0 + delta).clamp(0, 1)


def failure_rate(model, loader, device, eps=1.0, steps=40, max_batches=None):
    model.eval()
    changed = 0
    total = 0
    for batch_index, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_prediction = model(x).argmax(1)
        x_adv = pgd_l2_attack(model, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            adversarial_prediction = model(x_adv).argmax(1)
        changed += (adversarial_prediction != clean_prediction).sum().item()
        total += y.numel()
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return changed / total


def create_test_loader():
    test_set = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        test_set,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def generate_pgd_results(output_path=RESULTS_DIR / "pgd_failure_rates.csv"):
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    test_loader = create_test_loader()
    rows = []

    for model_path in MODEL_PATHS:
        model = MLP(784, WIDTHS, 10, ACT).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"{model_path.stem} clean accuracy: {accuracy(model, test_loader, device):.6f}")

        for eps in EPS_LIST:
            rate = failure_rate(
                model,
                test_loader,
                device,
                eps=eps,
                steps=PGD_STEPS,
                max_batches=MAX_BATCHES,
            )
            print(f"{model_path.stem}, radius={eps}, failure_rate={rate:.6f}")
            rows.append({"model": model_path.stem, "epsilon": eps, "failure_rate": rate})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("model", "epsilon", "failure_rate"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")
    return rows


if __name__ == "__main__":
    generate_pgd_results()
