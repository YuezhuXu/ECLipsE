import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


INPUT_SIZE = 5
OUTPUT_SIZE = 2

SMALL_LAYERS = [5, 10, 15, 20, 25]
SMALL_WIDTHS = [10, 20, 40, 60]
LARGE_LAYERS = [30, 40, 50, 60, 70]
LARGE_WIDTHS = [60, 80, 100, 120]
RADIUS_LAYERS = [5, 30, 60]
RADIUS_WIDTHS = [128]

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
RANDOM_DIR = PAPER_DIR / "datasets" / "random"


def generate_network(layers, width, norm_range):
    np.random.seed(width * 77 + layers * 9)
    net_dims = [INPUT_SIZE] + [width] * (layers - 1) + [OUTPUT_SIZE]
    weights = []
    biases = []

    for in_dim, out_dim in zip(net_dims[:-1], net_dims[1:]):
        target_norm = np.random.uniform(*norm_range)
        weight = np.random.randn(out_dim, in_dim)
        weight = target_norm * weight / np.linalg.norm(weight, 2)
        bias = np.random.randn(out_dim)
        weights.append(weight)
        biases.append(bias)

    return weights, biases


def save_network(weights, biases, output_path):
    weight_cells = np.empty((1, len(weights)), dtype=object)
    bias_cells = np.empty((1, len(biases)), dtype=object)
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        weight_cells[0, index] = weight
        bias_cells[0, index] = bias
    savemat(output_path, {"weights": weight_cells, "biases": bias_cells})


def case_configurations(case):
    cases = {
        "small": (SMALL_LAYERS, SMALL_WIDTHS, (0.8, 2.5)),
        "large": (LARGE_LAYERS, LARGE_WIDTHS, (0.8, 2.5)),
        "radius": (RADIUS_LAYERS, RADIUS_WIDTHS, (2.0, 2.5)),
    }
    selected = cases if case == "all" else {case: cases[case]}
    for layers, widths, norm_range in selected.values():
        for layer_count in layers:
            for width in widths:
                yield layer_count, width, norm_range


def generate_all_random(case="all", output_dir=RANDOM_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for layers, width, norm_range in case_configurations(case):
        output_path = output_dir / f"lyr{layers}n{width}.mat"
        weights, biases = generate_network(layers, width, norm_range)
        save_network(weights, biases, output_path)
        print(f"Generated {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        nargs="?",
        choices=("small", "large", "radius", "all"),
        default="all",
    )
    args = parser.parse_args()
    generate_all_random(args.case)


if __name__ == "__main__":
    main()
