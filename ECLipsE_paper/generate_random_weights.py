import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


INPUT_SIZE = 4
OUTPUT_SIZE = 1
NORM_RANGE = (0.4, 1.8)

STANDARD_LAYERS = [2, 5, 10, 20, 30, 50, 75, 100]
STANDARD_WIDTHS = [20, 40, 60, 80, 100]
DEEP_LAYERS = [100]
DEEP_WIDTHS = [120, 140, 160]
WIDE_LAYERS = [50]
WIDE_WIDTHS = [150, 200, 300, 400, 500, 1000]

SCRIPT_DIR = Path(__file__).resolve().parent
RANDOM_DIR = SCRIPT_DIR / "datasets" / "random"


def generate_network(layers, width):
    np.random.seed(width * 7 + layers * 13)
    net_dims = [INPUT_SIZE] + [width] * (layers - 1) + [OUTPUT_SIZE]
    weights = []

    for in_dim, out_dim in zip(net_dims[:-1], net_dims[1:]):
        target_norm = np.random.uniform(*NORM_RANGE)
        weight = np.random.randn(out_dim, in_dim)
        weight = target_norm * weight / np.linalg.norm(weight, 2)
        weights.append(weight)

    return weights


def save_network(weights, output_path):
    weight_cells = np.empty((1, len(weights)), dtype=object)
    for index, weight in enumerate(weights):
        weight_cells[0, index] = weight
    savemat(output_path, {"weights": weight_cells})


def case_configurations(case):
    cases = {
        "standard": (STANDARD_LAYERS, STANDARD_WIDTHS),
        "deep": (DEEP_LAYERS, DEEP_WIDTHS),
        "wide": (WIDE_LAYERS, WIDE_WIDTHS),
    }
    selected = cases if case == "all" else {case: cases[case]}
    for layers, widths in selected.values():
        for layer_count in layers:
            for width in widths:
                yield layer_count, width


def generate_all_random(case="all", output_dir=RANDOM_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for layers, width in case_configurations(case):
        output_path = output_dir / f"lyr{layers}n{width}test1.mat"
        save_network(generate_network(layers, width), output_path)
        print(f"Generated {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        nargs="?",
        choices=("standard", "deep", "wide", "all"),
        default="all",
    )
    args = parser.parse_args()
    generate_all_random(args.case)


if __name__ == "__main__":
    main()
