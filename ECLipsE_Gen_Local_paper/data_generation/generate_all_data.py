import argparse


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", action="store_true")
    group.add_argument("--mnist", action="store_true")
    group.add_argument("--pgd", action="store_true")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.random:
        from generate_random_ECLipsE_Gen_Local import generate_all_random

        generate_all_random()
    elif args.mnist:
        from training_mnist_jacreg import train_all

        train_all()
    elif args.pgd:
        from MNIST_attack import generate_pgd_results

        generate_pgd_results()
    else:
        from generate_random_ECLipsE_Gen_Local import generate_all_random
        from MNIST_attack import generate_pgd_results
        from training_mnist_jacreg import train_all

        generate_all_random()
        train_all()
        generate_pgd_results()


if __name__ == "__main__":
    main()
