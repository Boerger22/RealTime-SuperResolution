import argparse
from train import train


def main():
    # Add cli parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./config.yaml")

    subparsers = parser.add_subparsers(dest="mode")

    subparsers.add_parser("train")
    subparsers.add_parser("test")

    args = parser.parse_args()

    print(args)

    if args.mode == "train":
        train(config_path=args.config)
    elif args.mode == "test":
        ...
    else:
        print("Please specify a valid argument.")
        exit()


if __name__ == "__main__":
    main()
