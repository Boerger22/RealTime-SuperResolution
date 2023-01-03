import argparse
from train import train
from evaluate import evaluate


def main():
    # Add cli parameters
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # train parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, default="./config.yaml")

    # test parser
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model_path", type=str, required=True)
    test_parser.add_argument("--eval_path", type=str)
    test_parser.add_argument("--config", type=str)

    args = parser.parse_args()

    if args.mode == "train":
        train(config_path=args.config)
    elif args.mode == "test":
        evaluate(model_path=args.model_path, config_path=args.config, evaluation_path=args.eval_path)
    else:
        print("Please specify a valid argument.")
        exit()


if __name__ == "__main__":
    main()
