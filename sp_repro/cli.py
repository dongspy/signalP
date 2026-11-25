import argparse
import sys
from pathlib import Path

import torch

from . import __version__


def positive_int(value):
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return integer


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent academic signal peptide predictor")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="fit the CRF head and evaluate a held-out SignalP fold")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--cache", type=Path)
    train.add_argument("--encoder", type=Path, default=Path(".local-data/esm2"))
    train.add_argument("--encoder-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train.add_argument("--crf-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    train.add_argument("--encoder-batch-size", type=positive_int, default=128)
    train.add_argument("--batch-size", type=positive_int, default=128)
    train.add_argument("--threads", type=positive_int, default=4)
    train.add_argument("--epochs", type=positive_int, default=8)
    train.add_argument("--validation-fold", type=int, choices=(0, 1, 2), default=2)
    train.add_argument("--test-fold", type=int, choices=(0, 1, 2), default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--learning-rate", type=float, default=0.002)
    train.add_argument("--class-weight-power", type=float, default=0.5)
    train.add_argument("--limit-per-class", type=positive_int, help="debug run with at most this many records per fold and class")

    predict = commands.add_parser("predict", help="predict signal peptides from protein FASTA")
    predict.add_argument("--fasta", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--encoder", type=Path, default=Path(".local-data/esm2"))
    predict.add_argument("--organism", choices=("other", "eukarya"), default="other")
    predict.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    predict.add_argument("--batch-size", type=positive_int, default=64)
    predict.add_argument("--threads", type=positive_int, default=4)
    predict.add_argument("--per-residue", action="store_true")

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    try:
        if args.command == "train":
            from .training import train as run
        else:
            from .prediction import predict as run
        run(args)
    except (OSError, ValueError) as error:
        print(f"sp-repro: {error}", file=sys.stderr)
        raise SystemExit(1) from error
