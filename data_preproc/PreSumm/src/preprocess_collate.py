import os
import glob
import torch
from loguru import logger


def io_parser(description=None):
    import argparse

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-o",
        "--output_dir",
        default="",
        action="store",
        type=str,
        help="Directory to which output is written",
    )
    parser.add_argument(
        "-d",
        "--input_dir",
        default="",
        action="store",
        type=str,
        help="an input directory path",
    )

    return parser


def main():
    parser = io_parser("preprocess_collate")
    args = parser.parse_args()

    logger.info(args)
    train_data = []
    val_data = []
    test_data = []
    all_data = []

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError("input_dir must exist: {}".format(args.input_dir))
    base_dir = os.path.realpath(args.input_dir)
    out_dir = args.output_dir or base_dir
    os.makedirs(out_dir, exist_ok=True)

    for fname in glob.glob(os.path.join(base_dir, "*.pt")):
        if "train" in fname:
            train_data.extend(torch.load(fname))
        elif "valid" in fname:
            val_data.extend(torch.load(fname))
        elif "test" in fname:
            test_data.extend(torch.load(fname))
        else:
            all_data.extend(torch.load(fname))
    if len(train_data) > 0:
        fn = os.path.join(out_dir, "train.bert.pt")
        torch.save(train_data, fn)
        logger.info(f"saved {fn}")
    if len(val_data) > 0:
        fn = os.path.join(out_dir, "valid.bert.pt")
        torch.save(val_data, fn)
        logger.info(f"saved {fn}")
    if len(test_data) > 0:
        fn = os.path.join(out_dir, "test.bert.pt")
        torch.save(test_data, fn)
        logger.info(f"saved {fn}")
    else:
        fn = os.path.join(out_dir, "all.bert.pt")
        torch.save(all_data, fn)
        logger.warning("did not save any test .pt files")


if __name__ == "__main__":
    main()
