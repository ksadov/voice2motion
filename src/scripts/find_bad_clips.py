import os
import argparse

from pathlib import Path
from numpy import isin
from tqdm import tqdm

from src.data.dataset import HeadDataset


def check_dataset(data_dir: Path, n_ctx: int, device: str) -> list:
    """
    Check the dataset for bad clips and return a list of bad clips.

    Args:
        video_dir: Path to the video directory.
        data_dir: Path to the info directory.
        n_mels: Number of mel bands.
        n_ctx: Context window size.
        device: Device to use.

    Returns:
        list: List of bad clips.
    """
    bad_list = []
    bad_count = 0
    hd = HeadDataset(data_dir, n_ctx, device, True, 0)
    for i, data in tqdm(enumerate(hd), total=len(hd)):
        if not isinstance(data, tuple):
            print("found bad data:", data)
            bad_list.append(data)
            bad_count += 1
    print("Bad clips:")
    for bad in bad_list:
        print(bad)
    print(f"Bad clip count: {bad_count}")
    print(f"Bad percentage: {bad_count / len(hd) * 100:.2f}%")
    return bad_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_ctx", type=int, default=30)
    parser.add_argument("--remove_bad", action="store_true")
    args = parser.parse_args()
    bad_list = check_dataset(Path(args.data_dir), args.n_ctx, args.device)
    if args.remove_bad:
        for bad in bad_list:
            os.remove(bad)


if __name__ == "__main__":
    main()
