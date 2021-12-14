from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import pandas as pd


def create_df(base_dir):
    data = defaultdict(list)
    for f in base_dir.glob("*/*.png"):
        chara_dir = f.parent.name
        tmp = chara_dir.split("_")
        code = int(tmp[0])
        name = "_".join(tmp[1:])
        path = f.relative_to(base_dir)
        data["code"].append(code)
        data["name"].append(name)
        data["path"].append(path)
    df = pd.DataFrame(data)
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=Path)
    parser.add_argument("--save_path", type=Path, default="train.csv")
    args = parser.parse_args()
    df = create_df(args.image_path)
    df.to_csv(str(args.save_path), index=False)


if __name__ == "__main__":
    main()
