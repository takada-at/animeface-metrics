from argparse import ArgumentParser
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import pandas as pd


def create_fold(df):
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    df["fold"] = -1
    for fold, (train_index, test_index) in enumerate(
        splits.split(df.values, df["code"].values)
    ):
        df.loc[test_index, "fold"] = fold
    return df


def create_dataset(args):
    df_path = args.data_dir / "train.csv"
    df = pd.read_csv(str(df_path))
    df = create_fold(df)
    train_df = df[df["fold"] != 0].reset_index(drop=True)
    valid_df = df[df["fold"] == 0].reset_index(drop=True)
    train_transform = get_transforms(args.enable_augmentation, args.image_size)
    valid_transform = get_transforms(False, args.image_size)
    train_dataset = AnimeFaceDataset(train_df, args.image_dir, train_transform)
    valid_dataset = AnimeFaceDataset(valid_df, args.image_dir, valid_transform)
    return train_dataset, valid_dataset


def get_data_loader(args):
    train_dataset, valid_dataset = create_dataset(args)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_dataloader, valid_dataloader


def add_data_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/takada-at/notebooks/animeface-metrics"),
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=Path(
            "/home/takada-at/notebooks/animeface-metrics/animeface-character-dataset/thumb"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--enable_augmentation", action="store_true")
    return parser


def get_transforms(enable_augmentation: bool, image_size: int) -> A.Compose:
    transforms = [A.Resize(height=image_size, width=image_size)]
    if enable_augmentation:
        transforms += [
            A.RGBShift(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.2),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
        ]
    transforms += [
        A.Normalize(),
        A.pytorch.ToTensorV2(),
    ]
    return A.Compose(transforms)


class AnimeFaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform=None):
        self.codes = df["code"].tolist()
        self.pathes = df["path"].tolist()
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, item):
        code = self.codes[item]
        path = self.pathes[item]
        image_path = str(self.base_dir / path)
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dic = dict(image=img, label=code)
        if self.transform is not None:
            dic = self.transform(**dic)
        return dic
