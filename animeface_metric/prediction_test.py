from animeface_metric import data
from animeface_metric.model import MetricsLearningModel
from argparse import ArgumentParser
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch


def get_neighbors(embeddings, knn: int = 10):
    nn = NearestNeighbors(n_neighbors=knn)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    return distances, indices


def get_image_embedding(model_, dataloader, device):
    embeddings = []
    model_ = model_.to(device)
    model_.eval()
    for batch in dataloader:
        with torch.no_grad():
            image_embedding = model_.predict(batch["image"].to(device))
            embeddings.append(image_embedding.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def f1_score(y_true, y_pred):
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1


def precision_score(y_true, y_pred):
    precision = np.array([len(x[0] & x[1]) / len(x[1]) for x in zip(y_true, y_pred)])
    return precision


def recall_score(y_true, y_pred):
    precision = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return precision


def do_predict(
    embeddings, df, min_threshold, max_threshold, step_threshold, knn: int = 10
):
    thresholds = list(np.arange(min_threshold, max_threshold, step_threshold))
    distances, indices = get_neighbors(embeddings, knn=knn)
    scores = []
    for threshold in thresholds:
        predictions = []
        truthes = []
        for k in range(embeddings.shape[0]):
            idx = np.where(
                distances[
                    k,
                ]
                < threshold
            )[0]
            ids = indices[k, idx]
            label = df["code"][k]
            truth = set(df[df["code"] == label].index)
            predictions.append(set(ids))
            truthes.append(truth)

        truthes = pd.Series(truthes)
        predictions = pd.Series(predictions)
        f1 = f1_score(truthes, predictions)
        precision = precision_score(truthes, predictions)
        recall = recall_score(truthes, predictions)
        score = f1.mean()
        score2 = precision.mean()
        recall = recall.mean()
        print(
            f"Our f1 score for threshold {threshold:.4f} is f1={score:.4f} prec={score2:.4f} recall={recall:.4f}"
        )
        # print(truthes[:3], predictions[:3])
        scores.append(score)


def prediction(args):
    model_ = MetricsLearningModel.load_from_checkpoint(str(args.model_path))
    df_path = args.data_dir / "train.csv"
    df = pd.read_csv(str(df_path))
    df = data.create_fold(df)
    valid_df = df[df["fold"] == 0].reset_index(drop=True)
    valid_transform = data.get_transforms(False, args.image_size)
    valid_dataset = data.AnimeFaceDataset(valid_df, args.image_dir, valid_transform)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    embeddings = get_image_embedding(model_, valid_dataloader, args.device)
    print(embeddings.shape, valid_df.shape)
    do_predict(
        embeddings,
        valid_df,
        args.min_threshold,
        args.max_threshold,
        args.step_threshold,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min_threshold", type=float, default=2.0)
    parser.add_argument("--max_threshold", type=float, default=4.0)
    parser.add_argument("--step_threshold", type=float, default=0.1)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("model_path", type=Path)
    parser = data.add_data_args(parser)
    args = parser.parse_args()
    prediction(args)


if __name__ == "__main__":
    main()
