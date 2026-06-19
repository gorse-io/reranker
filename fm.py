import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Factorization Machines benchmark")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--context_ratio", type=float, default=0.8)
    parser.add_argument("--dataset", type=str, default="ml-100k")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    return parser.parse_args()


def safe_filename_part(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    safe_value = safe_value.strip("._")
    return safe_value or "unknown"


def metrics_output_path(dataset_name: str, model_name: str) -> Path:
    filename = (
        f"{safe_filename_part(dataset_name)}_"
        f"{safe_filename_part(model_name)}.csv"
    )
    return Path(filename)


def save_user_metrics(metrics: pd.DataFrame, dataset_name: str, model_name: str) -> Path:
    output_path = metrics_output_path(dataset_name, model_name)
    metrics[["user_id", "auc", "positive", "negative"]].to_csv(
        output_path, index=False
    )
    return output_path


@dataclass(frozen=True)
class Dataset:
    items: dict[str, dict]
    users: dict[str, dict]
    feedback: pd.DataFrame
    user_splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]]


def load_jsonl(path: Path, id_column: str) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[str(record[id_column])] = record
    return records


def split_feedback_by_user(
    feedback: pd.DataFrame, context_ratio: float
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    user_splits = {}
    for user_id, group in feedback.groupby("user_id", sort=True):
        sorted_group = group.sort_values("timestamp", ascending=True)
        train_size = int(len(sorted_group) * context_ratio)
        train = sorted_group.iloc[:train_size]
        test = sorted_group.iloc[train_size:]
        labels = [int(label) for label in test["label"].to_list()]
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue
        if not train.empty and not test.empty:
            user_splits[str(user_id)] = (train, test)
    return user_splits


def load_dataset(data_dir: Path, context_ratio: float, dataset_name: str) -> Dataset:
    feedback_columns = ["user_id", "item_id", "label", "timestamp"]
    items = load_jsonl(data_dir / "data" / dataset_name / "items.jsonl", "item_id")
    users = load_jsonl(data_dir / "data" / dataset_name / "users.jsonl", "user_id")
    feedback = pd.read_csv(
        data_dir / "data" / dataset_name / "feedback.csv",
        header=None,
        names=feedback_columns,
    )
    user_splits = split_feedback_by_user(feedback, context_ratio)
    return Dataset(items, users, feedback, user_splits)


def auc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_list()
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


class FactorizationMachine(nn.Module):
    def __init__(self, num_features: int, num_factors: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(num_features, 1)
        self.factors = nn.Embedding(num_features, num_factors)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.factors.weight, std=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        linear = self.linear(features).sum(dim=1).squeeze(-1)
        embeddings = self.factors(features)
        summed = embeddings.sum(dim=1)
        squared_sum = summed.square()
        sum_squared = embeddings.square().sum(dim=1)
        interactions = 0.5 * (squared_sum - sum_squared).sum(dim=1)
        return self.bias + linear + interactions


def build_id_maps(feedback: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    user_ids = sorted(str(user_id) for user_id in feedback["user_id"].unique())
    item_ids = sorted(str(item_id) for item_id in feedback["item_id"].unique())
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    return user_to_index, item_to_index


def encode_feedback(
    feedback: pd.DataFrame,
    user_to_index: dict[str, int],
    item_to_index: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    user_count = len(user_to_index)
    features = []
    labels = []
    for row in feedback.itertuples(index=False):
        user_id = str(row.user_id)
        item_id = str(row.item_id)
        if user_id not in user_to_index or item_id not in item_to_index:
            continue
        features.append([user_to_index[user_id], user_count + item_to_index[item_id]])
        labels.append(float(row.label))
    if not features:
        raise ValueError("No feedback rows could be encoded.")
    return (
        torch.tensor(features, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def train_model(
    model: FactorizationMachine,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
) -> None:
    model.to(device)
    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    progress = tqdm(range(1, epochs + 1), unit="epoch")
    for epoch in progress:
        total_loss = 0.0
        total_count = 0
        model.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            count = len(batch_labels)
            total_loss += float(loss.detach().cpu()) * count
            total_count += count

        progress.set_postfix_str(f"loss: {total_loss / total_count:.6f}")


@torch.inference_mode()
def score_feedback(
    model: FactorizationMachine,
    feedback: pd.DataFrame,
    user_to_index: dict[str, int],
    item_to_index: dict[str, int],
    device: torch.device,
) -> list[float]:
    features, _ = encode_feedback(feedback, user_to_index, item_to_index)
    model.eval()
    logits = model(features.to(device)).detach().cpu()
    return logits.tolist()


def run_benchmark(
    dataset: Dataset,
    factors: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    num_workers: int,
) -> tuple[float, pd.DataFrame]:
    user_to_index, item_to_index = build_id_maps(dataset.feedback)
    train_feedback = pd.concat(
        [train for train, _ in dataset.user_splits.values()],
        ignore_index=True,
    )
    train_features, train_labels = encode_feedback(
        train_feedback, user_to_index, item_to_index
    )

    model = FactorizationMachine(
        num_features=len(user_to_index) + len(item_to_index),
        num_factors=factors,
    )
    train_model(
        model,
        train_features,
        train_labels,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
    )

    user_metrics = []
    total_weighted_auc = 0.0
    total_weight = 0

    progress = tqdm(
        dataset.user_splits.items(),
        total=len(dataset.user_splits),
        unit="user",
    )
    for user_id, (_, test) in progress:
        labels = [int(label) for label in test["label"].to_list()]
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue

        scores = score_feedback(model, test, user_to_index, item_to_index, device)
        current_auc = auc_score(labels, scores)
        if current_auc is None:
            continue

        weight = len(labels)
        total_weighted_auc += current_auc * weight
        total_weight += weight
        running_gauc = total_weighted_auc / total_weight
        user_metrics.append(
            {
                "user_id": user_id,
                "auc": current_auc,
                "weight": weight,
                "positive": sum(labels),
                "negative": len(labels) - sum(labels),
            }
        )
        progress.set_postfix_str(f"GAUC: {running_gauc:.6f}")

    metrics = pd.DataFrame(user_metrics)
    if metrics.empty:
        raise ValueError("No users have both positive and negative test samples.")
    final_gauc = total_weighted_auc / total_weight
    return float(final_gauc), metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(
        snapshot_download(
            repo_id="gorse-io/gorse-reranker-benchmark", repo_type="dataset"
        )
    )
    dataset = load_dataset(data_dir, args.context_ratio, args.dataset)

    print(f"data: \t\t{data_dir}")
    print("model: \t\tfactorization_machines")
    print(f"device: \t{device}")
    print(f"#items: \t{len(dataset.items):,}")
    print(f"#users: \t{len(dataset.users):,}")
    print(f"#feedback: \t{len(dataset.feedback):,}")

    gauc, user_metrics = run_benchmark(
        dataset,
        factors=args.factors,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        num_workers=args.num_workers,
    )
    model_name = f"fm_f{args.factors}_e{args.epochs}"
    metrics_path = save_user_metrics(user_metrics, args.dataset, model_name)
    print(f"GAUC: {gauc:.6f}")
    print(f"Result:\t{metrics_path}")


if __name__ == "__main__":
    main()
