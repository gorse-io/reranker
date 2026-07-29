import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DEFAULT_INPUT = Path("ml-1m_Qwen_Qwen3-Reranker-4B.csv")
MIN_BUCKET_USERS = 50


@dataclass(frozen=True)
class UserMetric:
    auc: float
    weight: int


@dataclass(frozen=True)
class GaucBucket:
    label: str
    lower: int
    upper: int
    users: int
    weight: int
    gauc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped GAUC by each user's positive + negative sample count."
        )
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input user metrics CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_csv",
        type=Path,
        default=None,
        help="Input user metrics CSV. Kept for backward compatibility.",
    )
    parser.add_argument(
        "-n",
        "--group-size",
        type=int,
        default=50,
        help="Bucket width for positive + negative count, for example 1-n.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: same name as CSV with .png suffix.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window after saving.",
    )
    args = parser.parse_args()
    if args.input_csv is not None:
        args.csv = args.input_csv
    if args.output is None:
        args.output = args.csv.with_suffix(".png")
    return args


def read_user_metrics(path: Path) -> list[UserMetric]:
    metrics: list[UserMetric] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"auc", "positive", "negative"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                auc = float(row["auc"])
                positive = int(row["positive"])
                negative = int(row["negative"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value at row {row_number}") from exc

            weight = positive + negative
            if weight <= 0 or not math.isfinite(auc):
                continue
            metrics.append(UserMetric(auc=auc, weight=weight))

    if not metrics:
        raise ValueError(f"No valid user metrics found in {path}")
    return metrics


def group_gauc(metrics: list[UserMetric], group_size: int) -> list[GaucBucket]:
    if group_size <= 0:
        raise ValueError("--group-size/-n must be a positive integer")

    grouped: dict[int, dict[str, float | int]] = {}
    for metric in metrics:
        bucket_index = (metric.weight - 1) // group_size
        bucket = grouped.setdefault(
            bucket_index,
            {"weighted_auc": 0.0, "weight": 0, "users": 0},
        )
        bucket["weighted_auc"] += metric.auc * metric.weight
        bucket["weight"] += metric.weight
        bucket["users"] += 1

    buckets: list[GaucBucket] = []
    for bucket_index in sorted(grouped):
        lower = bucket_index * group_size + 1
        upper = (bucket_index + 1) * group_size
        values = grouped[bucket_index]
        users = int(values["users"])
        if users < MIN_BUCKET_USERS:
            continue
        weight = int(values["weight"])
        buckets.append(
            GaucBucket(
                label=f"{lower}-{upper}",
                lower=lower,
                upper=upper,
                users=users,
                weight=weight,
                gauc=float(values["weighted_auc"]) / weight,
            )
        )
    if not buckets:
        raise ValueError(
            f"No buckets have at least {MIN_BUCKET_USERS} users after grouping."
        )
    return buckets


def draw_histogram(
    ax: plt.Axes,
    count_ax: plt.Axes,
    metrics: list[UserMetric],
    group_size: int,
) -> None:
    buckets = group_gauc(metrics, group_size)
    labels = [bucket.label for bucket in buckets]
    gauc_values = [bucket.gauc for bucket in buckets]
    user_counts = [bucket.users for bucket in buckets]
    positions = list(range(len(buckets)))

    ax.clear()
    count_ax.clear()
    bars = ax.bar(
        positions,
        gauc_values,
        color="#4c78a8",
        edgecolor="#2f4b63",
        label="GAUC",
    )
    (count_line,) = count_ax.plot(
        positions,
        user_counts,
        color="#f58518",
        marker="o",
        linewidth=2,
        markersize=4,
        label="User Count",
    )
    ax.set_title("GAUC by user sample count")
    ax.set_xlabel("Sample Count")
    ax.set_ylabel("GAUC")
    ax.tick_params(axis="y", colors="#4c78a8")
    ax.yaxis.label.set_color("#4c78a8")
    y_min = min(gauc_values)
    y_max = max(gauc_values)
    if math.isclose(y_min, y_max):
        padding = 0.01
        y_min = max(0.0, y_min - padding)
        y_max = min(1.0, y_max + padding)
    else:
        padding = (y_max - y_min) * 0.08
        y_min = max(0.0, y_min - padding)
        y_max = min(1.0, y_max + padding)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    count_ax.set_ylabel("User Count")
    count_ax.yaxis.set_label_position("right")
    count_ax.yaxis.tick_right()
    count_ax.tick_params(axis="y", colors="#f58518")
    count_ax.yaxis.label.set_color("#f58518")
    count_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    count_y_min = min(user_counts)
    count_y_max = max(user_counts)
    if count_y_min == count_y_max:
        padding = max(1, int(count_y_min * 0.05))
        count_y_min = max(0, count_y_min - padding)
        count_y_max = count_y_max + padding
    else:
        padding = (count_y_max - count_y_min) * 0.05
        count_y_max += padding
    count_ax.set_ylim(count_y_min, count_y_max)

    tick_step = max(1, math.ceil(len(labels) / 30))
    tick_positions = positions[::tick_step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [labels[index] for index in tick_positions],
        rotation=45,
        ha="right",
    )

    for bar, bucket in zip(bars, buckets):
        if len(buckets) > 35:
            continue
        label_y = bucket.gauc
        label_va = "bottom"
        if math.isclose(bucket.gauc, y_max):
            label_y = bucket.gauc - (y_max - y_min) * 0.02
            label_va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{bucket.gauc:.3f}",
            ha="center",
            va=label_va,
            fontsize=8,
        )
    ax.legend([bars, count_line], ["GAUC", "User Count"], loc="upper right")


def print_bucket_summary(buckets: list[GaucBucket]) -> None:
    print("bucket,users,weight,gauc")
    for bucket in buckets:
        print(f"{bucket.label},{bucket.users},{bucket.weight},{bucket.gauc:.6f}")


def main() -> None:
    args = parse_args()
    metrics = read_user_metrics(args.csv)
    buckets = group_gauc(metrics, args.group_size)

    figsize = (max(10, min(28, len(buckets) * 0.35)), 6)

    fig, ax = plt.subplots(figsize=figsize)
    count_ax = ax.twinx()
    draw_histogram(ax, count_ax, metrics, args.group_size)
    fig.tight_layout()

    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved plot: {args.output}")
    print_bucket_summary(buckets)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
