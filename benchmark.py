import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import jinja2
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from vllm import LLM


def parse_args():
    parser = argparse.ArgumentParser(description="Gorse Reranker Benchmark Config")
    parser.add_argument("--context_size", type=int, default=50)
    parser.add_argument("--context_ratio", type=float, default=0.8)
    parser.add_argument("--dataset", type=str, default="ml-100k")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Reranker-4B",
        help="Hugging Face model repo id or a local model directory.",
    )
    parser.add_argument("--max_model_len", type=int, default=None)
    return parser.parse_args()


def resolve_model_dir(model: str) -> Path:
    model_path = Path(model).expanduser()
    if model_path.exists():
        if not model_path.is_dir():
            raise ValueError(
                f"--model points to a local path but is not a directory: {model}"
            )
        return model_path
    return Path(snapshot_download(repo_id=model, repo_type="model"))


def safe_filename_part(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    safe_value = safe_value.strip("._")
    return safe_value or "unknown"


def metrics_output_path(dataset_name: str, model_name: str) -> Path:
    model_path = Path(model_name).expanduser()
    model_filename_part = model_path.name if model_path.exists() else model_name
    filename = (
        f"{safe_filename_part(dataset_name)}_"
        f"{safe_filename_part(model_filename_part)}.csv"
    )
    return Path(filename)


def save_user_metrics(metrics: pd.DataFrame, dataset_name: str, model_name: str) -> Path:
    output_path = metrics_output_path(dataset_name, model_name)
    metrics[["user_id", "auc", "positive", "negative"]].to_csv(
        output_path, index=False
    )
    return output_path


args = parse_args()
data_dir = snapshot_download(
    repo_id="gorse-io/gorse-reranker-benchmark", repo_type="dataset"
)
model_dir = resolve_model_dir(args.model)


@dataclass(frozen=True)
class Dataset:
    items: dict[str, dict]
    users: dict[str, dict]
    feedback: pd.DataFrame
    instruction: str
    query_template: jinja2.Template
    document_template: jinja2.Template
    item_docs: dict[str, str]
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
    # Load Data
    FEEDBACK_COLUMNS = ["user_id", "item_id", "label", "timestamp"]
    items = load_jsonl(data_dir / "data" / dataset_name / "items.jsonl", "item_id")
    users = load_jsonl(data_dir / "data" / dataset_name / "users.jsonl", "user_id")
    feedback = pd.read_csv(
        data_dir / "data" / dataset_name / "feedback.csv",
        header=None,
        names=FEEDBACK_COLUMNS,
    )

    # Load Templates
    instruct_path = data_dir / "prompt" / dataset_name / "instruct.txt"
    query_path = data_dir / "prompt" / dataset_name / "query.jinja2"
    document_path = data_dir / "prompt" / dataset_name / "document.jinja2"

    instruction = instruct_path.read_text(encoding="utf-8").strip()
    query_template = jinja2.Template(query_path.read_text(encoding="utf-8"))
    document_template = jinja2.Template(document_path.read_text(encoding="utf-8"))

    # Process Data
    item_docs = {
        str(item_id): document_template.render(
            item=item, item_id=item["item_id"]
        ).strip()
        for item_id, item in items.items()
    }
    user_splits = split_feedback_by_user(feedback, context_ratio)
    return Dataset(
        items,
        users,
        feedback,
        instruction,
        query_template,
        document_template,
        item_docs,
        user_splits,
    )


SYSTEM_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def render_query(
    query_template: jinja2.Template,
    train_feedback: pd.DataFrame,
    item_docs: dict[str, str],
    instruction: str,
    max_feedback: int | None = 50,
    model_name: str = "",
) -> str:
    positive_feedback = train_feedback[train_feedback["label"] == 1].sort_values(
        "timestamp", ascending=True
    )
    if max_feedback is not None:
        positive_feedback = positive_feedback.tail(max_feedback)

    positive_history = [
        (str(row.item_id), int(row.label), int(row.timestamp))
        for row in positive_feedback.itertuples(index=False)
    ]
    rendered_query = query_template.render(
        prompt_history=positive_history, item_docs=item_docs
    ).strip()

    model_name_lower = model_name.lower()
    if (
        "bge-reranker" in model_name_lower
        or "nemotron" in model_name_lower
        or "mxbai" in model_name_lower
    ):
        return f"{instruction}\n{rendered_query}"
    else:
        return f"{SYSTEM_PREFIX}<Instruct>: {instruction}\n<Query>: {rendered_query}\n"


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


def run_benchmark(
    dataset: Dataset,
    model_dir: str,
    context_size: int | None = 50,
    max_model_len: int | None = None,
    model_name: str = "",
) -> tuple[float, pd.DataFrame]:

    model_name_lower = model_name.lower()

    if "bge-reranker" in model_name_lower:
        hf_overrides = {
            "architectures": ["GemmaForSequenceClassification"],
            "classifier_from_token": ["Yes"],
            "method": "no_post_processing",
        }
    elif "nemotron" in model_name_lower:
        hf_overrides = {
            "architectures": ["LlamaBidirectionalForSequenceClassification"]
        }
    elif "mxbai" in model_name_lower:
        hf_overrides = {
            "architectures": ["Qwen2ForSequenceClassification"],
            "classifier_from_token": ["0", "1"],
            "method": "from_2_way_softmax",
        }
    else:
        hf_overrides = {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        }

    # Initialize vLLM with sequence classification overrides
    model = LLM(
        model=str(model_dir),
        runner="pooling",
        hf_overrides=hf_overrides,
        dtype=torch.bfloat16,
        enforce_eager=True,  # Recommended for Colab to save memory
        trust_remote_code=True,
        max_model_len=max_model_len,
    )

    user_metrics = []
    total_weighted_auc = 0.0
    total_weight = 0

    for user_id, (train, test) in dataset.user_splits.items():
        test = test[test["item_id"].astype(str).isin(dataset.item_docs)]
        if test.empty:
            continue
        labels = [int(label) for label in test["label"].to_list()]
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue

        query = render_query(
            dataset.query_template,
            train,
            dataset.item_docs,
            dataset.instruction,
            max_feedback=context_size,
            model_name=model_name,
        )

        if "bge-reranker" in model_name_lower:
            # BGE format
            documents = []
            for item_id in test["item_id"].to_list():
                doc_text = dataset.item_docs[str(item_id)]
                prompt = f"A: {query}\nB: {doc_text}\nGiven a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
                documents.append(prompt)
            queries = [""] * len(
                documents
            )  # BGE prompt already has both, pass empty strings for query part
        elif "nemotron" in model_name_lower:
            # Nemotron format
            documents = []
            for item_id in test["item_id"].to_list():
                doc_text = dataset.item_docs[str(item_id)]
                prompt = f"question:{query} \n \n passage:{doc_text}"
                documents.append(prompt)
            queries = [""] * len(documents)
        elif "mxbai" in model_name_lower:
            # Mxbai format
            documents = []
            for item_id in test["item_id"].to_list():
                doc_text = dataset.item_docs[str(item_id)]
                prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nquery: {query}\ndocument: {doc_text}\nYou are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:<|im_end|>\n<|im_start|>assistant\n"
                documents.append(prompt)
            queries = [""] * len(documents)
        else:
            # Qwen format
            documents = [
                f"<Document>: {dataset.item_docs[str(item_id)]}{ASSISTANT_SUFFIX}"
                for item_id in test["item_id"].to_list()
            ]
            queries = [query] * len(documents)

        if (
            "bge-reranker" in model_name_lower
            or "nemotron" in model_name_lower
            or "mxbai" in model_name_lower
        ):
            outputs = model.score(
                documents, queries
            )  # or just texts if queries are empty
        else:
            outputs = model.score(queries, documents)

        scores = [output.outputs.score for output in outputs]

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
        print(
            f"{len(user_metrics)}/{len(dataset.user_splits)}\tuser id: {user_id}, AUC={current_auc:.6f}, current GAUC={running_gauc:.6f}"
        )

    metrics = pd.DataFrame(user_metrics)
    if metrics.empty:
        raise ValueError("No users have both positive and negative test samples.")
    final_gauc = total_weighted_auc / total_weight
    return float(final_gauc), metrics


dataset = load_dataset(Path(data_dir), args.context_ratio, args.dataset)
print(f"data: \t\t{data_dir}")
print(f"model: \t\t{model_dir}")
print(f"#items: \t{len(dataset.items):,}")
print(f"#users: \t{len(dataset.users):,}")
print(f"#feedback: \t{len(dataset.feedback):,}")
gauc, user_metrics = run_benchmark(
    dataset,
    model_dir,
    context_size=args.context_size,
    max_model_len=args.max_model_len,
    model_name=args.model,
)
metrics_path = save_user_metrics(user_metrics, args.dataset, args.model)
print(f"GAUC: {gauc:.6f}")
print(f"Result:\t{metrics_path}")
