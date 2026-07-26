from __future__ import annotations

import random
from collections.abc import Iterable

DatasetItem = str | tuple[str, str]
DatasetSplit = dict[str, list[DatasetItem]]


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def text_of(item: DatasetItem) -> str:
    return item[0] if isinstance(item, tuple) else item


def _split(
    toxic: list[str],
    benign: list[str],
    fraction: float = 0.8,
    seed: int = 42,
) -> DatasetSplit:
    rng = random.Random(seed)
    indices = list(range(min(len(toxic), len(benign))))
    rng.shuffle(indices)
    toxic = [toxic[index] for index in indices]
    benign = [benign[index] for index in indices]
    cutoff = int(len(toxic) * fraction)
    return {
        "train_toxic": toxic[:cutoff],
        "train_benign": benign[:cutoff],
        "test_prompts": toxic[cutoff:],
    }


def load_rtp(n: int = 500, seed: int = 42) -> DatasetSplit:
    n_each = n // 2
    dataset = _load_dataset(
        "allenai/real-toxicity-prompts",
        split="train",
        streaming=True,
    )
    toxic: list[str] = []
    benign: list[str] = []
    for row in dataset:
        score = row["prompt"]["toxicity"]
        if score is None:
            continue
        text = row["prompt"]["text"]
        if score > 0.85 and len(toxic) < n_each:
            toxic.append(text)
        if score < 0.10 and len(benign) < n_each:
            benign.append(text)
        if len(toxic) >= n_each and len(benign) >= n_each:
            break
    print(f"RTP: {len(toxic)} toxic, {len(benign)} benign")
    return _split(toxic, benign, seed=seed)


def load_jigsaw(n: int = 500, seed: int = 42) -> DatasetSplit:
    n_each = n // 2
    dataset = _load_dataset(
        "google/civil_comments",
        split="train",
        streaming=True,
    )
    toxic: list[str] = []
    benign: list[str] = []
    for row in dataset:
        text = (row.get("text") or "").strip()[:300]
        score = float(row.get("toxicity", 0.0))
        if len(text) < 20:
            continue
        if score >= 0.5 and len(toxic) < n_each:
            toxic.append(text)
        if score == 0.0 and len(benign) < n_each:
            benign.append(text)
        if len(toxic) >= n_each and len(benign) >= n_each:
            break
    print(f"Jigsaw: {len(toxic)} toxic, {len(benign)} benign")
    return _split(toxic, benign, seed=seed)


def load_emotion(n: int = 500, seed: int = 42) -> DatasetSplit:
    n_each = n // 2
    dataset = _load_dataset("dair-ai/emotion", split="train")
    toxic: list[str] = []
    benign: list[str] = []
    for row in dataset:
        text = row["text"].strip()
        label = row["label"]
        if label == 3 and len(toxic) < n_each:
            toxic.append(text)
        if label == 1 and len(benign) < n_each:
            benign.append(text)
        if len(toxic) >= n_each and len(benign) >= n_each:
            break
    print(f"Emotion: {len(toxic)} anger, {len(benign)} joy")
    return _split(toxic, benign, seed=seed)


def load_sycophancy(n: int = 500, seed: int = 42) -> DatasetSplit:
    del seed  # The source dataset has a fixed order, matching the paper notebook.
    n_each = n // 2
    n_train = int(n_each * 0.8)
    n_test = n_each - n_train
    dataset = _load_dataset(
        "Anthropic/model-written-evals",
        data_files="sycophancy/sycophancy_on_nlp_survey.jsonl",
        split="train",
    )
    train_toxic: list[DatasetItem] = []
    train_benign: list[DatasetItem] = []
    test_items: list[DatasetItem] = []
    for row in dataset:
        question = row["question"]
        sycophantic = row["answer_matching_behavior"]
        honest = row["answer_not_matching_behavior"]
        if len(train_toxic) < n_train:
            train_toxic.append(question + sycophantic)
            train_benign.append(question + honest)
        elif len(test_items) < n_test:
            test_items.append((question, sycophantic))
        if len(train_toxic) >= n_train and len(test_items) >= n_test:
            break
    print(
        f"Sycophancy: {len(train_toxic)} train pairs, "
        f"{len(test_items)} test"
    )
    return {
        "train_toxic": train_toxic,
        "train_benign": train_benign,
        "test_prompts": test_items,
    }


LOADERS = {
    "RTP": load_rtp,
    "Jigsaw": load_jigsaw,
    "Emotion": load_emotion,
    "Sycophancy": load_sycophancy,
}


def make_datasets(
    labels: Iterable[str] | None = None,
    n: int = 500,
    seed: int = 42,
) -> dict[str, DatasetSplit]:
    selected = list(labels) if labels is not None else list(LOADERS)
    unknown = set(selected) - set(LOADERS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return {label: LOADERS[label](n=n, seed=seed) for label in selected}
