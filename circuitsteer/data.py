from __future__ import annotations

import random
from collections.abc import Iterable

# A plain prompt, or a Sycophancy item:
# (question, sycophantic_answer, honest_answer). Both answers are kept:
# the sycophancy metric is contrastive and cannot be computed without
# the answer the model is supposed to prefer instead.
DatasetItem = str | tuple[str, str, str]
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
    val_fraction: float = 0.5,
    seed: int = 42,
) -> DatasetSplit:
    """Split into train / validation / test.

    The training portion is unchanged from the original 80% cut, so circuit
    discovery is unaffected. The held-out 20% is halved into `val_prompts`
    (used to pick the steering coefficient) and `test_prompts` (used only to
    report), so the reported delta is not read off the same prompts the
    coefficient was selected on. `test_benign` is the control condition:
    steering that merely suppresses fluency moves it as much as it moves
    `test_prompts`, while genuinely targeted steering leaves it alone.
    """
    rng = random.Random(seed)
    indices = list(range(min(len(toxic), len(benign))))
    rng.shuffle(indices)
    toxic = [toxic[index] for index in indices]
    benign = [benign[index] for index in indices]
    cutoff = int(len(toxic) * fraction)
    held_toxic = toxic[cutoff:]
    held_benign = benign[cutoff:]
    val_cutoff = int(len(held_toxic) * val_fraction)
    return {
        "train_toxic": toxic[:cutoff],
        "train_benign": benign[:cutoff],
        "val_prompts": held_toxic[:val_cutoff],
        "test_prompts": held_toxic[val_cutoff:],
        "test_benign": held_benign[val_cutoff:],
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
    # The source file has a fixed order. Honouring the seed shuffles which
    # items land in train/val/test, so repeated runs are genuinely
    # different draws; without it every seed produces an identical split
    # and an identical (deterministic, teacher-forced) score, leaving no
    # seed-level variance to put a confidence interval on.
    n_each = n // 2
    n_train = int(n_each * 0.8)
    n_held = n_each - n_train
    n_val = n_held // 2
    dataset = _load_dataset(
        "Anthropic/model-written-evals",
        data_files="sycophancy/sycophancy_on_nlp_survey.jsonl",
        split="train",
    )
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    train_toxic: list[DatasetItem] = []
    train_benign: list[DatasetItem] = []
    held_items: list[DatasetItem] = []
    for position in order:
        row = dataset[position]
        question = row["question"]
        sycophantic = row["answer_matching_behavior"]
        honest = row["answer_not_matching_behavior"]
        if len(train_toxic) < n_train:
            train_toxic.append(question + sycophantic)
            train_benign.append(question + honest)
        elif len(held_items) < n_held:
            # Both answers are kept: the score is P(sycophantic) normalised
            # against P(honest), which needs the pair.
            held_items.append((question, sycophantic, honest))
        if len(train_toxic) >= n_train and len(held_items) >= n_held:
            break
    print(
        f"Sycophancy: {len(train_toxic)} train pairs, "
        f"{n_val} val, {len(held_items) - n_val} test"
    )
    return {
        "train_toxic": train_toxic,
        "train_benign": train_benign,
        "val_prompts": held_items[:n_val],
        "test_prompts": held_items[n_val:],
        # The sycophancy score is already contrastive (it compares the two
        # answers on the same question), so it carries its own control and
        # needs no separate benign slice.
        "test_benign": [],
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
