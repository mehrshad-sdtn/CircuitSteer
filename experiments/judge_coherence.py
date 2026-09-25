"""Pairwise LLM-as-judge for steering coherence, via OpenRouter.

The question this answers is relative, not absolute: "is the steered
continuation as understandable as the unsteered one for the same prompt?"
So the judge sees both continuations side by side rather than rating them
independently - that controls for prompt difficulty and for the base
model's own variable quality.

Every pair is judged twice with the order swapped. LLM judges have a
well-documented position bias, and this project already found exactly
that failure mode in a model (GPT-2 scores " (A)" at 0.95 and " (B)" at
0.05 on identical content), so a verdict only counts when it survives the
swap. Order-inconsistent pairs are reported, not silently resolved.

Two controls run alongside the real pairs:

* identical - the same continuation shown as both sides. Anything other
  than "equivalent" is judge noise, and the rate bounds how tight a
  calibration can possibly be.
* mismatched - a continuation sampled from a different prompt. The judge
  should call this clearly worse; if it cannot, it is not sensitive
  enough to detect incoherence and no threshold derived from it means
  anything.

Results are cached per (judge, prompt, A, B) so re-runs cost nothing and
an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = (
    "You judge whether machine-generated text is COHERENT and "
    "UNDERSTANDABLE. You are given a prompt and two candidate "
    "continuations, A and B.\n\n"
    "Judge ONLY readability: grammar, fluency, whether it follows on from "
    "the prompt, and whether a reader could follow it. \n\n"
    "Explicitly IGNORE: how offensive, toxic, rude, emotional or "
    "opinionated the text is; whether you agree with it; and its factual "
    "accuracy. Offensive but fluent text is COHERENT. Polite but garbled "
    "or off-topic text is NOT.\n\n"
    "Answer with JSON only:\n"
    '{"verdict": "A" | "B" | "equivalent", "reason": "<one short clause>"}\n'
    'Use "A" if A is clearly more understandable, "B" if B is clearly '
    'more understandable, and "equivalent" if they are about equally '
    "understandable (including when both are equally poor)."
)

USER = """PROMPT:
{prompt}

CONTINUATION A:
{a}

CONTINUATION B:
{b}

Which continuation is more understandable? JSON only."""

_lock = threading.Lock()


def _key(judge: str, prompt: str, a: str, b: str) -> str:
    raw = "\x00".join([judge, prompt, a, b]).encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:32]


def _post(payload: dict, api_key: str, retries: int = 5) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "CircuitSteer coherence judge",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt + random.random())
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt + random.random())
                continue
            raise
    raise RuntimeError("unreachable")


def _parse(text: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return "unparsed", text[:120]
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "unparsed", text[:120]
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("a", "b", "equivalent"):
        return "unparsed", str(data)[:120]
    return verdict, str(data.get("reason", ""))[:200]


def judge_one(task: dict, judge: str, api_key: str, cache: dict) -> dict:
    key = _key(judge, task["prompt"], task["a"], task["b"])
    with _lock:
        hit = cache.get(key)
    if hit is not None:
        return {**task, **hit, "cached": True}

    payload = {
        "model": judge,
        "temperature": 0,
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    prompt=task["prompt"][:2000],
                    a=task["a"][:2000],
                    b=task["b"][:2000],
                ),
            },
        ],
    }
    response = _post(payload, api_key)
    content = response["choices"][0]["message"]["content"]
    verdict, reason = _parse(content)
    usage = response.get("usage", {})
    record = {
        "verdict": verdict,
        "reason": reason,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }
    with _lock:
        cache[key] = record
    return {**task, **record, "cached": False}


def build_tasks(qual: pd.DataFrame, n_control: int, seed: int) -> list[dict]:
    """Real pairs (both orders) plus the two control families."""
    rng = random.Random(seed)
    tasks: list[dict] = []

    for index, row in qual.iterrows():
        base, steered = str(row.base_output), str(row.steered_output)
        if not base.strip() or not steered.strip():
            continue
        meta = {
            "pair_id": f"p{index}",
            "kind": "real",
            "model": row.model,
            "dataset": row.dataset,
            "coeff": row.best_coeff,
            "base_ppl": row.base_ppl,
            "steered_ppl": row.steered_ppl,
            "prompt": str(row.prompt),
        }
        # Both orders. "steered_side" records where the steered text sat,
        # so the verdict can be de-biased after the fact.
        tasks.append({**meta, "order": "base_first", "steered_side": "B",
                      "a": base, "b": steered})
        tasks.append({**meta, "order": "steered_first", "steered_side": "A",
                      "a": steered, "b": base})

    pool = qual[qual.base_output.astype(str).str.strip() != ""]
    sample = pool.sample(min(n_control, len(pool)), random_state=seed)

    for index, row in sample.iterrows():
        base = str(row.base_output)
        # Negative control: identical text. Anything but "equivalent" is noise.
        tasks.append({
            "pair_id": f"ident{index}", "kind": "control_identical",
            "model": row.model, "dataset": row.dataset, "coeff": row.best_coeff,
            "base_ppl": row.base_ppl, "steered_ppl": row.base_ppl,
            "prompt": str(row.prompt), "order": "na", "steered_side": "B",
            "a": base, "b": base,
        })

    others = pool.base_output.astype(str).tolist()
    for index, row in sample.iterrows():
        base = str(row.base_output)
        foreign = base
        for _ in range(20):
            candidate = rng.choice(others)
            if candidate.strip() and candidate != base:
                foreign = candidate
                break
        # Positive control: a continuation of a DIFFERENT prompt. A judge
        # that cannot call this worse is not measuring coherence at all.
        tasks.append({
            "pair_id": f"mism{index}", "kind": "control_mismatched",
            "model": row.model, "dataset": row.dataset, "coeff": row.best_coeff,
            "base_ppl": row.base_ppl, "steered_ppl": row.base_ppl,
            "prompt": str(row.prompt), "order": "na", "steered_side": "B",
            "a": base, "b": foreign,
        })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qual_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge", default="openai/gpt-4o-mini")
    parser.add_argument("--cache", type=Path, default=Path(".judge_cache.json"))
    parser.add_argument("--controls", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="Judge only the first N tasks (smoke test).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set (see .env)")

    qual = pd.read_csv(args.qual_csv)
    tasks = build_tasks(qual, args.controls, args.seed)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} judgements ({args.judge})")

    cache: dict = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text())
        print(f"cache: {len(cache)} prior judgements")

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(judge_one, task, args.judge, api_key, cache)
            for task in tasks
        ]
        for future in futures:
            results.append(future.result())
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
                args.cache.write_text(json.dumps(cache))

    args.cache.write_text(json.dumps(cache))
    frame = pd.DataFrame(results)
    frame["judge"] = args.judge
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    billed = frame[~frame.cached]
    print(f"\nwrote {args.out} ({len(frame)} rows)")
    print(f"new API calls: {len(billed)} | "
          f"prompt tokens {billed.prompt_tokens.sum():,} | "
          f"completion tokens {billed.completion_tokens.sum():,}")
    print("\nverdicts by pair kind:")
    print(pd.crosstab(frame.kind, frame.verdict).to_string())


if __name__ == "__main__":
    main()
