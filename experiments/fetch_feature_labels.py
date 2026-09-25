"""Attach Neuronpedia explanations to the SAE features a circuit selected.

The geometry says the selected features point the same way; the labels
say what they mean. Together they are the interpretability claim, so
this keeps them in one file.

Neuronpedia is a public service and can rate-limit or simply not have an
explanation for a given feature, so a miss is recorded as null rather
than retried into the ground - a partially labelled figure is fine, a
stalled one is not.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://www.neuronpedia.org/api/feature"

#: Residual SAEs as Neuronpedia names them. 32k matches
#: llama_scope_lxr_8x: 8x expansion on d_model 4096 is 32768 features.
SOURCE = {
    "gemma": "{layer}-gemmascope-res-16k",
    "llama": "{layer}-llamascope-res-32k",
}
MODEL_SLUG = {"gemma": "gemma-2-2b", "llama": "llama3.1-8b"}


def fetch(model: str, layer: int, feature: int, timeout: float = 12.0):
    source = SOURCE[model].format(layer=layer)
    url = f"{BASE}/{MODEL_SLUG.get(model, model)}/{source}/{feature}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "circuitsteer-research/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    explanations = payload.get("explanations") or []
    if not explanations:
        return None
    text = explanations[0].get("description")
    return text.strip() if text else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path,
                        help="demo_geometry.json from the Colab export")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pause", type=float, default=0.12,
                        help="seconds between requests")
    args = parser.parse_args()

    data = json.loads(args.geometry.read_text())
    model = data["model"]

    wanted: set[tuple[int, int]] = set()
    for selection in data["selections"].values():
        for layer, features in selection["features"].items():
            for feature in features:
                wanted.add((int(layer), int(feature)))

    labels, hits = {}, 0
    for i, (layer, feature) in enumerate(sorted(wanted), 1):
        text = fetch(model, layer, feature)
        labels[f"L{layer}_{feature}"] = text
        hits += text is not None
        if i % 25 == 0:
            print(f"  {i}/{len(wanted)} ({hits} labelled)", flush=True)
        time.sleep(args.pause)

    out = args.out or args.geometry.with_name("feature_labels.json")
    out.write_text(json.dumps(labels, indent=1))
    print(f"labelled {hits}/{len(wanted)} features -> {out}")


if __name__ == "__main__":
    main()
