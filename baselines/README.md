# Baselines

Each baseline lives in its own subdirectory and is compared against
CircuitSteer under an identical protocol.

```
baselines/
  common.py        BaselineSteerer - the shared harness
  registry.py      name -> class, one line per baseline
  caa/             Contrastive Activation Addition (Rimsky et al. 2023)
  <next>/          one directory per baseline
```

## The contract

A baseline subclasses `BaselineSteerer` and implements exactly one
method:

```python
class MySteerer(BaselineSteerer):
    name = "MyMethod"
    requires_saes = False          # True only if the paper needs SAEs

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        self.steer_vecs = {layer: tensor, ...}
```

`BaselineSteerer` subclasses `CircuitSteer`, so hook placement, batched
generation, perplexity, scoring, the train/val/test split, the benign
control and the statistics are **the same code** for every method. A
baseline must not re-implement generation or scoring. The comparison is
only meaningful if the single difference is how the steering vector is
derived, not how it is applied or measured.

`steer_layers` defaults to the layers CircuitSteer's SAEs sit on, so no
method is advantaged purely by steering at a different depth. Override
it only if the paper is explicit about layer choice.

Register the class in `registry.py`, then it is runnable by name.

## Running a baseline

Identical invocation to the method, only `--method` and the output
directory change:

```bash
python main.py --method caa --models gemma llama \
  --dataset-size 500 --batch-size 32 --seed 42 \
  --fixed-lambda Emotion=-4 Jigsaw=-4 RTP=-3 Sycophancy=-4 \
  --output-dir results/baselines/caa/seed42
```

Repeat for seeds 42, 43, 44. Add `--release-saes` for Llama when the
baseline needs SAEs.

## Protocol that must not drift

| | Setting |
|---|---|
| Models | `gemma` (gemma-2-2b), `llama` (Llama-3.1-8B-Instruct) |
| Datasets | RTP, Jigsaw (civil_comments), Emotion, Sycophancy |
| Seeds | 42, 43, 44 |
| `--dataset-size` | 500 (200 train / 25 val / 25 test / 25 benign per class) |
| Coefficient grid | -4, -3, -2, -1 |
| Coefficient choice | one fixed value per (model, dataset), identical across seeds |
| Filtering | none; raw delta, ppl and norm_ppl are reported and filtered post hoc |
| Scorers | Detoxify `original` (RTP, Jigsaw), emotion-distilroberta (Emotion), normalised answer preference (Sycophancy) |

The fixed coefficients used for CircuitSteer, selected on **validation**
only, are:

| | Emotion | Jigsaw | RTP | Sycophancy |
|---|---|---|---|---|
| gemma | -4 | -4 | -3 | -4 |
| llama | -2 | -4 | -4 | -4 |

A baseline's own coefficient should be selected the same way - argmax
mean validation delta over the grid - and then held fixed across seeds.
Do not select on test.

## Reporting

Write to `results/baselines/<name>/`, then aggregate with the same
script the method uses:

```bash
python -m experiments.analyze results/baselines/caa --out results/baselines/caa/summary.csv
```

Quote the prompt-level statistics (`cohen_d`, `p_wilcoxon`, `n_pairs`)
rather than the seed-level CI: with 3 seeds the t-multiplier is 4.30 and
the interval is dominated by having three points. Use
`norm_ppl_median`, not `norm_ppl` - perplexity is heavy-tailed and the
mean-of-means is destroyed by single outliers (one sample reached
2.8e9).
