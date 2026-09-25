# CircuitSteer

Code for SAE-based multi-layer circuit steering experiments.

## Setup

Python 3.10+ and a CUDA GPU are recommended.

```bash
git clone https://github.com/mehrshad-sdtn/CircuitSteer.git
cd CircuitSteer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gemma and Llama require accepted Hugging Face model licenses and an access
token:

```bash
export HF_TOKEN="your_token"
```

## Run

Run the full main experiment:

```bash
python main.py
```

Run a smaller check first:

```bash
python main.py --models gemma --tasks Sycophancy --dataset-size 20 \
  --coefficients -3 --qualitative-samples 0
```

Additional paper experiments:

```bash
python -m experiments.sensitivity
python -m experiments.mmlu
python -m experiments.gsm8k
python -m experiments.layer_count
```

Use `--help` on any command to see configurable models, sample sizes, and
output paths. Results are written to `results/` by default.

## Evaluation

Each dataset is split into `train` (80%, used for circuit discovery) and a
held-out 20% that is halved into `val_prompts` and `test_prompts`. The
steering coefficient is chosen on validation and reported on test, so the
headline delta is not read off the prompts it was tuned on. Runs too small
to produce a validation split fall back to selecting on test and say so.

Metrics per coefficient:

| Column | Meaning |
| --- | --- |
| `delta` | mean score drop, base minus steered (higher is better) |
| `delta_sem` | standard error of `delta`; use this for error bars, not `delta_std` |
| `delta_max` | same, using the per-prompt maximum (RealToxicityPrompts style) |
| `norm_ppl` | steered perplexity / base perplexity; fluency guard |
| `degenerate_frac` | fraction of steered generations that came back empty |
| `valid_frac` | fraction of prompts that produced a scoreable pair |

Empty continuations score `NaN`, never `0.0`: scoring them as clean text
would make a coefficient that collapses generation look like the best
result. `split=test_benign` rows apply the same steering to benign prompts
as a control — targeted steering should barely move them.

Scorers: RTP and Jigsaw use Detoxify `original`, Emotion uses
`j-hartmann/emotion-english-distilroberta-base`, and Sycophancy uses the
model's own normalised preference between the matching and non-matching
answers, `P(sycophantic) / (P(sycophantic) + P(honest))`. Note that base
models carry a strong preference for the letter `(A)`, so the absolute
sycophancy score reflects position bias; the reported `delta` is a paired
difference over the same items, where that bias cancels.

`--samples-per-prompt` draws several continuations per prompt at
temperature 1.0 (the RealToxicityPrompts protocol uses 25). The default of
1 reproduces the original single-sample run.

## Tests

The lightweight tests mock model and SAE loading:

```bash
python -m unittest discover -s tests -v
```
