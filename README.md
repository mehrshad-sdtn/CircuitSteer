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
python main.py --models gemma --tasks Sycophancy --dataset-size 4 \
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

## Tests

The lightweight tests mock model and SAE loading:

```bash
python -m unittest discover -s tests -v
```
