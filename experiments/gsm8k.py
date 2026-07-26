from __future__ import annotations

import argparse
import random
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

from circuitsteer import CircuitSteer
from circuitsteer.data import LOADERS, make_datasets, text_of
from circuitsteer.io import append_row, read_rows, write_rows
from experiments.common import (
    add_circuit_arguments,
    config_from_args,
    release_model,
)

ITEM_FIELDS = [
    "model",
    "behavior",
    "best_lambda",
    "idx",
    "gold",
    "base_pred",
    "steered_pred",
    "base_ok",
    "steered_ok",
]
SUMMARY_FIELDS = [
    "model",
    "behavior",
    "best_lambda",
    "n",
    "base_acc",
    "steered_acc",
    "delta_acc",
]

FEW_SHOT = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she "
        "sold half as many clips in May. How many clips did she sell "
        "altogether in April and May?",
        "In April, Natalia sold 48 clips. In May, she sold half as many, "
        "so she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 "
        "clips. The answer is 72.",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did "
        "50 minutes of babysitting. How much did she earn?",
        "Weng earns 12 / 60 = $0.2 per minute. For 50 minutes she earned "
        "0.2 * 50 = $10. The answer is 10.",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty "
        "has only half of the money she needs. Her parents decided to give "
        "her $15 for that purpose, and her grandparents twice as much as "
        "her parents. How much more money does Betty need to buy the wallet?",
        "Betty has half of $100, which is 100 / 2 = $50. Her parents give "
        "$15. Her grandparents give twice as much, 15 * 2 = $30. Now Betty "
        "has 50 + 15 + 30 = $95. She still needs 100 - 95 = $5. The answer "
        "is 5.",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. "
        "How many pages does he write a year?",
        "Each time James writes 3 * 2 = 6 pages. He does this twice a week, "
        "so 6 * 2 = 12 pages per week. In a year that is 12 * 52 = 624 "
        "pages. The answer is 624.",
    ),
    (
        "Mark has a garden with flowers. He planted plants of three "
        "different colors in it. Ten of them are yellow, and there are 80% "
        "more of those in purple. There are only 25% as many green flowers "
        "as there are yellow and purple flowers. How many flowers does Mark "
        "have in his garden?",
        "There are 10 yellow flowers. Purple flowers are 80% more than "
        "yellow: 10 + 10 * 0.8 = 18. Yellow plus purple is 10 + 18 = 28. "
        "Green flowers are 25% of that: 28 * 0.25 = 7. In total Mark has "
        "10 + 18 + 7 = 35 flowers. The answer is 35.",
    ),
    (
        "Albert is wondering how much pizza he can eat in one day. He buys "
        "2 large pizzas and 2 small pizzas. A large pizza has 16 slices and "
        "a small pizza has 8 slices. If he eats it all, how many pieces "
        "does he eat that day?",
        "The large pizzas have 2 * 16 = 32 slices. The small pizzas have "
        "2 * 8 = 16 slices. Altogether Albert eats 32 + 16 = 48 slices. "
        "The answer is 48.",
    ),
    (
        "Ken created a care package to send to his brother, who was away "
        "at boarding school. Ken placed a box on a scale, and then he poured "
        "into the box enough jelly beans to bring the weight to 2 pounds. "
        "Then, he added enough brownies to cause the weight to triple. Next, "
        "he added another 2 pounds of jelly beans. And finally, he added "
        "enough gummy worms to double the weight once again. What was the "
        "final weight of the box of goodies, in pounds?",
        "After the jelly beans the box weighed 2 pounds. After the brownies "
        "it tripled to 2 * 3 = 6 pounds. After 2 more pounds of jelly beans "
        "it was 6 + 2 = 8 pounds. After the gummy worms it doubled to "
        "8 * 2 = 16 pounds. The answer is 16.",
    ),
    (
        "Tina makes $18.00 an hour. If she works more than 8 hours per shift, "
        "she is eligible for overtime, which is paid by your hourly wage + "
        "1/2 your hourly wage. If she works 10 hours every day for 5 days, "
        "how much money does she make?",
        "Tina works 10 hours a day, which is 2 hours of overtime per day. "
        "Regular pay per day is 18 * 8 = $144. Overtime rate is "
        "18 + 18 / 2 = $27, so overtime pay is 27 * 2 = $54 per day. Daily "
        "total is 144 + 54 = $198. Over 5 days she makes 198 * 5 = $990. "
        "The answer is 990.",
    ),
]

ANSWER_PATTERN = re.compile(r"(-?\d[\d,]*\.?\d*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base and steered GSM8K accuracy."
    )
    add_circuit_arguments(parser)
    parser.set_defaults(models=["gemma"])
    parser.add_argument(
        "--behavior",
        choices=LOADERS,
        default="RTP",
    )
    parser.add_argument("--best-lambda", type=float, default=-3.0)
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--gsm8k-samples", type=int, default=250)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument(
        "--instruct-models",
        nargs="*",
        default=["llama"],
        help="Models that should use their chat template.",
    )
    return parser.parse_args()


def extract_answer(text: str) -> float | None:
    match = re.search(
        r"answer is\s*\$?(-?\d[\d,]*\.?\d*)",
        text,
        re.IGNORECASE,
    )
    matches = ANSWER_PATTERN.findall(text)
    candidate = match.group(1) if match else (matches[-1] if matches else None)
    if candidate is None:
        return None
    try:
        return float(candidate.replace(",", ""))
    except ValueError:
        return None


def gold_answer(answer: str) -> float | None:
    value = answer.split("####")[-1].strip().replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def base_prompt(question: str) -> str:
    examples = "".join(
        f"Question: {example}\nAnswer: {answer}\n\n"
        for example, answer in FEW_SHOT
    )
    return examples + f"Question: {question}\nAnswer:"


def chat_prompt(model, question: str) -> str:
    messages = []
    for example, answer in FEW_SHOT:
        messages.append(
            {"role": "user", "content": f"Question: {example}"}
        )
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": f"Question: {question}"})
    return model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(
    steerer: CircuitSteer,
    prompt: str,
    coefficient: float,
    is_instruct: bool,
    max_new_tokens: int,
) -> str:
    model = steerer.model
    input_tokens = model.to_tokens(
        prompt,
        prepend_bos=not is_instruct,
    )
    input_length = input_tokens.shape[1]
    hooks = (
        steerer.hooks(coefficient) if coefficient != 0.0 else None
    )

    def call_model():
        try:
            return model.generate(
                input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
        except Exception:
            return model.generate(
                input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                verbose=False,
            )

    if coefficient != 0.0 and hooks:
        with model.hooks(fwd_hooks=hooks):
            output = call_model()
    else:
        output = call_model()
    return model.tokenizer.decode(
        output[0, input_length:],
        skip_special_tokens=True,
    )


def main() -> None:
    args = parse_args()
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    selected = indices[: args.gsm8k_samples]
    summary_path = args.output_dir / "circuitsteer_gsm8k_summary.csv"

    for model_key in args.models:
        item_path = (
            args.output_dir / f"circuitsteer_gsm8k_{model_key}.csv"
        )
        existing = read_rows(item_path)
        done = {int(row["idx"]) for row in existing}
        base_correct = sum(row["base_ok"] == "True" for row in existing)
        steered_correct = sum(
            row["steered_ok"] == "True" for row in existing
        )
        seen = len(existing)

        source = make_datasets(
            labels=[args.behavior],
            n=args.dataset_size,
            seed=args.seed,
        )[args.behavior]
        toxic = [text_of(item) for item in source["train_toxic"]]
        benign = [text_of(item) for item in source["train_benign"]]
        steerer = CircuitSteer(model_key, config=config_from_args(args))
        steerer.fit(toxic, benign)
        is_instruct = model_key in args.instruct_models

        try:
            progress = tqdm(selected, desc=f"GSM8K/{model_key}")
            for dataset_index in progress:
                if dataset_index in done:
                    continue
                item = dataset[dataset_index]
                gold = gold_answer(item["answer"])
                prompt = (
                    chat_prompt(steerer.model, item["question"])
                    if is_instruct
                    else base_prompt(item["question"])
                )
                base_output = generate(
                    steerer,
                    prompt,
                    0.0,
                    is_instruct,
                    args.max_new_tokens,
                )
                steered_output = generate(
                    steerer,
                    prompt,
                    args.best_lambda,
                    is_instruct,
                    args.max_new_tokens,
                )
                base_prediction = extract_answer(base_output)
                steered_prediction = extract_answer(steered_output)
                base_ok = (
                    base_prediction is not None
                    and gold is not None
                    and abs(base_prediction - gold) < 1e-3
                )
                steered_ok = (
                    steered_prediction is not None
                    and gold is not None
                    and abs(steered_prediction - gold) < 1e-3
                )
                append_row(
                    item_path,
                    {
                        "model": model_key,
                        "behavior": args.behavior,
                        "best_lambda": args.best_lambda,
                        "idx": dataset_index,
                        "gold": gold,
                        "base_pred": base_prediction,
                        "steered_pred": steered_prediction,
                        "base_ok": base_ok,
                        "steered_ok": steered_ok,
                    },
                    ITEM_FIELDS,
                )
                done.add(dataset_index)
                base_correct += int(base_ok)
                steered_correct += int(steered_ok)
                seen += 1
                progress.set_postfix(
                    base=f"{base_correct / seen:.3f}",
                    steered=f"{steered_correct / seen:.3f}",
                )

            base_accuracy = base_correct / max(seen, 1)
            steered_accuracy = steered_correct / max(seen, 1)
            prior_summaries = read_rows(summary_path)
            summary = {
                "model": model_key,
                "behavior": args.behavior,
                "best_lambda": args.best_lambda,
                "n": seen,
                "base_acc": round(base_accuracy, 4),
                "steered_acc": round(steered_accuracy, 4),
                "delta_acc": round(
                    base_accuracy - steered_accuracy,
                    4,
                ),
            }
            updated_summaries = [
                row
                for row in prior_summaries
                if row["model"] != model_key
            ]
            updated_summaries.append(summary)
            write_rows(
                summary_path,
                updated_summaries,
                SUMMARY_FIELDS,
            )
            print(
                f"{model_key}: base={base_accuracy:.3f} "
                f"steered={steered_accuracy:.3f}"
            )
        finally:
            release_model(steerer)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
