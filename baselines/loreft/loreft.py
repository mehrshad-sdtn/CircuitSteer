"""LoReFT: low-rank representation finetuning as a steering baseline.

Reference: Wu et al., "ReFT: Representation Finetuning for Language
Models", NeurIPS 2024 (arXiv:2404.03592); official implementation
stanfordnlp/pyreft. The intervention itself lives in `intervention.py`
and follows pyreft's `LoreftIntervention` exactly.

This is the one baseline here that TRAINS parameters. Every other method
derives a steering vector from activations in closed form; LoReFT fits
R, W and b by gradient descent, so it is the supervised comparison the
draft describes as "rank-4 subspace fine-tuning, 10 epochs".

Three adaptations, all recorded because they change how the number reads:

1. What it is trained on. ReFT trains on (instruction, response) pairs,
   which these datasets do not provide. The training signal here is the
   language-modelling loss on the BENIGN half of the split, so the
   intervention learns to move the model towards non-target text. That
   is the natural supervised analogue of what the other baselines derive
   contrastively.

2. What the coefficient means. A trained intervention has no signed
   direction to flip, so the shared lambda grid is read as a STRENGTH:
   scale = |lambda|, with |lambda| = 1 being the intervention exactly as
   trained and larger values extrapolating past it. Reporting a signed
   lambda here would imply a direction the method does not have.

3. Where it applies. pyreft usually intervenes on a few prefix or suffix
   positions; this applies at every position, at the same layers as the
   other baselines. Both are valid LoReFT configurations, and the
   all-position choice avoids the failure that made SpARE's Sycophancy
   cells vacuous: that metric is teacher-forced and reads logits[:-1],
   so a last-token-only edit cannot reach it.
"""
from __future__ import annotations

import os

import torch
from tqdm import tqdm

from baselines.common import BaselineSteerer
from baselines.loreft.intervention import LoReFT


class LoReFTSteerer(BaselineSteerer):
    name = "LoReFT"
    requires_saes = False

    #: Draft: "rank-4 subspace fine-tuning, 10 epochs".
    rank: int = 4
    epochs: int = 10
    learning_rate: float = 1e-3
    train_batch_size: int = 8
    max_length: int = 128
    #: Samples per backward pass. The optimiser still sees
    #: `train_batch_size` through gradient accumulation, so training is
    #: unchanged; this only bounds how many sequences' activations are
    #: held at once. Llama-3.1-8B backward over 32 layers exceeds a 40GB
    #: card at the full batch, so it accumulates instead.
    #: Gemma-2-2B at micro 8 fits RealToxicityPrompts on a 22GB
    #: card but not the longer-sequence datasets, so it is 4.
    micro_batch_size: dict = {"gemma": 4, "llama": 2}

    def _budget(self) -> tuple[int, int, int]:
        """Epochs, effective batch, and per-backward micro-batch."""
        epochs = int(os.environ.get("LOREFT_EPOCHS", self.epochs))
        batch = int(os.environ.get("LOREFT_BATCH", self.train_batch_size))
        micro = int(os.environ.get(
            "LOREFT_MICRO", self.micro_batch_size.get(self.model_key, 8)
        ))
        return max(1, epochs), max(1, batch), max(1, min(micro, batch))

    def _tokenise(self, texts: list[str]) -> torch.Tensor:
        tokens = [
            self.model.to_tokens(text)[0, : self.max_length] for text in texts
        ]
        width = max(int(t.shape[0]) for t in tokens)
        pad = self.model.tokenizer.pad_token_id or 0
        batch = torch.full((len(tokens), width), pad, dtype=torch.long,
                           device=self.device)
        mask = torch.zeros((len(tokens), width), dtype=torch.bool,
                           device=self.device)
        for i, row in enumerate(tokens):
            batch[i, : row.shape[0]] = row
            mask[i, : row.shape[0]] = True
        return batch, mask

    def _hooks_for(self, modules, scale: float):
        def make(layer):
            def hook(residual, hook=None, **kwargs):
                del kwargs
                return modules[layer](residual, scale=scale)
            return hook

        return [
            (f"blocks.{layer}.hook_resid_post", make(layer))
            for layer in modules
        ]

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        del toxic                      # trained on the benign half only
        if not benign:
            raise ValueError("LoReFT needs benign training text")

        d_model = self.model.cfg.d_model
        modules = {
            layer: LoReFT(d_model, rank=self.rank, dtype=torch.float32)
            .to(self.device)
            for layer in self.steer_layers
        }
        parameters = [p for m in modules.values() for p in m.parameters()]
        total = sum(m.parameter_count() for m in modules.values())
        epochs, batch_size, micro = self._budget()
        print(
            f"{self.name}: rank {self.rank} at layers "
            f"{list(self.steer_layers)}, {total:,} trainable parameters, "
            f"{epochs} epochs x batch {batch_size} "
            f"(micro {micro}) on {len(benign)} texts"
        )

        optimiser = torch.optim.AdamW(parameters, lr=self.learning_rate)
        order = list(range(len(benign)))
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            rng = torch.randperm(len(order))
            running, steps = 0.0, 0
            for start in tqdm(
                range(0, len(order), batch_size),
                desc=f"{self.name} epoch {epoch + 1}/{epochs}", leave=False,
            ):
                chunk = [benign[order[i]] for i in rng[start:start + batch_size]]
                if not chunk:
                    continue

                optimiser.zero_grad()
                total_loss, pieces = 0.0, 0
                # Accumulate over micro-batches: identical gradient to one
                # large backward, a fraction of the peak memory.
                for begin in range(0, len(chunk), micro):
                    piece = chunk[begin:begin + micro]
                    tokens, mask = self._tokenise(piece)
                    if tokens.shape[1] < 2:
                        continue
                    logits = self.model.run_with_hooks(
                        tokens, fwd_hooks=self._hooks_for(modules, 1.0)
                    )
                    targets = tokens[:, 1:]
                    valid = mask[:, 1:]
                    log_probs = torch.log_softmax(
                        logits[:, :-1, :].float(), dim=-1
                    )
                    picked = log_probs.gather(
                        2, targets.unsqueeze(-1)
                    ).squeeze(-1)
                    piece_loss = -(picked * valid).sum() / valid.sum().clamp_min(1)
                    # Weight by share of the batch so the accumulated
                    # gradient matches a single full-batch backward.
                    (piece_loss * (len(piece) / len(chunk))).backward()
                    total_loss += float(piece_loss) * (len(piece) / len(chunk))
                    pieces += 1
                    del logits, log_probs, picked
                if pieces == 0:
                    continue
                optimiser.step()
                loss = total_loss

                running += float(loss)
                steps += 1
            if steps:
                print(f"{self.name}: epoch {epoch + 1}/{epochs} "
                      f"loss={running / steps:.4f}", flush=True)

        for module in modules.values():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.interventions = modules
        # A residual-space summary so anything inspecting a steerer sees
        # something; `hooks()` does not read it, so nothing is applied twice.
        self.steer_vecs = {
            layer: torch.zeros(d_model, device=self.device, dtype=self.dtype)
            for layer in modules
        }
        print(f"{self.name}: trained {len(modules)} interventions")

    def hooks(self, coefficient, vectors=None, layers=None):
        """Apply the trained edit, scaled by |coefficient|.

        The magnitude is used rather than the signed value: a trained
        intervention has no direction to reverse, so a negative sign
        would not mean "suppress" the way it does for a contrastive
        vector. See the module docstring.
        """
        del vectors, layers
        modules = getattr(self, "interventions", None)
        if not modules or coefficient == 0.0:
            return []
        return self._hooks_for(modules, abs(float(coefficient)))
