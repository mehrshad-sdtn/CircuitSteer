from __future__ import annotations

import gc
import math
from collections import defaultdict
from collections.abc import Iterable

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .config import CIRCUIT_DEFAULTS, MODEL_CONFIGS, CircuitConfig
from .selection import select_edges


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype(device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class CircuitSteer:
    """SAE-based multi-layer feature circuit steering."""

    def __init__(
        self,
        model_key: str,
        config: CircuitConfig = CIRCUIT_DEFAULTS,
        device: str | None = None,
    ) -> None:
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")

        self.model_key = model_key
        self.profile = MODEL_CONFIGS[model_key]
        self.config = config
        self.device = device or default_device()
        self.dtype = default_dtype(self.device)
        print(f"device={self.device} dtype={self.dtype}")

        self.model = HookedTransformer.from_pretrained(
            self.profile.model_name,
            device=self.device,
            dtype=self.dtype,
        ).eval()
        # An override lets a model drop a layer from the circuit without
        # editing its profile. Llama steers from layer 3 of 32, whose
        # perturbation is then amplified through 29 downstream layers;
        # Gemma's earliest is 6 of 26. Restricting the layers is a
        # configuration of the same method, not a different one.
        self.sae_layers = tuple(
            self.config.sae_layers
            if self.config.sae_layers is not None
            else self.profile.sae_layers
        )
        unknown = set(self.sae_layers) - set(self.profile.sae_layers)
        if unknown:
            raise ValueError(
                f"{model_key} has no SAE for layers {sorted(unknown)}; "
                f"available: {list(self.profile.sae_layers)}"
            )
        if len(self.sae_layers) < 2:
            raise ValueError("circuit discovery needs at least two layers")
        self.saes = self._load_saes()
        self.circuit: list[tuple[tuple[str, str], float]] = []
        self.steer_vecs: dict[int, torch.Tensor] = {}

    def _load_saes(self) -> dict[int, SAE]:
        saes: dict[int, SAE] = {}
        print(f"Loading {len(self.sae_layers)} SAEs...")
        for layer in self.sae_layers:
            loaded = SAE.from_pretrained(
                release=self.profile.sae_release,
                sae_id=self.profile.sae_id_format.format(layer),
                device=self.device,
            )
            # Supports both old and new sae-lens return signatures.
            sae = loaded[0] if isinstance(loaded, tuple) else loaded
            sae.eval()
            saes[layer] = sae
        return saes

    @torch.no_grad()
    def _build_edges(
        self,
        texts: Iterable[str],
    ) -> defaultdict[tuple[str, str], int]:
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        # Cosine of every retained edge, kept so the selection criterion
        # can be shown directly rather than inferred from outcomes.
        self.edge_cosines: dict[tuple[str, str], float] = getattr(
            self, "edge_cosines", {}
        )
        for text in tqdm(texts, desc="Building edges", leave=False):
            _, cache = self.model.run_with_cache(
                text,
                stop_at_layer=self.sae_layers[-1] + 1,
            )
            active: dict[int, torch.Tensor] = {}
            for layer in self.sae_layers:
                # A dictionary may read a different hook than the residual
                # stream (the MLP-neuron ablation reads neuron activations).
                hook = getattr(self.saes[layer], "hook_name",
                               "blocks.{}.hook_resid_post")
                residual = cache[hook.format(layer)][:, -1, :]
                feature_activations = self.saes[layer].encode(residual)
                if self.config.act_top_n is not None:
                    # Rank instead of threshold: the same number of
                    # features per layer whatever the SAE's activation
                    # scale. Still requires a positive activation, so a
                    # position with few live features contributes few.
                    values = feature_activations[0]
                    live = int((values > 0).sum())
                    keep = min(self.config.act_top_n, live)
                    indices = (
                        torch.topk(values, keep).indices if keep
                        else values.new_empty(0, dtype=torch.long)
                    )
                else:
                    indices = torch.nonzero(
                        feature_activations[0] > self.config.act_thresh
                    ).squeeze(-1)
                if indices.ndim == 0:
                    indices = indices.unsqueeze(0)
                active[layer] = indices

            for current, following in zip(
                self.sae_layers[:-1],
                self.sae_layers[1:],
            ):
                current_indices = active[current]
                following_indices = active[following]
                if (
                    current_indices.numel() == 0
                    or following_indices.numel() == 0
                ):
                    continue

                current_vectors = torch.nn.functional.normalize(
                    self.saes[current].W_dec[current_indices],
                    p=2,
                    dim=1,
                )
                following_vectors = torch.nn.functional.normalize(
                    self.saes[following].W_dec[following_indices],
                    p=2,
                    dim=1,
                )
                similarity = current_vectors @ following_vectors.T
                # The pool is deliberately unfiltered: every co-activating
                # pair enters, and the geometric criterion is applied when
                # edges are SELECTED, not when they are built. Filtering
                # here instead would give each variant its own pool and a
                # different circuit size, so any difference between
                # variants would confound the selection rule with the
                # number of edges. Cosine is a deterministic property of
                # a feature pair, so deferring the filter leaves the
                # surviving edges and their counts unchanged.
                keep = torch.ones_like(similarity, dtype=torch.bool)
                rows, columns = torch.nonzero(keep, as_tuple=True)
                for row, column in zip(rows.tolist(), columns.tolist()):
                    edge = (
                        f"L{current}_{current_indices[row].item()}",
                        f"L{following}_{following_indices[column].item()}",
                    )
                    counts[edge] += 1
                    self.edge_cosines[edge] = float(similarity[row, column])
        return counts

    def discover_circuit(
        self,
        toxic: list[str],
        benign: list[str],
    ) -> None:
        self.edge_cosines = {}
        print(
            f"Discovering circuit "
            f"({len(toxic)} toxic, {len(benign)} benign)..."
        )
        toxic_counts = self._build_edges(toxic)
        benign_counts = self._build_edges(benign)
        all_edges = set(toxic_counts) | set(benign_counts)
        n_toxic = max(len(toxic), 1)
        n_benign = max(len(benign), 1)

        scored = []
        for edge in all_edges:
            difference = (
                toxic_counts.get(edge, 0) / n_toxic
                - benign_counts.get(edge, 0) / n_benign
            )
            if difference >= self.config.diff_thresh:
                scored.append((edge, difference))
        # Secondary key on the edge name makes the ordering deterministic.
        # `all_edges` is a set, `sorted` is stable, and the scores are
        # counts over a fixed denominator, so ties are common and would
        # otherwise keep set-iteration order - which varies per process
        # with string hash randomisation. Without this the same seed
        # yields a slightly different circuit on every run.
        if self.config.tie_break == "cosine":
            self.pool = sorted(
                scored,
                key=lambda item: (
                    -item[1], -self.edge_cosines.get(item[0], 0.0), item[0]
                ),
            )
        else:
            self.pool = sorted(scored, key=lambda item: (-item[1], item[0]))
        self.pool_size = len(self.pool)
        self.circuit = self._select_edges(self.pool)
        cosines = [self.edge_cosines.get(edge, 0.0) for edge, _ in self.circuit]
        self.selected_cosine = (
            sum(cosines) / len(cosines) if cosines else float("nan")
        )
        print(
            f"Circuit: pool={self.pool_size} selected={len(self.circuit)} "
            f"mode={self.config.select_mode} "
            f"mean_cos={self.selected_cosine:.4f}"
        )

    def _select_edges(
        self,
        pool: list[tuple[tuple[str, str], float]],
    ) -> list[tuple[tuple[str, str], float]]:
        """Pick the steering edges from the pool under the active rule."""
        return select_edges(
            pool,
            self.edge_cosines,
            mode=self.config.select_mode,
            k=self.config.steer_edges,
            sim_thresh=self.config.sim_thresh,
            seed=self.config.select_seed,
        )

    def feature_vectors(
        self,
        include_destinations: bool | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
        """Construct layer vectors from selected circuit features.

        `include_destinations` defaults to the config's `steer_nodes`;
        passing it explicitly overrides that, which the layer sweep relies
        on.
        """
        if include_destinations is None:
            include_destinations = self.config.steer_nodes == "both"
        layer_features: defaultdict[int, set[int]] = defaultdict(set)
        for (origin, destination), _ in self.circuit[
            : self.config.steer_edges
        ]:
            nodes = (origin, destination) if include_destinations else (origin,)
            for node in nodes:
                layer_text, feature_text = node.split("_")
                layer_features[int(layer_text[1:])].add(int(feature_text))

        vectors: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}
        for layer, features in layer_features.items():
            if layer not in self.saes:
                continue
            feature_indices = torch.tensor(
                sorted(features),
                device=self.device,
                dtype=torch.long,
            )
            decoder_vectors = self.saes[layer].W_dec[feature_indices]
            if self.config.steer_pool == "mean":
                pooled = decoder_vectors.mean(0)
            else:
                pooled = decoder_vectors.sum(0)
                if self.config.steer_pool == "unit":
                    # Strength then comes from lambda alone, rather than
                    # from how many features happened to be selected in
                    # this layer.
                    pooled = pooled / pooled.norm().clamp_min(1e-6)
            vectors[layer] = pooled.to(self.dtype)
            counts[layer] = len(features)
        return vectors, counts

    def build_steering_vectors(self) -> None:
        self.steer_vecs, counts = self.feature_vectors()
        # Norms are printed because they, not lambda alone, set the
        # intervention strength under "sum" pooling: the same lambda is a
        # different push in every layer and every dataset.
        summary = ", ".join(
            f"L{layer}: n={counts[layer]} "
            f"|v|={float(self.steer_vecs[layer].float().norm()):.2f}"
            for layer in sorted(self.steer_vecs)
        )
        print(
            f"Steering vectors ({self.config.steer_pool} pooling, "
            f"{self.config.steer_nodes} nodes) -> {summary}"
        )

    def release_saes(self) -> None:
        """Free the SAEs once steering vectors exist.

        SAEs are only needed for circuit discovery; `hooks()` uses the
        pooled steering vectors alone. On a 24GB card, Llama-3.1-8B's
        weights plus five SAEs leave no room for a batched forward pass,
        so dropping them between fit and evaluation is what makes the run
        fit. `fit()` reloads them on demand.
        """
        if not self.saes:
            return
        self.saes.clear()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("SAEs released")

    def fit(self, toxic: list[str], benign: list[str]) -> None:
        if not self.saes:
            self.saes = self._load_saes()
        self.discover_circuit(toxic, benign)
        self.build_steering_vectors()

    def hooks(
        self,
        coefficient: float,
        vectors: dict[int, torch.Tensor] | None = None,
        layers: Iterable[int] | None = None,
    ) -> list[tuple[str, object]]:
        source = self.steer_vecs if vectors is None else vectors
        allowed = set(layers) if layers is not None else set(source)
        scaled = {
            layer: vector * coefficient
            for layer, vector in source.items()
            if layer in allowed
        }
        if not scaled:
            return []

        def hook(residual, hook=None, **kwargs):
            del kwargs
            layer = hook.layer()
            if layer in scaled:
                residual = residual + scaled[layer]
            return residual

        return [
            (f"blocks.{layer}.hook_resid_post", hook)
            for layer in scaled
        ]

    def _prepare_batching(self) -> None:
        """Left-pad so batched generation aligns on the final token."""
        tokenizer = self.model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        coefficient: float,
        seed: int,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        batch_size: int = 32,
    ) -> list[str]:
        """Generate for many prompts at once.

        TransformerLens decodes one token per Python step, so a single
        prompt leaves the GPU almost idle: on an A100, 64 prompts cost the
        same wall time as 1. Batching is the difference between a ~10 hour
        and a ~1.5 hour sweep.

        The seed is set once per batch rather than once per prompt, so
        outputs are reproducible for a given (seed, batch_size, prompt
        order) but are not bit-identical to the unbatched path.
        """
        if not prompts:
            return []
        self._prepare_batching()
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "verbose": False,
        }
        hooks = self.hooks(coefficient)
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            torch.manual_seed(seed + start)
            if coefficient != 0.0 and hooks:
                with self.model.hooks(fwd_hooks=hooks):
                    generated = self.model.generate(chunk, **generation_kwargs)
            else:
                generated = self.model.generate(chunk, **generation_kwargs)
            for prompt, text in zip(chunk, generated):
                # Padded batches come back with the pad/BOS prefix intact,
                # so slice on the prompt rather than on a token count.
                index = text.find(prompt)
                tail = (
                    text[index + len(prompt):]
                    if index != -1
                    else text[len(prompt):]
                )
                outputs.append(tail.strip())
        return outputs

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        coefficient: float,
        seed: int,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
    ) -> str:
        torch.manual_seed(seed)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "verbose": False,
        }
        hooks = self.hooks(coefficient)
        if coefficient != 0.0 and hooks:
            with self.model.hooks(fwd_hooks=hooks):
                output = self.model.generate(prompt, **generation_kwargs)
        else:
            output = self.model.generate(prompt, **generation_kwargs)
        return output[len(prompt):].strip()

    @torch.no_grad()
    def perplexity_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[float]:
        """Per-text perplexity, computed one text at a time.

        Sequences here have different lengths and padding would corrupt
        the mean loss, so this keeps the simple loop; it is ~1% of runtime.
        """
        del batch_size
        return [self.perplexity(text) for text in texts]

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        # Empty text has no perplexity. Returning 0.0 made a collapsed
        # generation look infinitely fluent and let it pass the norm_ppl
        # guard, so the undefined case is NaN and callers aggregate with
        # nan-aware means.
        if not text.strip():
            return math.nan
        loss = self.model(
            self.model.to_tokens(text),
            return_type="loss",
        )
        return float(torch.exp(loss).item())
