from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .config import CIRCUIT_DEFAULTS, MODEL_CONFIGS, CircuitConfig


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
        self.sae_layers = self.profile.sae_layers
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
        for text in tqdm(texts, desc="Building edges", leave=False):
            _, cache = self.model.run_with_cache(
                text,
                stop_at_layer=self.sae_layers[-1] + 1,
            )
            active: dict[int, torch.Tensor] = {}
            for layer in self.sae_layers:
                residual = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
                feature_activations = self.saes[layer].encode(residual)
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
                rows, columns = torch.nonzero(
                    similarity > self.config.sim_thresh,
                    as_tuple=True,
                )
                for row, column in zip(rows.tolist(), columns.tolist()):
                    edge = (
                        f"L{current}_{current_indices[row].item()}",
                        f"L{following}_{following_indices[column].item()}",
                    )
                    counts[edge] += 1
        return counts

    def discover_circuit(
        self,
        toxic: list[str],
        benign: list[str],
    ) -> None:
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
        self.circuit = sorted(scored, key=lambda item: -item[1])
        print(
            f"Circuit: {len(self.circuit)} edges after contrastive filtering"
        )

    def feature_vectors(
        self,
        include_destinations: bool = False,
    ) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
        """Construct layer vectors from selected circuit features."""
        layer_features: defaultdict[int, set[int]] = defaultdict(set)
        for (origin, destination), _ in self.circuit[
            : self.config.top_k * 2
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
            vectors[layer] = pooled.to(self.dtype)
            counts[layer] = len(features)
        return vectors, counts

    def build_steering_vectors(self) -> None:
        self.steer_vecs, _ = self.feature_vectors()
        print(
            f"Steering vectors built for layers: "
            f"{sorted(self.steer_vecs)}"
        )

    def fit(self, toxic: list[str], benign: list[str]) -> None:
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
    def perplexity(self, text: str) -> float:
        if not text.strip():
            return 0.0
        loss = self.model(
            self.model.to_tokens(text),
            return_type="loss",
        )
        return float(torch.exp(loss).item())
