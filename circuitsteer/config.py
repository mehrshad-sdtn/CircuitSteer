from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    sae_release: str
    sae_layers: tuple[int, ...]
    sae_id_format: str


MODEL_CONFIGS = {
    "gemma": ModelConfig(
        model_name="google/gemma-2-2b",
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_layers=(6, 12, 18, 24),
        sae_id_format="layer_{}/width_16k/canonical",
    ),
    "llama": ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        sae_release="llama_scope_lxr_8x",
        sae_layers=(3, 8, 16, 24, 29),
        sae_id_format="l{}r_8x",
    ),
}


@dataclass(frozen=True)
class CircuitConfig:
    sim_thresh: float = 0.10
    act_thresh: float = 1.5

    #: Features counted as active at a position. `act_thresh` is an
    #: ABSOLUTE cut on the SAE activation, which is not comparable across
    #: SAE families: at 1.5, Gemma-Scope yields a 9.4k-edge candidate
    #: pool where Llama-Scope yields 1.5k on the same data, so Llama's
    #: circuit is drawn from a pool six times thinner and its top-k
    #: reaches far lower specificity. Setting `act_top_n` instead keeps
    #: the n most strongly active features per layer, which makes pool
    #: density a property of the method rather than of the SAE's scale.
    act_top_n: int | None = None
    diff_thresh: float = 0.05
    top_k: int = 30
    #: How the selected features' decoder vectors are pooled into one
    #: steering vector per layer.
    #:
    #:   "sum"  - historical default. ||v|| grows with the square root of
    #:            the feature count, which varies by layer and dataset
    #:            (9 to 27 across the runs so far), so a shared lambda
    #:            does not mean a shared intervention strength.
    #:   "mean" - divides that out but leaves ||v|| tied to the decoder
    #:            norms.
    #:   "unit" - sum, then normalise to unit norm, so lambda alone sets
    #:            the strength and is comparable across layers, datasets
    #:            and models.
    steer_pool: str = "sum"

    #: Which end of each discovered edge contributes a steering feature.
    #: "source" is the historical default and leaves the deepest SAE
    #: layer unsteered, since that layer only ever appears as an edge
    #: destination (layer 24 for Gemma, 29 for Llama). "both" steers the
    #: whole discovered circuit.
    steer_nodes: str = "source"
    #: Geometric-alignment condition on an edge's decoder directions.
    #: Legacy field, kept so earlier runs reproduce. "positive" is the
    #: method, "none" is NoGeo, "negative" thresholds at cos < -sim_thresh.
    #: New work should set `select_mode` instead.
    align_mode: str = "positive"

    #: How the steering edges are chosen from the candidate pool. Every
    #: mode draws from the SAME pool (co-activating pairs surviving the
    #: contrastive filter) and returns the SAME number of edges, so a
    #: difference between modes cannot be explained by circuit size:
    #:
    #:   "aligned" - method: cos > sim_thresh, ranked by contrastive diff
    #:   "none"    - NoGeo: no geometric condition, ranked by diff
    #:   "anti"    - the k MOST anti-aligned edges (lowest cosine)
    #:   "random"  - k edges drawn uniformly, seeded
    #:
    #: "anti" is a bottom-k rule rather than a threshold on purpose. A
    #: threshold at cos < -sim_thresh can select nothing at all, and a
    #: circuit with no edges applies no steering, so its zero effect is
    #: vacuous rather than evidence about anti-alignment.
    select_mode: str = "aligned"

    #: Number of edges used to build the steering vectors. None means
    #: top_k * 2, the historical default.
    n_steer_edges: int | None = None

    #: Restrict circuit discovery to these SAE layers, as a subset of the
    #: model profile's. None uses the profile unchanged.
    sae_layers: tuple[int, ...] | None = None

    #: Seed for the "random" selection mode.
    select_seed: int = 0

    #: How edges tied on contrastive specificity are ordered. Scores are
    #: counts over a fixed denominator, so ties are the common case and
    #: which tied edge is kept decides the circuit.
    #:
    #:   "name"   - deterministic but arbitrary: lexicographic on the
    #:              edge id. Reproducible, and nothing more.
    #:   "cosine" - prefer the more geometrically aligned edge. Same
    #:              determinism, but the tiebreak now follows the method's
    #:              own criterion instead of an accident of naming.
    tie_break: str = "name"

    SELECT_MODES = ("aligned", "none", "anti", "random", "negative")

    def __post_init__(self) -> None:
        if self.steer_pool not in {"mean", "sum", "unit"}:
            raise ValueError("steer_pool must be 'mean', 'sum' or 'unit'")
        if self.steer_nodes not in {"source", "both"}:
            raise ValueError("steer_nodes must be 'source' or 'both'")
        if self.align_mode not in {"positive", "none", "negative"}:
            raise ValueError(
                "align_mode must be 'positive', 'none' or 'negative'"
            )
        if self.tie_break not in {"name", "cosine"}:
            raise ValueError("tie_break must be 'name' or 'cosine'")
        if self.select_mode not in self.SELECT_MODES:
            raise ValueError(
                f"select_mode must be one of {self.SELECT_MODES}"
            )
        if self.act_top_n is not None and self.act_top_n <= 0:
            raise ValueError("act_top_n must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.n_steer_edges is not None and self.n_steer_edges <= 0:
            raise ValueError("n_steer_edges must be positive")

    @property
    def steer_edges(self) -> int:
        return (
            self.n_steer_edges
            if self.n_steer_edges is not None
            else self.top_k * 2
        )


CIRCUIT_DEFAULTS = CircuitConfig()
# Steering only ever subtracts the discovered circuit; positive
# coefficients amplify the behaviour and 0.0 is a no-op whose delta
# is zero by construction, so neither is swept.
DEFAULT_COEFFICIENTS = (-4.0, -3.0, -2.0, -1.0)
GLOBAL_SEED = 42
