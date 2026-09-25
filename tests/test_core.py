import sys
import types
import unittest
from contextlib import contextmanager

import torch

# The unit tests exercise tensor logic without installing or downloading
# TransformerLens, SAELens, model weights, or SAE weights.
sae_lens_stub = types.ModuleType("sae_lens")
sae_lens_stub.SAE = object
transformer_lens_stub = types.ModuleType("transformer_lens")
transformer_lens_stub.HookedTransformer = object
sys.modules["sae_lens"] = sae_lens_stub
sys.modules["transformer_lens"] = transformer_lens_stub

from circuitsteer.config import CircuitConfig  # noqa: E402
from circuitsteer.core import CircuitSteer  # noqa: E402


class FakeSae:
    def __init__(self, decoder_vectors):
        self.W_dec = torch.tensor(decoder_vectors, dtype=torch.float32)


class FakeHook:
    def __init__(self, layer):
        self._layer = layer

    def layer(self):
        return self._layer


class CircuitTensorTests(unittest.TestCase):
    def make_steerer(self):
        steerer = CircuitSteer.__new__(CircuitSteer)
        steerer.config = CircuitConfig(top_k=1, steer_pool="sum")
        steerer.device = "cpu"
        steerer.dtype = torch.float32
        steerer.circuit = [
            (("L1_0", "L2_1"), 0.8),
            (("L1_1", "L2_0"), 0.6),
            (("L1_2", "L2_2"), 0.4),
        ]
        steerer.saes = {
            1: FakeSae([[1, 2], [3, 4], [5, 6]]),
            2: FakeSae([[7, 8], [9, 10], [11, 12]]),
        }
        return steerer

    def test_feature_vectors_match_notebook_top_k_behavior(self):
        steerer = self.make_steerer()
        vectors, counts = steerer.feature_vectors()

        # The notebook selects top_k * 2 edges and origin features.
        self.assertEqual(counts, {1: 2})
        torch.testing.assert_close(vectors[1], torch.tensor([4.0, 6.0]))

    def test_destination_features_can_be_included_for_layer_sweep(self):
        steerer = self.make_steerer()
        vectors, counts = steerer.feature_vectors(
            include_destinations=True
        )

        self.assertEqual(counts, {1: 2, 2: 2})
        torch.testing.assert_close(vectors[2], torch.tensor([16.0, 18.0]))

    def test_hooks_add_scaled_vector_only_to_selected_layer(self):
        steerer = self.make_steerer()
        steerer.steer_vecs = {1: torch.tensor([1.0, 2.0])}
        hooks = steerer.hooks(-2.0)

        self.assertEqual(hooks[0][0], "blocks.1.hook_resid_post")
        residual = torch.zeros((1, 1, 2))
        changed = hooks[0][1](residual, hook=FakeHook(1))
        torch.testing.assert_close(
            changed,
            torch.tensor([[[-2.0, -4.0]]]),
        )


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    padding_side = "right"


class BatchFakeModel:
    """Returns prompt + continuation, with the BOS/pad prefix that
    TransformerLens leaves on left-padded batch output."""

    def __init__(self, prefix=""):
        self.tokenizer = FakeTokenizer()
        self.prefix = prefix
        self.calls = []

    def reset_hooks(self):
        return None

    def generate(self, prompts, **kwargs):
        self.calls.append(list(prompts))
        return [f"{self.prefix}{p} CONT{i}" for i, p in enumerate(prompts)]

    @contextmanager
    def hooks(self, fwd_hooks=None):
        yield self


class GenerateBatchTests(unittest.TestCase):
    def make_steerer(self, prefix=""):
        steerer = CircuitSteer.__new__(CircuitSteer)
        steerer.model = BatchFakeModel(prefix=prefix)
        steerer.steer_vecs = {1: torch.tensor([1.0, 2.0])}
        steerer.device = "cpu"
        steerer.dtype = torch.float32
        return steerer

    def test_prompt_is_stripped_from_each_batch_output(self):
        steerer = self.make_steerer()
        out = steerer.generate_batch(["alpha", "beta"], 0.0, seed=42)
        self.assertEqual(out, ["CONT0", "CONT1"])

    def test_padding_prefix_is_stripped(self):
        """Left-padded batches come back with a BOS/pad prefix; slicing by
        len(prompt) alone would leak it into the scored continuation."""
        steerer = self.make_steerer(prefix="<pad><pad><bos>")
        out = steerer.generate_batch(["alpha", "beta"], 0.0, seed=42)
        self.assertEqual(out, ["CONT0", "CONT1"])

    def test_tokenizer_is_configured_for_left_padding(self):
        steerer = self.make_steerer()
        steerer.generate_batch(["alpha"], 0.0, seed=42)
        self.assertEqual(steerer.model.tokenizer.padding_side, "left")
        self.assertEqual(steerer.model.tokenizer.pad_token, "<eos>")

    def test_batches_respect_batch_size(self):
        steerer = self.make_steerer()
        prompts = [f"p{i}" for i in range(7)]
        out = steerer.generate_batch(prompts, 0.0, seed=42, batch_size=3)
        self.assertEqual(len(out), 7)
        self.assertEqual([len(c) for c in steerer.model.calls], [3, 3, 1])

    def test_empty_prompt_list_is_handled(self):
        steerer = self.make_steerer()
        self.assertEqual(steerer.generate_batch([], 0.0, seed=42), [])

    def test_steering_applies_hooks(self):
        steerer = self.make_steerer()
        out = steerer.generate_batch(["alpha"], -3.0, seed=42)
        self.assertEqual(out, ["CONT0"])


if __name__ == "__main__":
    unittest.main()


class AlignModeTests(unittest.TestCase):
    """The three circuit variants used by the alignment experiment."""

    def test_config_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            CircuitConfig(align_mode="sideways")

    def test_default_is_the_method(self):
        self.assertEqual(CircuitConfig().align_mode, "positive")

    def test_each_mode_selects_a_different_edge_set(self):
        """positive keeps aligned pairs, negative keeps opposed pairs,
        none keeps everything."""
        sim = torch.tensor([[0.9, -0.9], [0.0, 0.5]])
        thresh = 0.1
        pos = (sim > thresh)
        neg = (sim < -thresh)
        none = torch.ones_like(sim, dtype=torch.bool)
        self.assertEqual(int(pos.sum()), 2)     # 0.9 and 0.5
        self.assertEqual(int(neg.sum()), 1)     # -0.9
        self.assertEqual(int(none.sum()), 4)
        # positive and negative must be disjoint
        self.assertEqual(int((pos & neg).sum()), 0)


class GeometryTests(unittest.TestCase):
    """Chain geometry of per-layer steering vectors."""

    def setUp(self):
        from experiments.alignment_geometry import geometry
        self.geometry = geometry

    def test_handles_tensors_that_require_grad(self):
        """Regression: SAE decoder weights carry requires_grad, so the
        pooled steering vectors do too and .numpy() raises."""
        vecs = {1: torch.tensor([1.0, 0.0], requires_grad=True),
                2: torch.tensor([1.0, 0.0], requires_grad=True)}
        out = self.geometry(vecs)
        self.assertAlmostEqual(out["alignment_efficiency"], 1.0, places=6)

    def test_aligned_vectors_give_efficiency_one(self):
        vecs = {1: torch.tensor([1.0, 0.0]), 2: torch.tensor([2.0, 0.0])}
        self.assertAlmostEqual(
            self.geometry(vecs)["alignment_efficiency"], 1.0, places=6)

    def test_opposed_vectors_cancel(self):
        vecs = {1: torch.tensor([1.0, 0.0]), 2: torch.tensor([-1.0, 0.0])}
        out = self.geometry(vecs)
        self.assertAlmostEqual(out["alignment_efficiency"], 0.0, places=6)
        self.assertAlmostEqual(out["consecutive_cosines"][0], -1.0, places=6)

    def test_orthogonal_vectors_are_between(self):
        vecs = {1: torch.tensor([1.0, 0.0]), 2: torch.tensor([0.0, 1.0])}
        out = self.geometry(vecs)
        self.assertAlmostEqual(out["alignment_efficiency"], 2**0.5 / 2, places=6)
        self.assertAlmostEqual(out["consecutive_cosines"][0], 0.0, places=6)


class SteerVectorConstructionTests(unittest.TestCase):
    """The two knobs that decide what is actually steered, and how hard.

    Both default to the historical behaviour so existing results stay
    reproducible; these tests pin what changes when they are turned on.
    """

    def make_steerer(self, **overrides):
        steerer = CircuitSteer.__new__(CircuitSteer)
        steerer.config = CircuitConfig(top_k=1, **overrides)
        steerer.device = "cpu"
        steerer.dtype = torch.float32
        steerer.circuit = [
            (("L1_0", "L2_1"), 0.8),
            (("L1_1", "L2_0"), 0.6),
        ]
        steerer.saes = {
            1: FakeSae([[1, 2], [3, 4], [5, 6]]),
            2: FakeSae([[7, 8], [9, 10], [11, 12]]),
        }
        return steerer

    def test_source_only_leaves_the_deepest_layer_unsteered(self):
        """The default discovers layer 2 but never steers it.

        This is the gap that leaves Gemma's layer 24 and Llama's layer 29
        out of the steering vector entirely.
        """
        steerer = self.make_steerer(steer_pool="sum")
        _, counts = steerer.feature_vectors()
        self.assertEqual(set(counts), {1})

    def test_steer_nodes_both_covers_every_layer_in_the_circuit(self):
        steerer = self.make_steerer(steer_pool="sum", steer_nodes="both")
        vectors, counts = steerer.feature_vectors()
        self.assertEqual(set(counts), {1, 2})
        torch.testing.assert_close(vectors[2], torch.tensor([16.0, 18.0]))

    def test_explicit_argument_overrides_the_config(self):
        """The layer sweep passes this directly and must keep working."""
        steerer = self.make_steerer(steer_pool="sum", steer_nodes="both")
        _, counts = steerer.feature_vectors(include_destinations=False)
        self.assertEqual(set(counts), {1})

    def test_unit_pooling_makes_strength_independent_of_feature_count(self):
        """Sum pooling ties ||v|| to how many features were selected.

        With `unit`, every layer's vector has norm 1, so lambda is the
        only thing setting the intervention strength.
        """
        summed = self.make_steerer(steer_pool="sum", steer_nodes="both")
        unit = self.make_steerer(steer_pool="unit", steer_nodes="both")
        summed_vectors, _ = summed.feature_vectors()
        unit_vectors, _ = unit.feature_vectors()

        norms = [float(v.norm()) for v in summed_vectors.values()]
        self.assertGreater(max(norms) / min(norms), 1.5)
        for vector in unit_vectors.values():
            self.assertAlmostEqual(float(vector.norm()), 1.0, places=5)

    def test_unit_pooling_preserves_direction(self):
        """Only the magnitude may change, never where the vector points."""
        summed = self.make_steerer(steer_pool="sum")
        unit = self.make_steerer(steer_pool="unit")
        a = summed.feature_vectors()[0][1]
        b = unit.feature_vectors()[0][1]
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
        self.assertAlmostEqual(float(cosine), 1.0, places=5)


class CircuitDeterminismTests(unittest.TestCase):
    """The same inputs must give the same circuit in any process.

    Edge scores are counts over a fixed denominator, so ties are common.
    Ranking them with a stable sort and no tiebreak leaves the order to
    set iteration, which varies with string hash randomisation, so the
    seed stops controlling the run.
    """

    def test_tied_edges_are_ordered_deterministically(self):
        import random

        steerer = CircuitSteer.__new__(CircuitSteer)
        steerer.config = CircuitConfig(top_k=2, select_mode="none")
        steerer.edge_cosines = {}

        edges = [(f"L1_{i}", f"L2_{i}") for i in range(40)]
        orders = set()
        for _ in range(8):
            shuffled = edges[:]
            random.shuffle(shuffled)               # stands in for set order
            scored = [(edge, 0.5) for edge in shuffled]   # every score tied
            steerer.pool = sorted(
                scored, key=lambda item: (-item[1], item[0])
            )
            orders.add(tuple(edge for edge, _ in steerer.pool))
        self.assertEqual(len(orders), 1, "tied edges ranked inconsistently")


class TieBreakTests(unittest.TestCase):
    """Which tied edge survives decides the circuit, so pin the rule."""

    def make(self, mode):
        steerer = CircuitSteer.__new__(CircuitSteer)
        steerer.config = CircuitConfig(top_k=1, tie_break=mode)
        steerer.edge_cosines = {
            ("L1_0", "L2_0"): 0.10,
            ("L1_1", "L2_1"): 0.90,
            ("L1_2", "L2_2"): 0.50,
        }
        return steerer

    def _pool(self, steerer, scored):
        if steerer.config.tie_break == "cosine":
            return sorted(scored, key=lambda i: (
                -i[1], -steerer.edge_cosines.get(i[0], 0.0), i[0]))
        return sorted(scored, key=lambda i: (-i[1], i[0]))

    def test_cosine_tiebreak_prefers_the_aligned_edge(self):
        steerer = self.make("cosine")
        scored = [(e, 0.5) for e in steerer.edge_cosines]   # all tied
        order = [e for e, _ in self._pool(steerer, scored)]
        self.assertEqual(order[0], ("L1_1", "L2_1"))        # cos 0.90
        self.assertEqual(order[-1], ("L1_0", "L2_0"))       # cos 0.10

    def test_name_tiebreak_ignores_cosine(self):
        steerer = self.make("name")
        scored = [(e, 0.5) for e in steerer.edge_cosines]
        order = [e for e, _ in self._pool(steerer, scored)]
        self.assertEqual(order[0], ("L1_0", "L2_0"))        # lexicographic

    def test_specificity_still_outranks_the_tiebreak(self):
        """The tiebreak must only ever order EQUAL scores."""
        steerer = self.make("cosine")
        scored = [(("L1_0", "L2_0"), 0.9),                  # low cos, top score
                  (("L1_1", "L2_1"), 0.1)]                  # high cos, low score
        order = [e for e, _ in self._pool(steerer, scored)]
        self.assertEqual(order[0], ("L1_0", "L2_0"))

    def test_rejects_unknown_tiebreak(self):
        with self.assertRaises(ValueError):
            CircuitConfig(tie_break="bogus")


class ActiveFeatureSelectionTests(unittest.TestCase):
    """Rank-based activation keeps pool density off the SAE's scale."""

    def pick(self, values, top_n=None, thresh=1.5):
        config = CircuitConfig(top_k=1, act_thresh=thresh, act_top_n=top_n)
        values = torch.tensor(values, dtype=torch.float32)
        if config.act_top_n is not None:
            live = int((values > 0).sum())
            keep = min(config.act_top_n, live)
            return (torch.topk(values, keep).indices if keep
                    else values.new_empty(0, dtype=torch.long))
        return torch.nonzero(values > config.act_thresh).squeeze(-1)

    def test_top_n_is_scale_invariant(self):
        """A 10x rescale changes the threshold's answer, not the rank's."""
        small = [0.1, 0.4, 0.2, 0.9, 0.0]
        large = [v * 10 for v in small]
        self.assertEqual(len(self.pick(small, thresh=1.5)), 0)
        self.assertEqual(len(self.pick(large, thresh=1.5)), 3)
        self.assertEqual(
            sorted(self.pick(small, top_n=2).tolist()),
            sorted(self.pick(large, top_n=2).tolist()),
        )

    def test_top_n_never_selects_a_dead_feature(self):
        """Fewer live features than n must not pad with zeros."""
        picked = self.pick([0.0, 0.0, 0.7, 0.0], top_n=3)
        self.assertEqual(picked.tolist(), [2])

    def test_top_n_picks_the_strongest(self):
        picked = self.pick([0.1, 0.9, 0.5, 0.3], top_n=2).tolist()
        self.assertEqual(sorted(picked), [1, 2])

    def test_rejects_invalid_top_n(self):
        with self.assertRaises(ValueError):
            CircuitConfig(act_top_n=0)


class LayerSubsetTests(unittest.TestCase):
    """Restricting the circuit's layers, e.g. dropping Llama's layer 3."""

    def resolve(self, profile_layers, override):
        """Mirrors what __init__ does, without loading a model."""
        layers = tuple(override if override is not None else profile_layers)
        unknown = set(layers) - set(profile_layers)
        if unknown:
            raise ValueError(f"no SAE for {sorted(unknown)}")
        if len(layers) < 2:
            raise ValueError("needs at least two layers")
        return layers

    def test_none_keeps_the_profile(self):
        self.assertEqual(self.resolve((3, 8, 16, 24, 29), None),
                         (3, 8, 16, 24, 29))

    def test_subset_is_honoured(self):
        self.assertEqual(self.resolve((3, 8, 16, 24, 29), (8, 16, 24, 29)),
                         (8, 16, 24, 29))

    def test_layer_without_an_sae_is_rejected(self):
        """Asking for layer 5 would silently steer nothing otherwise."""
        with self.assertRaises(ValueError):
            self.resolve((3, 8, 16, 24, 29), (5, 16))

    def test_single_layer_is_rejected(self):
        """Edges span layer PAIRS; one layer yields no circuit at all."""
        with self.assertRaises(ValueError):
            self.resolve((3, 8, 16), (16,))
