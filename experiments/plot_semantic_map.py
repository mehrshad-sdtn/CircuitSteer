"""Where each selection rule lands in the SAE's semantic space.

One map of meaning, built from the Neuronpedia explanations of every
feature any rule selected, then the same map twice: once showing what
the aligned rule picked, once the anti-aligned rule. Concentration is
the whole point, so the figure is built to make it a matter of looking
rather than of reading numbers.

Layout of meaning comes from TF-IDF over the explanations, reduced with
an SVD and laid out with t-SNE. That is lexical similarity standing in
for semantic similarity - fine here because the explanations are short
and reuse vocabulary within a topic, but it is a proxy, and a cluster is
only ever as good as the words that built it.

Territories are drawn as smooth density hulls rather than hard polygons:
a convex hull around a handful of points implies a boundary the data
does not support.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from experiments.palette import (  # noqa: E402
    DPI, MARK_ALPHA, PAGE_WIDTH, SIZE_ANNOT, SIZE_LEGEND,
    SIZE_TITLE,
    use_paper_style,
    ALIGNED, ANTI, BACKDROP, DARK_GREY, GRID, INK, INK2, MUTED, NOGEO, RANDOM,
    SURFACE, ZONE, ZONE_FILL, display_name,
)

#: One shape for every rule; only colour separates them. Blue, green and
#: orange stay distinct for the common red-green deficiencies, and the
#: small fixed offsets below keep co-selected features from hiding each
#: other, which is what the differing shapes used to do.
MODE_STYLE = {
    "aligned": (ALIGNED, "o", "CircuitSteer"),
    "none": (NOGEO, "o", "NoGeo"),
    "anti": (ANTI, "o", "AntiAlign"),
    "random": (RANDOM, "o", "RandomEdges"),
}

#: Words that appear across every Neuronpedia explanation and so carry
#: no topic information; they would otherwise dominate cluster names.
STOP = {
    "terms", "related", "references", "phrases", "words", "concepts",
    "specific", "various", "context", "contexts", "expressions", "term",
    "reference", "phrase", "word", "concept", "associated", "involving",
    "mentions", "instances", "descriptions", "elements", "items", "used",
    "including", "particularly", "especially", "often", "general",
}

#: One lexicon per behaviour. Each is a coarse keyword match over a third
#: party's explanations - a proxy for "on topic", not a measurement of
#: what a feature encodes - and each was fixed before per-variant counts
#: were looked at.
LEXICONS = {
    "toxicity": re.compile(
        r"\b("
        r"profan\w*|vulgar|obscen\w*|swear\w*|curs\w*|slur\w*|"
        r"insult\w*|derogat\w*|offens\w*|abus\w*|demean\w*|"
        r"hostil\w*|aggress\w*|anger|angry|outrage\w*|threat\w*|violen\w*|"
        r"sexual\w*|sex|explicit|"
        r"foolish\w*|stupid\w*|idioc\w*|incompeten\w*|"
        r"critic\w*|negative|frustrat\w*|contempt\w*|hypocris\w*|crude"
        r")\b",
        re.I,
    ),
    # Sycophancy is answer-matching: deferring to a stated view, agreeing,
    # affirming, softening disagreement. The survey framing of the dataset
    # means opinion/belief vocabulary is part of the target too.
    "sycophancy": re.compile(
        r"\b("
        r"agree\w*|agreement|concur\w*|consensus|"
        r"flatter\w*|prais\w*|compliment\w*|approv\w*|endors\w*|"
        r"validat\w*|affirm\w*|reassur\w*|supportive|"
        r"defer\w*|deferen\w*|concede\w*|conform\w*|appeas\w*|"
        r"accommodat\w*|polite\w*|obsequious|sycophan\w*|"
        r"opinion\w*|belief\w*|stance|attitude\w*|preference\w*|"
        r"persuas\w*|convinc\w*|assent|acknowledg\w*"
        r")\b",
        re.I,
    ),
}

#: Which behaviour a dataset label is about.
TASK_BEHAVIOUR = {
    "RTP": "toxicity", "Jigsaw": "toxicity", "Emotion": "toxicity",
    "Sycophancy": "sycophancy",
}
BEHAVIOUR_WORD = {"toxicity": "toxicity-related",
                  "sycophancy": "agreement-related"}


def build_space(texts, seed=0, n_clusters=6):
    vectoriser = TfidfVectorizer(stop_words="english", min_df=1,
                                 sublinear_tf=True)
    tfidf = vectoriser.fit_transform(texts)
    dims = min(24, tfidf.shape[1] - 1, len(texts) - 1)
    reduced = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(
        tfidf
    )
    xy = TSNE(
        n_components=2, random_state=seed,
        perplexity=min(18, max(5, len(texts) // 6)),
        init="pca", learning_rate="auto",
    ).fit_transform(reduced)
    labels = KMeans(n_clusters=n_clusters, random_state=seed,
                    n_init=10).fit_predict(reduced)
    return xy, labels, tfidf, np.array(vectoriser.get_feature_names_out())


def name_clusters(tfidf, cluster_ids, vocabulary, n_words=2):
    names = {}
    for cluster in sorted(set(cluster_ids)):
        mask = cluster_ids == cluster
        weight = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        order = np.argsort(weight)[::-1]
        picked = [
            vocabulary[i] for i in order
            if vocabulary[i] not in STOP and len(vocabulary[i]) > 2
        ][:n_words]
        names[cluster] = " / ".join(picked) if picked else f"cluster {cluster}"
    return names


def toxic_zone(ax, xy, toxic, grid, colour=ZONE):
    """The region of meaning the toxicity lexicon occupies.

    Computed once from the flagged features and drawn identically in both
    panels, so the two rules are judged against the same target rather
    than each against a zone its own selections defined.
    """
    points = xy[toxic]
    if len(points) < 4:
        return None
    kde = gaussian_kde(points.T, bw_method=0.30)
    xx, yy = grid
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    z /= z.max() or 1.0
    ax.contourf(xx, yy, z, levels=np.linspace(0.30, 1.0, 5),
                colors=[colour], alpha=0.12, zorder=2)
    ax.contour(xx, yy, z, levels=[0.30], colors=[colour], linewidths=1.6,
               alpha=0.8, linestyles="--", zorder=3)
    return z


def name_regions(z, grid, xy, flagged, tfidf, vocabulary, level=0.30):
    """Name each shaded lobe from the features inside it.

    The shading is a density over flagged features and the cluster
    labels come from k-means; they are two different partitions, so a
    cluster name only lands on a lobe by luck. Naming the lobes directly
    from their own contents is what makes every shaded region on the
    figure account for itself.
    """
    from scipy import ndimage

    xx, yy = grid
    components, count = ndimage.label(z >= level)
    if count == 0:
        return []

    # Which component, if any, each point falls in.
    col = np.clip(np.searchsorted(xx[0], xy[:, 0]) - 1, 0, xx.shape[1] - 1)
    row = np.clip(np.searchsorted(yy[:, 0], xy[:, 1]) - 1, 0, yy.shape[0] - 1)
    member = components[row, col]

    regions = []
    for comp in range(1, count + 1):
        inside = np.where((member == comp) & flagged)[0]
        if len(inside) == 0:
            continue
        weight = np.asarray(tfidf[inside].mean(axis=0)).ravel()
        picked = [
            vocabulary[i] for i in np.argsort(weight)[::-1]
            if vocabulary[i] not in STOP and len(vocabulary[i]) > 2
        ][:2]
        if not picked:
            continue
        total = int(np.sum(member == comp))
        purity = len(inside) / max(total, 1)
        regions.append({
            "name": " / ".join(picked),
            "anchor": xy[inside].mean(axis=0),
            "size": len(inside),
            "purity": purity,
        })
    return regions


def _place_names(ax, fig, xy, items, zone=None):
    """Put cluster names where nothing else already is.

    Positions are searched outward from each cluster centroid and scored
    on measured text extents, so a name never lands on the legend, on
    another name, or on top of a crowd of points. Nudging centroids
    apart by a fixed margin - the earlier approach - cannot see any of
    those things and still collided.
    """
    if not items:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)

    taken = []
    legend = ax.get_legend()
    if legend is not None:
        taken.append(legend.get_window_extent(renderer).expanded(1.06, 1.12))

    points = ax.transData.transform(xy)
    # Cells of the shaded regions, as obstacles: a name should sit beside
    # a region, not on top of the shading it names.
    zone_pts = (ax.transData.transform(zone) if zone is not None and len(zone)
                else np.empty((0, 2)))
    span = xy.max(0) - xy.min(0)
    # The compact layout searches further out: a crowded cluster's name is
    # better a short leader line away than sitting on the points.
    radii = ([0.0, 0.035, 0.07, 0.11, 0.16, 0.22, 0.29, 0.36]
             if SHORT_LEGEND else [0.0, 0.035, 0.07, 0.11, 0.16])
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    strongest = max((it["size"] for it in items), default=1)
    for item in items:
        label, centroid = item["name"], item["anchor"]
        # A label's weight reflects what it rests on: how many flagged
        # features the region holds, and how pure the region is. A name
        # derived from one feature should not look as authoritative as
        # one derived from five.
        # Size alone drives how loud a name looks, so that reading the
        # figure by weight is reading it by how many flagged features the
        # region holds. Purity used to be multiplied in here, which broke
        # the correspondence: a two-feature region that happened to be
        # pure came out looking heavier than a five-feature one that was
        # not. Purity now only adjusts opacity, within a narrow band.
        share = item["size"] / strongest if strongest else 0.0
        strong = share >= 0.6
        alpha = min(1.0, 0.55 + 0.35 * share + 0.10 * item["purity"])
        text = ax.text(
            centroid[0], centroid[1], label.upper(),
            fontsize=SIZE_ANNOT - 1.3 + 2.4 * share - (1.0 if SHORT_LEGEND else 0.0),
            color=INK if strong else INK2,
            weight="bold" if strong else "normal",
            alpha=alpha,
            ha="center", va="center",
            # In the compact layout names sit under the highlighted
            # features and let them show through, rather than hide them.
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.28" if SHORT_LEGEND
                      else "round,pad=0.32", fc=SURFACE,
                      ec=INK2 if strong else GRID,
                      lw=1.0 if strong else 0.6,
                      alpha=(0.95 if SHORT_LEGEND else 0.92) * alpha),
        )

        best = None
        for radius in radii:
            for angle in ([0.0] if radius == 0 else angles):
                offset = np.array([np.cos(angle), np.sin(angle)]) * span * radius
                text.set_position(centroid + offset)
                box = text.get_window_extent(renderer)
                if not axes_box.contains(box.x0, box.y0) or \
                   not axes_box.contains(box.x1, box.y1):
                    continue
                if any(box.overlaps(other) for other in taken):
                    continue
                pad = 4.0        # keep a little clear space around glyphs
                crowd = int(np.sum(
                    (points[:, 0] > box.x0 - pad)
                    & (points[:, 0] < box.x1 + pad)
                    & (points[:, 1] > box.y0 - pad)
                    & (points[:, 1] < box.y1 + pad)
                ))
                # Covering a marker costs far more than moving further
                # out. Weighted the other way, a label happily sits on
                # two or three points rather than step outward once.
                covered = int(np.sum(
                    (zone_pts[:, 0] > box.x0) & (zone_pts[:, 0] < box.x1)
                    & (zone_pts[:, 1] > box.y0) & (zone_pts[:, 1] < box.y1)
                ))
                score = (radius * 100 + crowd * (160 if SHORT_LEGEND else 45)
                         + covered * (25 if SHORT_LEGEND else 0))
                # Mild preference for sitting above a cluster, which
                # reads as a caption rather than as a data point.
                if np.sin(angle) < 0:
                    score += 12
                if best is None or score < best[0]:
                    best = (score, centroid + offset)
            if best is not None and best[0] < 1:
                break

        text.set_position(best[1] if best else centroid)
        # With no free spot the fallback is the centroid, which can hang
        # off a narrow panel; slide the label back inside the axes.
        box = text.get_window_extent(renderer)
        dx = max(0.0, axes_box.x0 + 3 - box.x0) - max(0.0, box.x1 - axes_box.x1 + 3)
        dy = max(0.0, axes_box.y0 + 3 - box.y0) - max(0.0, box.y1 - axes_box.y1 + 3)
        if dx or dy:
            px, py = ax.transData.transform(text.get_position())
            text.set_position(ax.transData.inverted().transform((px + dx, py + dy)))
        # Clamping can land it on a label placed earlier; step it
        # vertically, toward the roomier half of the panel, until clear.
        box = text.get_window_extent(renderer)
        mid = (axes_box.y0 + axes_box.y1) / 2
        step = box.height * 0.6 * (1 if box.y0 < mid else -1)
        for _ in range(12):
            box = text.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in taken):
                break
            px, py = ax.transData.transform(text.get_position())
            text.set_position(ax.transData.inverted().transform((px, py + step)))
        taken.append(text.get_window_extent(renderer).expanded(1.12, 1.9))


#: Full model names, for titles that need the exact checkpoint.
FULL_NAME = {"gemma": "Gemma-2-2B", "llama": "Llama-3.1-8B"}

#: Multiplier on marker and text sizes; set from --scale.
SCALE = 1.0

#: Where each panel's legend goes; set from --legend-below.
LEGEND_BELOW = False

#: --clean: small, faint marks for off-topic selections and no region
#: names, so the eye goes straight to the on-topic features.
CLEAN = False

#: Multiplier on legend text only; set from --legend-scale.
LEGEND_SCALE = 1.0

#: --short-legend: the behaviour word moves into the legend title, so a
#: legend inside a half-width panel stays narrower than the panel.
SHORT_LEGEND = False


#: Each rule is drawn at a small fixed offset around a feature's true
#: position, so a feature picked by two rules shows up twice instead of
#: one marker hiding the other.
OFFSETS = {"aligned": (-1.0, 0.58), "none": (0.0, -1.15),
           "anti": (1.0, 0.58), "random": (0.0, 1.15)}


def panel(ax, fig, xy, row, modes, grid, word):
    """One model: the shared map, with every rule overlaid."""
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)

    z = toxic_zone(ax, xy, row["flagged"], grid)
    ax.scatter(xy[:, 0], xy[:, 1], s=16 * SCALE ** 2, color=BACKDROP, zorder=1,
               linewidth=0)

    step = (xy.max(0) - xy.min(0)) * 0.013
    handles = []
    for mode in modes:
        colour, marker, name = MODE_STYLE.get(mode, (MUTED, "o", mode))
        # Compact layout: AntiAlign in dark grey, so the two methods that
        # select on-topic features are the only coloured ones.
        if SHORT_LEGEND and mode == "anti":
            colour = DARK_GREY
        mask = row["selected"](mode)
        dx, dy = OFFSETS.get(mode, (0.0, 0.0))
        shift = np.array([dx * step[0], dy * step[1]])
        pts = xy[mask] + shift

        flagged = row["flagged"][mask]
        ax.scatter(pts[~flagged, 0], pts[~flagged, 1],
                   s=(14 if CLEAN else 34) * SCALE ** 2, color=colour,
                   marker=marker,
                   alpha=(0.35 if CLEAN else 0.30) * MARK_ALPHA,
                   edgecolor="none" if CLEAN or SHORT_LEGEND else INK,
                   linewidth=0.0 if CLEAN or SHORT_LEGEND else 0.40, zorder=4)
        # A hairline outline separates overlapping markers without the
        # heavy ring that made the earlier version look cluttered.
        # Compact layout: solid, unoutlined markers, so colour alone
        # carries the method and the on-topic features stand out.
        ax.scatter(pts[flagged, 0], pts[flagged, 1], s=132 * SCALE ** 2, color=colour,
                   marker=marker, alpha=1.0 if SHORT_LEGEND else MARK_ALPHA,
                   edgecolor="none" if SHORT_LEGEND else INK,
                   linewidth=0.0 if SHORT_LEGEND else 0.95, zorder=7)

        hits = int(flagged.sum())
        handles.append(plt.Line2D(
            [], [], marker=marker, linestyle="none", markersize=9 * SCALE,
            markerfacecolor=colour,
            markeredgecolor="none" if SHORT_LEGEND else INK,
            markeredgewidth=0.0 if SHORT_LEGEND else 0.55,
            label=(f"{name}   {hits}/{int(mask.sum())}  "
                   f"({hits / max(mask.sum(), 1):.0%})"
                   if LEGEND_BELOW or SHORT_LEGEND else
                   f"{name}   {hits}/{int(mask.sum())} "
                   f"{word}  ({hits / max(mask.sum(), 1):.0%})"),
        ))

    # The shaded lobes are a density over flagged features, not clusters,
    # so no cluster name will ever sit on them. Naming the shading in the
    # legend stops it reading as an unlabelled cluster.
    handles.append(mpatches.Patch(
        facecolor=ZONE_FILL, edgecolor=ZONE, linewidth=1.4,
        linestyle="--",
        label=(f"{word.split('-')[0].capitalize()} region"
               if LEGEND_BELOW and SHORT_LEGEND
               else f"Region of {word} features"
               if LEGEND_BELOW or SHORT_LEGEND
               else f"Where {word} features lie"),
    ))

    # Legend first: placement measures it as an obstacle.
    if LEGEND_BELOW:
        # Below the map, so a narrow panel keeps its whole area for data.
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), frameon=True,
                  ncol=2 if SHORT_LEGEND else 1, columnspacing=1.2,
                  handletextpad=0.4,
                  title=(f"{word.capitalize()} share of selected features"
                         if SHORT_LEGEND
                         else f"Selected features that are {word}"),
                  title_fontsize=SIZE_LEGEND * LEGEND_SCALE,
                  fontsize=SIZE_LEGEND * LEGEND_SCALE, framealpha=1.0,
                  edgecolor=GRID, borderpad=0.6, labelspacing=0.45)
    else:
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  title=(f"Selected features that are {word}"
                         if SHORT_LEGEND else None),
                  title_fontsize=SIZE_LEGEND * LEGEND_SCALE,
                  fontsize=SIZE_LEGEND * LEGEND_SCALE, framealpha=1.0,
                  edgecolor=GRID, borderpad=0.7, labelspacing=0.55)
    regions = name_regions(z, grid, xy, row["flagged"], row["tfidf"],
                           row["vocabulary"]) if z is not None else []
    regions.sort(key=lambda r: -r["size"])
    if z is not None:
        xx, yy = grid
        inside = z >= 0.30
        regions_zone = np.column_stack([xx[inside], yy[inside]])
        for r in regions:
            r["zone"] = regions_zone
    # Placed by the caller once the layout is final: text keeps its point
    # size while tight_layout shrinks the axes, so names placed now would
    # collide after the resize.
    return regions


def prepare(geometry_path, labels_path, n_clusters, seed):
    """Everything one (model, task) panel-row needs."""
    data = json.loads(geometry_path.read_text())
    labels = json.loads(labels_path.read_text())

    keys = sorted(k for k, v in labels.items() if v)
    texts = [labels[k] for k in keys]
    index = {k: i for i, k in enumerate(keys)}

    def selected(mode):
        feats = data["selections"][mode]["features"]
        mask = np.zeros(len(keys), dtype=bool)
        for layer, items in feats.items():
            for feature in items:
                i = index.get(f"L{layer}_{feature}")
                if i is not None:
                    mask[i] = True
        return mask

    xy, cluster_ids, tfidf, vocabulary = build_space(
        texts, seed=seed, n_clusters=n_clusters
    )
    names = name_clusters(tfidf, cluster_ids, vocabulary)

    behaviour = TASK_BEHAVIOUR.get(data["task"], "toxicity")
    lexicon = LEXICONS[behaviour]
    flagged = np.array([bool(lexicon.search(t)) for t in texts])

    shares = [
        (flagged & (cluster_ids == c)).sum() / max((cluster_ids == c).sum(), 1)
        for c in sorted(set(cluster_ids))
    ]
    hot = int(np.argmax(shares))
    # Name the toxic region plus the largest few. A cluster holding most
    # of the map is too diffuse for one centroid label to describe, so
    # `--clusters` should be high enough to break it up.
    sizes = sorted(set(cluster_ids),
                   key=lambda c: -int((cluster_ids == c).sum()))
    return {
        "data": data, "xy": xy, "cluster_ids": cluster_ids, "names": names,
        "flagged": flagged, "hot": hot, "keep": {hot, *sizes[:4]},
        "selected": selected, "behaviour": behaviour,
        "tfidf": tfidf, "vocabulary": vocabulary,
    }


def main() -> None:
    use_paper_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path, nargs="+",
                        help="one or more demo_geometry JSON files; each "
                             "becomes a row")
    parser.add_argument("--labels", type=Path, nargs="*", default=None,
                        help="matching label files (default: sibling "
                             "feature_labels.json)")
    parser.add_argument("--ncols", type=int, default=1,
                        help="Panels per row; 2 lays four "
                             "panels out as a 2x2 grid.")
    parser.add_argument("--clusters", type=int, default=10,
                        help="too few and one cluster covers most "
                             "of the map, leaving large regions "
                             "with no usable label")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+",
                        default=["aligned", "none", "anti"],
                        help="selection rules to show, one column each")
    parser.add_argument("--title", default=None)
    parser.add_argument("--panel-title", default="{model}  \u00b7  {task}",
                        help="format for each panel title; fields "
                             "{model}, {model_full} and {task}")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiplier on marker and text sizes")
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--axis-labels", action="store_true",
                        help="label the two t-SNE axes")
    parser.add_argument("--legend-below", action="store_true")
    parser.add_argument("--short-legend", action="store_true")
    parser.add_argument("--top-pad", type=float, default=0.12,
                        help="empty space above the data, as a fraction "
                             "of its height; room for an in-panel legend")
    parser.add_argument("--legend-scale", type=float, default=1.0,
                        help="multiplier on legend text")
    parser.add_argument("--clean", action="store_true",
                        help="no region names; small faint marks for "
                             "off-topic selections")
    parser.add_argument("--width", type=float, default=PAGE_WIDTH,
                        help="figure width, inches")
    parser.add_argument("--height", type=float, default=None,
                        help="figure height per row, inches")
    parser.add_argument("--out", type=Path,
                        default=Path("results/figures/fig_semantic_map.png"))
    args = parser.parse_args()
    global SCALE, LEGEND_BELOW, CLEAN, LEGEND_SCALE
    SCALE, LEGEND_BELOW, CLEAN = args.scale, args.legend_below, args.clean
    LEGEND_SCALE = args.legend_scale
    global SHORT_LEGEND
    SHORT_LEGEND = args.short_legend

    label_paths = args.labels or [
        g.with_name("feature_labels.json") for g in args.geometry
    ]
    if len(label_paths) != len(args.geometry):
        raise SystemExit("need one label file per geometry file")

    rows = [
        prepare(g, l, args.clusters, args.seed)
        for g, l in zip(args.geometry, label_paths)
    ]

    n = len(rows)
    ncols = max(1, int(getattr(args, "ncols", 1) or 1))
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(args.width, (args.height or (
                                 3.9 if ncols == 1 else 3.4)) * nrows),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    flat = [a for rowaxes in axes for a in rowaxes]
    for extra in flat[n:]:                    # unused cell in a ragged grid
        extra.set_visible(False)
    placements = []
    for row, ax in zip(rows, flat):
        xy = row["xy"]
        pad = (xy.max(0) - xy.min(0)) * 0.12
        lo, hi = xy.min(0) - pad, xy.max(0) + pad
        hi[1] = xy.max(0)[1] + (xy.max(0) - xy.min(0))[1] * args.top_pad
        grid = np.meshgrid(np.linspace(lo[0], hi[0], 200),
                           np.linspace(lo[1], hi[1], 200))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        regions = panel(ax, fig, xy, row, args.modes, grid,
                        BEHAVIOUR_WORD[row["behaviour"]])
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        placements.append((ax, xy, regions))
        ax.set_title(
            args.panel_title.format(
                model=display_name(row['data']['model']),
                model_full=FULL_NAME.get(row['data']['model'],
                                         row['data']['model']),
                task=row['data']['task']),
            fontsize=SIZE_TITLE * max(SCALE, 1.0), color=INK, weight="bold",
            pad=10,
        )
        if args.axis_labels:
            # t-SNE coordinates have no units; what the axes carry is
            # nearness in meaning of the features' descriptions.
            ax.set_xlabel("Semantic dimension 1 (t-SNE)",
                          fontsize=SIZE_LEGEND, color=INK2)
            ax.set_ylabel("Semantic dimension 2 (t-SNE)",
                          fontsize=SIZE_LEGEND, color=INK2)

    if args.title:
        fig.suptitle(args.title, fontsize=SIZE_TITLE + 1, color=INK,
                     y=1.0 - 0.005 * nrows)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97 if args.title else 1.0))
    for ax, xy, regions in placements:
        if not CLEAN:
            _place_names(ax, fig, xy, regions,
                         zone=regions[0].get("zone") if regions else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE,
                bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE,
                bbox_inches="tight")
    print(f"wrote {args.out}")
    for row in rows:
        d = row["data"]
        print(f"  {d['model']}/{d['task']} ({row['behaviour']}):")
        for c in sorted(set(row["cluster_ids"])):
            m = row["cluster_ids"] == c
            print(f"    {row['names'][c]:34s} n={int(m.sum()):3d} "
                  f"flagged={(row['flagged'] & m).sum() / max(m.sum(), 1):.0%}")


if __name__ == "__main__":
    main()
