"""The figures' shared colour scheme and typography.

One definition, imported by every plot, so the angle sweep and the
semantic map cannot drift apart. Editing a value here changes it
everywhere rather than in whichever figure was touched last.

The categorical hues are the angle-polar figure's own two hues,
steel blue and terracotta, with a purple added for the third condition.
Keeping the polar figure's pair means a reader meets the same blue for
"the method" and the same warm tone for the ablation in both figures;
purple then reads as the third thing rather than as a restatement of
either. The set passes every check (worst adjacent dE 19.2 protan, 25.1
tritan, 28.5 normal, all >= 3:1 on white).

Type is sans-serif, which stays legible at the small sizes a figure
label runs at and is the usual choice for figures even in a serif-set
paper. Sizes assume the figure is placed at its natural width: a figure
drawn 10 inches wide and dropped into a 7-inch column has its 9pt labels
rendered at 6pt, which is the usual reason figure text is unreadable.
"""
from __future__ import annotations

#: Ink and surface. A white surface rather than a tinted one: figures sit
#: on the page, and a cream panel reads as a slide.
INK = "#1A1A2E"      # the overview figure's ink
INK2 = "#444444"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
SURFACE = "#ffffff"

#: The shaded "worse than no steering" region in the polar figure, and
#: any other area shading that should read as a warning rather than data.
HARM = "#f0eeea"

#: Models. They take the first two selection-rule hues rather than the
#: first and third: blue against purple reads at dE 3.8 under
#: deuteranopia, which for a two-series figure is no separation at all.
GEMMA = "#2B6CB0"
LLAMA = "#E76F51"

#: Selection rules. GEMMA/ALIGNED and LLAMA/ANTI share hues on purpose:
#: the same blue always means "the method", the same vermillion always
#: means "its opposite". Bluish green sits between them for the ablation
#: that drops the criterion without reversing it.
ALIGNED = "#2B6CB0"
NOGEO = "#E76F51"
ANTI = "#7D3C9E"
RANDOM = "#9a9a9a"

#: Extra hues for figures that compare method families. Red replaces
#: terracotta wherever both would appear (ΔE 14 even at normal vision).
#: Red against green is 6.8 under deuteranopia, legal only with a second
#: encoding, so any figure using both also varies marker shape.
RED = "#D1453B"
GREEN = "#2F9E6E"
ORANGE = "#E8871E"
DARK_GREY = "#3d3d3d"
MID_GREY = "#7a7a7a"
LIGHT_GREY = "#b5b5b5"
#: Light red wash for an out-of-range band (e.g. beyond the perplexity window).
WINDOW_FILL = "#fcecea"

#: Region shading in the semantic map.
ZONE = "#9a9a9a"
ZONE_FILL = "#f0f0f0"

#: Faint points for features nothing selected.
BACKDROP = "#dcdcdc"

#: Layer ramp, light to dark with depth.
LAYERS = ("#A9C2DB", "#5186BD", "#2B6CB0", "#E76F51", "#7D3C9E")

#: Sans-serif stack, in order of preference.
FONT_STACK = ["Helvetica", "Arial", "DejaVu Sans"]

#: Point sizes as they will RENDER, given a figure drawn at the width it
#: will occupy on the page (see PAGE_WIDTH).
SIZE_TITLE = 10.0
SIZE_LABEL = 9.0
SIZE_TICK = 8.0
SIZE_LEGEND = 8.5
SIZE_ANNOT = 8.0

#: Inches of a full-width figure in a two-column paper. Drawing at this
#: width is what makes the point sizes above mean what they say.
PAGE_WIDTH = 7.0

#: Raster resolution for the PNG. The PDF is vector and unaffected.
DPI = 600

#: Fill opacity for a data mark. Slightly translucent so that two markers
#: landing on the same point read as two rather than as one hard disc
#: hiding another; kept high enough that the hue is still the palette's.
MARK_ALPHA = 0.82


def use_paper_style() -> None:
    """Apply the shared type and line defaults to matplotlib.

    Called by each figure script before drawing, so a figure cannot
    quietly ship in matplotlib's default sans.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": SIZE_TITLE,
        "axes.labelsize": SIZE_LABEL,
        "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "legend.fontsize": SIZE_LEGEND,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "pdf.fonttype": 42,      # embed as TrueType, not Type 3
        "ps.fonttype": 42,
    })


def display_name(model: str) -> str:
    """Model names as they should appear in a title."""
    return {"gemma": "Gemma", "llama": "Llama"}.get(model, model.capitalize())
