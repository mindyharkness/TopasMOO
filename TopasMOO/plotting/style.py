"""Two-style plotting system for TopasMOO with publication variants.

Available styles
----------------
``"fast"``
    Cleaned-up matplotlib-default look for in-loop plotting during optimization.
``"publication"``
    Publication-grade style with selectable variants:

    * ``"clean"``  — Modern clean: subtle grid, sans-serif, colorblind-safe palette.
    * ``"nature"`` — Modern Nature/Science: bold sans-serif, high contrast.
    * ``"ieee"``  — Engineering/IEEE: Computer Modern serif, boxed axes,
      dense ticks (uses mathtext 'cm' fontset; no LaTeX install required).
    * ``"medicalphysics"`` — follows the Medical Physics (AAPM) figure guidelines:
      80 mm / 180 mm column widths, sans-serif text no smaller than 10 pt as
      printed, no bold, multi-panel parts labeled (a), (b), ...

Styles are bundled as Matplotlib style sheets so they do not require a LaTeX
installation or third-party style packages at render time.
"""

import contextlib
import logging
import string
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    MaxNLocator,
    NullLocator,
    ScalarFormatter,
)

logger = logging.getLogger(__name__)

_STYLE_DIR = Path(__file__).parent

_PUBLICATION_VARIANTS = {
    "clean": _STYLE_DIR / "topasmoo_pub_clean.mplstyle",
    "nature": _STYLE_DIR / "topasmoo_pub_nature.mplstyle",
    "ieee": _STYLE_DIR / "topasmoo_pub_ieee.mplstyle",
    "medicalphysics": _STYLE_DIR / "topasmoo_pub_medicalphysics.mplstyle",
}

_STYLE_FILES = {
    "fast": _STYLE_DIR / "topasmoo_fast.mplstyle",
    "publication": _PUBLICATION_VARIANTS["clean"],
}

DEFAULT_PUBLICATION_VARIANT = "clean"

_active_style: str = "fast"
_active_publication_variant: str = DEFAULT_PUBLICATION_VARIANT

SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.0

# Shared accent (warm orange) for "look here" overlays -- the knee-point star,
# the mean line in box plots -- chosen to stand apart from every variant's
# data-color cycle.
ACCENT_COLOR = "#E55A00"

# Medical Physics fits figures to an 80 mm (single) or 180 mm (double) column
# and asks for sans-serif text of 20 pt or more on a figure that will be shrunk
# to about half a page, i.e. at least 10 pt as printed. The variant therefore
# authors at the true column widths with text no smaller than 10 pt, and labels
# the parts of multi-panel figures (a), (b), ... as the journal's captions do.
MEDICAL_PHYSICS_SINGLE_COL_WIDTH = 80 / 25.4   # ~3.15 in
MEDICAL_PHYSICS_DOUBLE_COL_WIDTH = 180 / 25.4  # ~7.09 in
MEDICAL_PHYSICS_MIN_TEXT_PT = 10.0
MIN_RASTER_DPI = 600

# Intermediate, in-loop monitoring plots are regenerated every ``plot_frequency``
# evaluations and immediately overwritten, so they render at a much lower DPI than
# the publication floor to keep that repeated work cheap.
INTERMEDIATE_PLOT_DPI = 150


def available_publication_variants() -> list[str]:
    """Return the list of supported publication variant names."""
    return list(_PUBLICATION_VARIANTS)


def _medical_physics_active() -> bool:
    return _active_style == "publication" and _active_publication_variant == "medicalphysics"


def scale_figsize(width: float, height: float) -> tuple[float, float]:
    """Scale a standard ``(width, height)`` in inches to the active style.

    Plotting functions size panels in standard single-column units; this keeps
    them unchanged for the clean/nature/ieee styles. For Medical Physics the
    width snaps to the journal's 80 mm column (anything up to a standard single
    column) or 180 mm column (anything wider, including multi-panel grids). A
    figure is then never wider than the column, so the journal never shrinks
    its text below 10 pt. Up to a double column the height keeps the requested
    aspect ratio; wider grids keep their requested height instead, so their
    narrowed panels stay tall enough for 10 pt labels.
    """
    if not _medical_physics_active():
        return (width, height)
    target = (
        MEDICAL_PHYSICS_SINGLE_COL_WIDTH
        if width <= SINGLE_COL_WIDTH
        else MEDICAL_PHYSICS_DOUBLE_COL_WIDTH
    )
    return (target, height if width > DOUBLE_COL_WIDTH else height * target / width)


def text_size(size) -> float:
    """Font size in points for a relative (``"small"``) or absolute size.

    Floored at :data:`MEDICAL_PHYSICS_MIN_TEXT_PT` for the Medical Physics
    variant, whose guidelines require text that stays legible in print.
    """
    points = FontProperties(size=size).get_size_in_points()
    return max(points, MEDICAL_PHYSICS_MIN_TEXT_PT) if _medical_physics_active() else points


def line_width(factor: float = 1.0) -> float:
    """Line width in points: *factor* times the active style's default.

    Plot functions must size strokes through this (or :func:`marker_area`)
    rather than hardcoding point values, so that variants with heavier base
    weights (e.g. ``medicalphysics``) scale every element together.
    """
    return factor * plt.rcParams["lines.linewidth"]


def marker_edge_width(factor: float = 1.0) -> float:
    """Scatter marker outline width: :func:`line_width`, or 0 when the style
    disables marker edges (``lines.markeredgewidth: 0``, e.g. ``medicalphysics``).
    """
    return line_width(factor) if plt.rcParams["lines.markeredgewidth"] else 0.0


def marker_area(factor: float = 1.0) -> float:
    """Scatter marker area (pt²): ``(factor * style markersize)²``.

    ``factor`` scales the marker *diameter* relative to the active style's
    ``lines.markersize``, so a factor of 2 yields 4x the area.
    """
    return (factor * plt.rcParams["lines.markersize"]) ** 2


def style_colors(n: int) -> list:
    """Return the first *n* colors of the active style's property cycle.

    Falls back to matplotlib's ``"C0".."Cn"`` cycle if the active style does
    not define a color cycle, and wraps around if *n* exceeds the cycle length.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        return [f"C{i}" for i in range(n)]
    return [cycle[i % len(cycle)] for i in range(n)]


def _resolve_style_file(style: str, variant: str | None = None) -> Path:
    """Return the .mplstyle Path for *style* (and *variant* for publication)."""
    if style not in _STYLE_FILES:
        raise ValueError(
            f"Unknown style '{style}'. Choose from: {list(_STYLE_FILES)}"
        )
    if style == "publication":
        chosen_variant = variant or _active_publication_variant
        if chosen_variant not in _PUBLICATION_VARIANTS:
            raise ValueError(
                f"Unknown publication variant '{chosen_variant}'. "
                f"Choose from: {list(_PUBLICATION_VARIANTS)}"
            )
        return _PUBLICATION_VARIANTS[chosen_variant]
    return _STYLE_FILES[style]


def apply_style(
    style: str | None = None,
    variant: str | None = None,
) -> None:
    """Apply a TopasMOO plotting style globally.

    When *style* is given it becomes the new active style for this session.
    When called with no argument the previously set active style is reapplied.

    :param style: ``"fast"`` or ``"publication"``. If *None*, reapplies the
        current active style.
    :param variant: Only used when *style* is ``"publication"``. One of
        ``"clean"`` (default), ``"nature"``, ``"ieee"``,
        or ``"medicalphysics"``.
    """
    global _active_style, _active_publication_variant

    if style is not None:
        _resolve_style_file(style, variant)
        _active_style = style
        if style == "publication" and variant is not None:
            _active_publication_variant = variant
    elif variant is not None and _active_style == "publication":
        _resolve_style_file("publication", variant)
        _active_publication_variant = variant

    style_file = _resolve_style_file(_active_style, _active_publication_variant)

    # Reset to matplotlib defaults first so each style is self-contained;
    # otherwise settings a style omits (e.g. a bold label weight from a
    # previously applied style) leak across apply_style calls.
    plt.style.use(["default", str(style_file)])


@contextlib.contextmanager
def publication_style(
    style: str | None = None,
    variant: str | None = None,
):
    """Context manager that temporarily applies a TopasMOO style.

    :param style: Style name (see :func:`apply_style`). Defaults to the current
        active style.
    :param variant: Publication variant (see :func:`apply_style`). Only used when
        the active style is ``"publication"``.

    .. code-block:: python

        >>> with publication_style("publication", variant="nature"):
        ...     fig, ax = plt.subplots()
        ...     ax.plot([0, 1], [0, 1])
    """
    chosen_style = style if style is not None else _active_style
    chosen_variant = variant if variant is not None else _active_publication_variant
    style_file = _resolve_style_file(chosen_style, chosen_variant)
    # Reset to defaults first (see apply_style) so the style is self-contained.
    with plt.style.context(["default", str(style_file)]):
        yield


def format_publication_axes(
    ax,
    *,
    x_integer: bool = False,
    y_integer: bool = False,
) -> None:
    """Apply stable, publication-friendly axis formatting to a 2D Axes.

    The bundled styles set most visual defaults. This helper handles the
    parts that are easier to express in Python: natural tick intervals,
    consistent scalar formatting, and major/minor tick directions.

    Axes with categorical / manually-placed ticks (a :class:`FixedLocator`,
    e.g. a box plot keyed by parameter name) are left untouched on that axis
    so the existing labels survive.
    """
    if getattr(ax, "name", "") in {"3d", "polar"}:
        return

    ax.set_axisbelow(True)
    ax.grid(bool(plt.rcParams.get("axes.grid", False)), which="major")
    ax.tick_params(which="major", direction="out")
    ax.tick_params(which="minor", direction="in")

    # Don't let ticks bleed onto edges whose spine the style has hidden.
    ax.tick_params(
        which="both",
        top=ax.spines["top"].get_visible(),
        right=ax.spines["right"].get_visible(),
    )

    # "Natural" major intervals (1, 2, 5, 10 ...); avoids arbitrary steps.
    steps = [1, 2, 5, 10]
    _format_linear_axis(ax.xaxis, integer=x_integer, steps=steps)
    _format_linear_axis(ax.yaxis, integer=y_integer, steps=steps)


def _format_linear_axis(axis, *, integer: bool, steps: list[float]) -> None:
    """Apply natural locators/formatter to one axis, skipping fixed ticks."""
    # Categorical axes use a FixedLocator with hand-written labels; replacing
    # its locator would silently drop those labels and inject numeric ticks.
    if isinstance(axis.get_major_locator(), FixedLocator):
        return
    if axis.get_scale() != "linear":
        return

    # "auto" sizes the tick count to the axis length and font size; the default
    # (10 bins) overprints labels on narrow panels and long axes.
    axis.set_major_locator(MaxNLocator(nbins="auto", integer=integer, steps=steps))
    axis.set_major_formatter(_scalar_formatter())

    # Honor the active style's minor-tick preference instead of forcing them on.
    minor_param = f"{axis.axis_name}tick.minor.visible"
    if not bool(plt.rcParams.get(minor_param, False)):
        axis.set_minor_locator(NullLocator())
        return

    try:
        axis.set_minor_locator(AutoMinorLocator())
    except ValueError:
        axis.set_minor_locator(NullLocator())


def _scalar_formatter() -> ScalarFormatter:
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_useOffset(False)
    formatter.set_powerlimits((-3, 4))
    return formatter


def save_publication_figure(fig, save_path, dpi=None, formats=("pdf", "png")):
    """Save a figure in PDF and/or PNG format.

    :param fig: Matplotlib Figure instance.
    :param save_path: Base path (without extension).
    :param dpi: Raster DPI (defaults to ``MIN_RASTER_DPI``).
    :param formats: File types to write; ``("png",)`` for monitoring plots that
        are overwritten during a run.

    Cropping follows the active style's ``savefig.bbox``: most styles crop
    tight, while ``medicalphysics`` keeps the exact authored column width.
    """
    if dpi is None:
        dpi = MIN_RASTER_DPI
    base = Path(save_path)
    stem = base.parent / base.stem
    # Embedding TrueType fonts makes fontTools warn about harmless metadata
    # quirks in some system fonts (e.g. macOS Helvetica) on every save.
    font_log = logging.getLogger("fontTools")
    font_level = font_log.level
    font_log.setLevel(logging.ERROR)
    try:
        for ext, kw in [(f".{fmt}", {"dpi": dpi} if fmt == "png" else {}) for fmt in formats]:
            try:
                fig.savefig(f"{stem}{ext}", **kw)
            except Exception:
                fig.savefig(f"{stem}{ext}", bbox_inches=fig.bbox_inches, **kw)
    finally:
        font_log.setLevel(font_level)
    logger.debug("Saved figure to %s.{pdf,png}", stem)


def finalize_figure(fig, save_path, *, own_fig: bool = True, dpi=None, formats=("pdf", "png")) -> None:
    """Save (PDF + PNG) and close a figure this call owns, if a path was given.

    The single place the plotting modules express the "save then close" teardown,
    so the save/close contract changes in one spot. A figure embedded in a
    caller-supplied Axes (``own_fig=False``) is neither saved nor closed.
    """
    if save_path is not None and own_fig:
        if _medical_physics_active():
            _label_panels(fig)
            # The figure is saved at the exact column width (no tight crop), so
            # a title longer than the column would be cut off; wrap it instead.
            for text in [ax.title for ax in fig.axes] + [fig._suptitle]:
                if text is not None:
                    text.set_wrap(True)
        save_publication_figure(fig, save_path, dpi=dpi, formats=formats)
        plt.close(fig)


def _label_panels(fig) -> None:
    """Prefix the visible panels of a multi-panel figure with (a), (b), ...

    Colorbars and hidden surplus grid cells are not panels. Single-panel
    figures are left alone.
    """
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]
    if len(panels) < 2:
        return
    for letter, ax in zip(string.ascii_lowercase, panels, strict=False):  # >26 panels: rest unlabeled
        ax.set_title(f"({letter}) {ax.get_title()}".rstrip())
