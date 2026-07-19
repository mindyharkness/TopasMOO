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
    * ``"medicalphysics"`` — Medical Physics journal-inspired formatting:
      single-column sizing, high-contrast axes, large sans-serif text.

Styles are bundled as Matplotlib style sheets so they do not require a LaTeX
installation or third-party style packages at render time.
"""

import contextlib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
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
# and will not shrink them further. Their guidance asks for >=20 pt fonts so
# text stays legible *after* that fit. We therefore author at ~2.3x the final
# column width: with the fonts fixed at 20 pt, the larger canvas gives the plot
# interior more room, and the figure still reduces to ~9 pt text in print.
# Double-column figures inherit the same scale factor via :func:`scale_figsize`.
MEDICAL_PHYSICS_SINGLE_COL_WIDTH = 2.3 * 80 / 25.4   # ~7.24 in (prints at 80 mm)
MIN_RASTER_DPI = 600

# Intermediate, in-loop monitoring plots are regenerated every ``plot_frequency``
# evaluations and immediately overwritten, so they render at a much lower DPI than
# the publication floor to keep that repeated work cheap.
INTERMEDIATE_PLOT_DPI = 150

# Authoring single-column width per publication variant. Variants absent here
# author at :data:`SINGLE_COL_WIDTH`; list one only when it authors wider so its
# mandated fonts stay proportionate after the journal fits it to a print column.
_VARIANT_AUTHORING_WIDTH = {
    "medicalphysics": MEDICAL_PHYSICS_SINGLE_COL_WIDTH,
}


def available_publication_variants() -> list[str]:
    """Return the list of supported publication variant names."""
    return list(_PUBLICATION_VARIANTS)


def active_publication_variant() -> str:
    """Return the currently active publication variant name."""
    return _active_publication_variant


def _style_width_ratio() -> float:
    """Scale factor from the standard single-column width to the active style.

    Most styles author at :data:`SINGLE_COL_WIDTH` (ratio 1.0). A variant listed
    in :data:`_VARIANT_AUTHORING_WIDTH` (e.g. Medical Physics, which authors at
    ~2x the print column so its mandated >=20 pt fonts stay proportionate) is
    enlarged by the ratio of its authoring width to the standard one.
    """
    if _active_style != "publication":
        return 1.0
    width = _VARIANT_AUTHORING_WIDTH.get(_active_publication_variant, SINGLE_COL_WIDTH)
    return width / SINGLE_COL_WIDTH


def scale_figsize(width: float, height: float) -> tuple[float, float]:
    """Scale a standard ``(width, height)`` in inches to the active style.

    Plotting functions size panels in standard single-column units; this keeps
    them unchanged for the clean/nature/ieee styles while enlarging them for
    the Medical Physics variant so fonts remain legible after the journal fits
    the figure to its column.
    """
    ratio = _style_width_ratio()
    return (width * ratio, height * ratio)


def line_width(factor: float = 1.0) -> float:
    """Line width in points: *factor* times the active style's default.

    Plot functions must size strokes through this (or :func:`marker_area`)
    rather than hardcoding point values, so that variants with heavier base
    weights (e.g. ``medicalphysics``) scale every element together.
    """
    return factor * plt.rcParams["lines.linewidth"]


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

    try:
        # Reset to matplotlib defaults first so each style is self-contained;
        # otherwise settings a style omits (e.g. a bold label weight from a
        # previously applied style) leak across apply_style calls.
        plt.style.use(["default", str(style_file)])
    except Exception as exc:
        logger.warning(
            "Failed to apply style '%s' (%s); falling back to matplotlib defaults.",
            _active_style, exc,
        )
        plt.style.use("default")


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
    grid: bool | None = None,
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

    if grid is None:
        grid = bool(plt.rcParams.get("axes.grid", False))

    ax.set_axisbelow(True)
    ax.grid(grid, which="major")
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

    axis.set_major_locator(MaxNLocator(integer=integer, steps=steps))
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


def save_publication_figure(fig, save_path, dpi=None):
    """Save a figure in both PDF and PNG format for publication.

    :param fig: Matplotlib Figure instance.
    :param save_path: Base path (without extension). Both ``.pdf`` and
        ``.png`` files will be written.
    :param dpi: Raster DPI (defaults to ``MIN_RASTER_DPI``).
    """
    if dpi is None:
        dpi = MIN_RASTER_DPI
    base = Path(save_path)
    stem = base.parent / base.stem
    for ext, kw in [(".pdf", {}), (".png", {"dpi": dpi})]:
        try:
            fig.savefig(f"{stem}{ext}", bbox_inches="tight", **kw)
        except Exception:
            fig.savefig(f"{stem}{ext}", **kw)
    logger.debug("Saved figure to %s.{pdf,png}", stem)


def finalize_figure(fig, save_path, *, own_fig: bool = True, dpi=None) -> None:
    """Save (PDF + PNG) and close a figure this call owns, if a path was given.

    The single place the plotting modules express the "save then close" teardown,
    so the save/close contract changes in one spot. A figure embedded in a
    caller-supplied Axes (``own_fig=False``) is neither saved nor closed.
    """
    if save_path is not None and own_fig:
        save_publication_figure(fig, save_path, dpi=dpi)
        plt.close(fig)
