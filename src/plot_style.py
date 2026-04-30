"""
Plotting style utilities for publication-quality figures.

Typical usage
-------------
Call ``use_house_style()`` once at the top of your notebook or script, then
use ``style_axes()`` on each Axes object after plotting::

    from src.utils.plot_style import use_house_style, style_axes, clean_legend
    from src.utils.plot_style import BLUES, REDS, GREYS

    use_house_style()           # sets global rcParams (serif font, dpi, etc.)

    fig, ax = plt.subplots()
    ax.plot(r_km, T_MeV, color=BLUES[0], label='Temperature')
    style_axes(ax,
               xlabel='Radius (km)', ylabel='T (MeV)',
               ylog=True)       # log scale on Y, minor ticks, thick spines
    clean_legend(ax)            # deduplicated, frameless legend
    plt.show()

API
---
- use_house_style(dpi)                     -> sets global rcParams
- style_axes(ax, xlog, ylog, xlim, ylim,
             xlabel, ylabel, title, ...)   -> styles a single Axes
- clean_legend(ax, loc, ncol, fontsize)    -> adds a clean legend

Color palettes
--------------
- BLUES : 3 blue shades  (dark -> light)
- REDS  : 4 red/orange shades
- GREYS : 3 grey shades
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator
from matplotlib.lines import Line2D
import numpy as np


def use_house_style(dpi: int = 120) -> None:
    """Set global matplotlib rcParams for a consistent house style.

    Call this once before creating any figures — it affects all subsequent
    plots in the session.

    Parameters
    ----------
    dpi : int
        Figure resolution. Default 120 (good for notebooks); use 300 for
        publication exports.

    Example
    -------
    >>> from src.utils.plot_style import use_house_style
    >>> use_house_style()          # default 120 dpi
    >>> use_house_style(dpi=300)   # high-res for saving to PDF/PNG
    """
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "font.family": "serif",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "text.usetex": False,
    })
    plt.rcParams["axes.formatter.use_mathtext"] = True


def style_axes(ax, xlog: bool = False, ylog: bool = False,
               xlim=None, ylim=None,
               xlabel: str = None, ylabel: str = None, title: str = None,
               ticksize: int = 17, spine_width: float = 1.5,
               major_len: int = 8, minor_len: int = 4) -> None:
    """Apply house style to a matplotlib Axes: spines, ticks, labels, scales.

    Call this *after* plotting so axis limits are already set before minor
    tick locators are applied.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to style.
    xlog : bool
        If True, set x-axis to log scale with decade major ticks and
        sub-decade minor ticks.
    ylog : bool
        If True, same as xlog but for the y-axis.
    xlim : tuple (float, float), optional
        Set x-axis limits, e.g. ``(0, 300)``.
    ylim : tuple (float, float), optional
        Set y-axis limits.
    xlabel : str, optional
        X-axis label (fontsize 23). Supports LaTeX math via ``$...$``.
    ylabel : str, optional
        Y-axis label (fontsize 23).
    title : str, optional
        Axes title (fontsize 15).
    ticksize : int
        Font size for tick labels. Default 17.
    spine_width : float
        Linewidth for axis spines and major ticks. Default 1.5.
    major_len : int
        Length of major ticks in points. Default 8.
    minor_len : int
        Length of minor ticks in points. Default 4.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(r_km, T_MeV)
    >>> style_axes(ax, ylog=True, xlabel='Radius (km)', ylabel='T (MeV)',
    ...            xlim=(0, 300))
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(spine_width)
    ax.tick_params(axis="both", which="major", width=spine_width, length=major_len, direction="in")
    ax.tick_params(axis="both", which="minor", width=spine_width * 0.8, length=minor_len, direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(labelsize=ticksize)

    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    if xlim   is not None: ax.set_xlim(xlim)
    if ylim   is not None: ax.set_ylim(ylim)
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=23)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=23)
    if title  is not None: ax.set_title(title,   fontsize=15)


def clean_legend(ax, loc: str = "best", ncol: int = 1, fontsize: float = 13) -> None:
    """Add a deduplicated, frameless legend to an Axes.

    Removes duplicate labels that appear when multiple lines share the same
    label (e.g. multi-panel loops).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the legend to.
    loc : str
        Legend location string passed to matplotlib, e.g. ``"upper right"``,
        ``"best"``. Default ``"best"``.
    ncol : int
        Number of columns in the legend. Default 1.
    fontsize : float
        Font size for legend text. Default 13.

    Example
    -------
    >>> ax.plot(r_km, T_MeV, label='T (MeV)')
    >>> clean_legend(ax, loc="upper right", fontsize=14)
    """
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        H.append(h)
        L.append(l)
    ax.legend(H, L, frameon=False, loc=loc, ncol=ncol, fontsize=fontsize)


# ---------- color palettes ----------
# Use by index: BLUES[0] is darkest, BLUES[-1] is lightest.
BLUES = ["#313695", "#4575b4", "#74add1"]
REDS  = ["#a50026", "#d73027", "#f46d43", "#fdae61"]
GREYS = ["#222222", "#555555", "#888888"]
