#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Global matplotlib style for the thesis figures.

Usage
-----
Place this file somewhere on your PYTHONPATH, e.g.:

    thesis_project/experiment_analysis/thesis_plot_style.py

Then in each plotting script, call:

    from thesis_plot_style import set_thesis_style
    set_thesis_style()

before creating any figures.

Or use it as a context manager:

    from thesis_plot_style import thesis_style

    with thesis_style():
        fig, ax = plt.subplots()
        ...

This keeps all figures in the thesis visually consistent.
"""

import matplotlib as mpl
from contextlib import contextmanager


THESIS_RC = {
    # ---- Figure / layout ----
    "figure.figsize": (6.0, 3.5),        # 默认单图尺寸（脚本里仍可覆盖）
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    # ---- Fonts ----
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],  # 服务器上基本都有
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    # ---- Axes / lines ----
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "axes.axisbelow": True,              # grid 在曲线下面

    "lines.linewidth": 1.6,
    "lines.markersize": 5,

    # ---- Ticks ----
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,

    # ---- Legend ----
    "legend.frameon": False,
    "legend.borderaxespad": 0.5,

    # ---- PDF / text ----
    "pdf.fonttype": 42,                  # TrueType（避免 LaTeX 中乱码）
    "ps.fonttype": 42,
}


def set_thesis_style() -> None:
    """
    Set global matplotlib rcParams for thesis figures.

    Call this once at the beginning of each plotting script.
    """
    mpl.rcParams.update(THESIS_RC)


@contextmanager
def thesis_style():
    """
    Context manager version of the thesis style.

    Example:
        with thesis_style():
            fig, ax = plt.subplots()
            ...
    """
    old_rc = mpl.rcParams.copy()
    try:
        mpl.rcParams.update(THESIS_RC)
        yield
    finally:
        mpl.rcParams.update(old_rc)
