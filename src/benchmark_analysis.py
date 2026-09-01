"""Numerical and plotting helpers for the Koichi benchmark notebook."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import ulysses
from ulysses.etab1DME import fast_RHS as fast_rhs_dme

from src.etaB_1DME_sct import EtaB_1DME_sct
from src.etaB_1BE1F_sct import EtaB_1BE1F_sct
from src.etaB_1BE_flavoured_sct import EtaB_1BE2F_sct, EtaB_1BE3F_sct
from src.plot_style import clean_legend, style_axes
from src.scattering import scat_Ss, scat_St, washout_sct
from utils.koichi_benchmarks import scale_heavy_masses


ETA_B_OBS = 6.12e-10


class EtaB_1DME_LambdaMuZero(ulysses.EtaB_1DME):
    """Paper diagnostic DME with the muon decoherence rate switched off."""

    def RHS(self, y0, z, epstt, epsmm, epsee, epstm, epste, epsme,
            c1t, c1m, c1e, k):
        if z != self._currz or z == self.zmin:
            n1, d1, w1 = self.NDW1(k, z)
            self._d, self._w1, self._n1eq = d1.real, w1.real, n1
            self._currz = z
        width_tau = 485e-10 * self.MP / self.M1
        return fast_rhs_dme(
            y0, self._d, self._w1, self._n1eq,
            epstt, epsmm, epsee, epstm, epste, epsme,
            c1t, c1m, c1e, width_tau, 0.0,
        )


class EtaB_1DME_sct_LambdaMuZero(EtaB_1DME_sct):
    """Scattering DME with the paper's auxiliary Lambda_mu=0 switch."""

    include_muon_decoherence = False


def epsilon_matrix(params: dict) -> np.ndarray:
    """Return ULYSSES' Hermitian CP-asymmetry matrix epsilon^(1)."""
    model = ulysses.EtaB_1DME()
    model.setParams(params)
    return np.array(
        [[model.epsilon1ab(a, b) for b in range(3)] for a in range(3)],
        dtype=complex,
    )


def benchmark_summary(params: dict, *, scattering: bool = True) -> dict:
    """Solve a benchmark and return epsilon, kappa1 and eta_B."""
    model_cls = EtaB_1DME_sct if scattering else ulysses.EtaB_1DME
    model = model_cls()
    model.setParams(params)
    eta_b = float(np.real(model.EtaB))
    return {
        "epsilon": epsilon_matrix(params),
        "trace": np.trace(epsilon_matrix(params)),
        "kappa1": float(np.real(model.k1)),
        "eta_B": eta_b,
        "model": model,
    }


def scan_eta_b(params: dict, m1_values, *, progress: bool = True) -> dict:
    """Scan Fig. 1's four solvers and the DME scattering extension."""
    masses = np.asarray(m1_values, dtype=float)
    names_and_classes = (
        ("DME", ulysses.EtaB_1DME),
        ("DME+sct", EtaB_1DME_sct),
        ("1BE1F", ulysses.EtaB_1BE1F),
        ("1BE2F", ulysses.EtaB_1BE2F),
        ("1BE3F", ulysses.EtaB_1BE3F),
    )
    values = {name: np.empty_like(masses) for name, _ in names_and_classes}
    for i, mass in enumerate(masses):
        point = scale_heavy_masses(params, mass)
        for name, cls in names_and_classes:
            model = cls()
            model.setParams(point)
            # ULYSSES 2.0.11's 1BE1F contains an unconditional diagnostic print.
            with redirect_stdout(StringIO()):
                values[name][i] = np.real(model.EtaB)
        if progress:
            print(f"{i + 1:>2}/{len(masses)}  M1={mass:.3e} GeV", end="\r")
    if progress:
        print()
    return {"M1": masses, **values}


def scan_dme_scattering(params: dict, m1_values, *, progress=True) -> dict:
    """Compare the validated DME benchmark with its scattering extension."""
    masses = np.asarray(m1_values, dtype=float)
    plain = np.empty_like(masses)
    scattered = np.empty_like(masses)
    for i, mass in enumerate(masses):
        point = scale_heavy_masses(params, mass)
        for output, cls in ((plain, ulysses.EtaB_1DME),
                            (scattered, EtaB_1DME_sct)):
            model = cls()
            model.setParams(point)
            output[i] = np.real(model.EtaB)
        if progress:
            print(f"{i + 1:>3}/{len(masses)}  M1={mass:.3e} GeV", end="\r")
    if progress:
        print()
    return {"M1": masses, "DME": plain, "DME+sct": scattered}


def scan_boltzmann_scattering(params: dict, m1_values, *, progress=True) -> dict:
    """Task 1-a: compare 1-, 2-, and 3-flavour BEs with/without scattering."""
    masses = np.asarray(m1_values, dtype=float)
    solvers = (
        ("1BE1F", ulysses.EtaB_1BE1F),
        ("1BE1F+sct", EtaB_1BE1F_sct),
        ("1BE2F", ulysses.EtaB_1BE2F),
        ("1BE2F+sct", EtaB_1BE2F_sct),
        ("1BE3F", ulysses.EtaB_1BE3F),
        ("1BE3F+sct", EtaB_1BE3F_sct),
    )
    values = {name: np.empty_like(masses) for name, _ in solvers}
    for i, mass in enumerate(masses):
        point = scale_heavy_masses(params, mass)
        for name, cls in solvers:
            model = cls()
            model.setParams(point)
            with redirect_stdout(StringIO()):
                values[name][i] = np.real(model.EtaB)
        if progress:
            print(f"{i + 1:>3}/{len(masses)}  M1={mass:.3e} GeV", end="\r")
    if progress:
        print()
    return {"M1": masses, **values}


def scan_all_methods_scattering(params: dict, m1_values, *, progress=True) -> dict:
    """Scan DME and all three BE approximations, with/without scattering."""
    masses = np.asarray(m1_values, dtype=float)
    solvers = (
        ("DME", ulysses.EtaB_1DME),
        ("DME+sct", EtaB_1DME_sct),
        ("1BE1F", ulysses.EtaB_1BE1F),
        ("1BE1F+sct", EtaB_1BE1F_sct),
        ("1BE2F", ulysses.EtaB_1BE2F),
        ("1BE2F+sct", EtaB_1BE2F_sct),
        ("1BE3F", ulysses.EtaB_1BE3F),
        ("1BE3F+sct", EtaB_1BE3F_sct),
    )
    values = {name: np.empty_like(masses) for name, _ in solvers}
    for i, mass in enumerate(masses):
        point = scale_heavy_masses(params, mass)
        for name, cls in solvers:
            model = cls()
            model.setParams(point)
            with redirect_stdout(StringIO()):
                values[name][i] = np.real(model.EtaB)
        if progress:
            print(f"{i + 1:>3}/{len(masses)}  M1={mass:.3e} GeV", end="\r")
    if progress:
        print()
    return {"M1": masses, **values}


def plot_all_methods_scattering(scan: dict):
    """Overlay Fig. 1 methods; dashed=no scattering, solid=with scattering."""
    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor="white")
    colors = {
        "DME": "tab:blue", "1BE1F": "tab:orange",
        "1BE2F": "tab:green", "1BE3F": "tab:red",
    }
    for method, color in colors.items():
        ax.plot(scan["M1"], np.abs(scan[method]) * 1e10,
                color=color, ls="--", lw=2.0,
                label=f"{method}, no scattering")
        ax.plot(scan["M1"], np.abs(scan[f"{method}+sct"]) * 1e10,
                color=color, ls="-", lw=2.2,
                label=f"{method}, with scattering")
    ax.axhline(ETA_B_OBS * 1e10, color="0.55", lw=1.0,
               label="observed BAU")
    ax.axvline(1e12, color="0.25", lw=1.0)
    style_axes(ax, xlog=True, ylog=True, xlim=(1e8, 1e14), ylim=(1e-18, 1e5),
               xlabel=r"$M_1$ [GeV]", ylabel=r"$|\eta_B|\times10^{10}$",
               title="Real-CI benchmark: methods and scattering")
    clean_legend(ax, fontsize=8.2, ncol=2)
    ax.text(0.035, 0.035,
            r"Dashed: no scattering; solid: with scattering. "
            r"Real CI gives $\eta_B^{\rm 1BE1F}=0$.",
            transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    return fig, ax


def plot_boltzmann_scattering(scan: dict):
    """Plot literal BE comparison requested in task 1-a."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5), facecolor="white")
    colors = {"1BE2F": "tab:green", "1BE3F": "tab:red"}
    base_styles = {"1BE2F": "--", "1BE3F": "-."}
    # For a real CI matrix Tr(epsilon)=0, so 1BE1F vanishes analytically;
    # retain its numerical values in the table but do not distort the plot.
    for base in ("1BE2F", "1BE3F"):
        ax.plot(scan["M1"], np.abs(scan[base]) * 1e10,
                color=colors[base], ls=base_styles[base], lw=2,
                label=f"{base}, without scattering")
        ax.plot(scan["M1"], np.abs(scan[f"{base}+sct"]) * 1e10,
                color=colors[base], ls="-", lw=2,
                label=f"{base}, with scattering")
    ax.axhline(ETA_B_OBS * 1e10, color="0.55", lw=1.0, label="observed BAU")
    ax.axvline(1e12, color="0.25", lw=1.0)
    style_axes(ax, xlog=True, ylog=True, xlim=(1e8, 1e14), ylim=(1e-3, 1e5),
               xlabel=r"$M_1$ [GeV]", ylabel=r"$|\eta_B|\times10^{10}$",
               title="Boltzmann equations: scattering comparison")
    clean_legend(ax, fontsize=8.5, ncol=2)
    ax.text(0.04, 0.05, r"Real CI: $\eta_B^{\rm 1BE1F}=0$",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_dme_scattering(scan: dict):
    """Plot task 1-a with the same axes and normalization as paper Fig. 1."""
    fig, ax = plt.subplots(figsize=(6.2, 5.3), facecolor="white")
    specifications = (
        ("DME", "tab:blue", "without scattering"),
        ("DME+sct", "tab:purple", r"with $\Delta L=1$ scattering"),
    )
    for key, color, label in specifications:
        eta = np.asarray(scan[key])
        positive = np.where(eta >= 0, np.abs(eta) * 1e10, np.nan)
        negative = np.where(eta < 0, np.abs(eta) * 1e10, np.nan)
        ax.plot(scan["M1"], positive, color=color, lw=2.2,
                label=f"{label} (+)")
        ax.plot(scan["M1"], negative, color=color, lw=2.2, ls="--",
                label=f"{label} (-)")
    ax.axhline(ETA_B_OBS * 1e10, color="0.55", lw=1.0,
               label="observed BAU")
    ax.axvline(1e12, color="0.25", lw=1.0)
    style_axes(ax, xlog=True, ylog=True, xlim=(1e8, 1e14), ylim=(1e-3, 1e5),
               xlabel=r"$M_1$ [GeV]", ylabel=r"$|\eta_B|\times10^{10}$",
               title="Real-CI benchmark: scattering comparison")
    clean_legend(ax, fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_eta_b_scan(scan: dict, title: str):
    """Paper-like signed-line plot; line style records the BAU sign."""
    fig, ax = plt.subplots(figsize=(7.2, 5.3), facecolor="white")
    styles = {
        "DME": ("tab:blue", 2.3), "DME+sct": ("tab:purple", 2.3),
        "1BE1F": ("tab:orange", 1.6), "1BE2F": ("tab:green", 1.6),
        "1BE3F": ("tab:red", 1.6),
    }
    for name, (color, width) in styles.items():
        y = np.asarray(scan[name])
        for positive, linestyle, suffix in ((True, "-", r"$\eta_B>0$"),
                                             (False, "--", r"$\eta_B<0$")):
            shown = np.where((y >= 0) == positive, np.abs(y), np.nan)
            ax.plot(scan["M1"], shown, color=color, ls=linestyle,
                    lw=width, label=f"{name}, {suffix}")
    ax.axhline(ETA_B_OBS, color="0.45", lw=1.3, label="observed")
    ax.axvline(1e12, color="0.2", lw=1.0, ls=":")
    style_axes(ax, xlog=True, ylog=True, xlabel=r"$M_1$ [GeV]",
               ylabel=r"$|\eta_B|$", title=title)
    ax.grid(which="both", ls=":", color="gray", alpha=0.22)
    clean_legend(ax, fontsize=10, ncol=2)
    fig.tight_layout()
    return fig, ax


def paper_fig1_points(base_params: dict) -> dict:
    """Return the four parameter choices in the caption of paper Fig. 1."""
    panels = {}
    for row, sign in (("top", -1.0), ("bottom", 1.0)):
        for column, y3 in (("left", 30.0), ("right", 0.0)):
            point = dict(base_params)
            point.update({"x1": sign * 10.0, "x2": sign * 20.0,
                          "x3": sign * 10.0, "y1": 0.0,
                          "y2": 0.0, "y3": y3})
            panels[(row, column)] = point
    return panels


def scan_paper_fig1(base_params: dict, m1_values, *, progress=True) -> dict:
    """Compute exactly the five curves shown in each panel of Fig. 1."""
    masses = np.asarray(m1_values, dtype=float)
    solvers = (
        ("DME", ulysses.EtaB_1DME),
        ("DME (Lambda_mu=0)", EtaB_1DME_LambdaMuZero),
        ("1BE1F", ulysses.EtaB_1BE1F),
        ("1BE2F", ulysses.EtaB_1BE2F),
        ("1BE3F", ulysses.EtaB_1BE3F),
    )
    result = {}
    panels = paper_fig1_points(base_params)
    total = len(panels) * len(masses)
    count = 0
    for panel, benchmark in panels.items():
        curves = {name: np.empty_like(masses) for name, _ in solvers}
        for i, mass in enumerate(masses):
            point = scale_heavy_masses(benchmark, mass)
            for name, cls in solvers:
                model = cls()
                model.setParams(point)
                with redirect_stdout(StringIO()):
                    curves[name][i] = np.real(model.EtaB)
            count += 1
            if progress:
                print(f"{count:>3}/{total}  panel={panel}, M1={mass:.2e}", end="\r")
        result[panel] = {"M1": masses, **curves}
    if progress:
        print()
    return result


def plot_paper_fig1(scans: dict):
    """Four-panel reproduction using the paper's normalization and styles."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.3), sharex=True, sharey=True,
                             facecolor="white")
    styles = {
        "1BE1F": dict(color="orange", ls=":"),
        "1BE2F": dict(color="green", ls="--"),
        "1BE3F": dict(color="red", ls="-."),
    }
    for row_i, row in enumerate(("top", "bottom")):
        for col_i, column in enumerate(("left", "right")):
            ax = axes[row_i, col_i]
            scan = scans[(row, column)]
            mass = scan["M1"]
            dme = scan["DME"]
            # Split the DME at sign changes as in the caption.
            ax.plot(mass, np.where(dme >= 0, np.abs(dme) * 1e10, np.nan),
                    color="tab:blue", lw=2, label="DME (+)")
            ax.plot(mass, np.where(dme < 0, np.abs(dme) * 1e10, np.nan),
                    color="tab:blue", lw=2, ls="--", label="DME (-)")
            ax.plot(mass, np.abs(scan["DME (Lambda_mu=0)"]) * 1e10,
                    color="tab:blue", lw=1.5, ls=":", label=r"DME ($\Lambda_\mu=0$)")
            for name, style in styles.items():
                ax.plot(mass, np.abs(scan[name]) * 1e10, lw=1.8,
                        label=name, **style)
            ax.axhline(ETA_B_OBS * 1e10, color="0.55", lw=0.9)
            ax.axvline(1e12, color="0.25", lw=0.9)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e8, 1e14)
            ax.set_ylim(1e-3, 1e5)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if row_i == 1:
                ax.set_xlabel(r"$M_1$ [GeV]", fontsize=14)
            if col_i == 0:
                ax.set_ylabel(r"$|\eta_B|\times10^{10}$", fontsize=14)
            sign = "-" if row == "top" else "+"
            y3 = 30 if column == "left" else 0
            ax.text(0.04, 0.06, rf"$x_i:{sign},\ y_3={y3}^\circ$",
                    transform=ax.transAxes, fontsize=9)
            if row_i == 0 and col_i == 0:
                clean_legend(ax, fontsize=8)
    fig.suptitle("Reproduction of Fig. 1 (arXiv:2107.02079)", fontsize=14)
    fig.tight_layout()
    return fig, axes


def scan_paper_fig1_with_scattering(base_params: dict, m1_values, *,
                                    baseline_scans=None, progress=True):
    """Add scattering to Fig. 1, optionally reusing precomputed baseline scans."""
    masses = np.asarray(m1_values, dtype=float)
    scattering_solvers = (
        ("DME+sct", EtaB_1DME_sct),
        ("DME (Lambda_mu=0)+sct", EtaB_1DME_sct_LambdaMuZero),
        ("1BE1F+sct", EtaB_1BE1F_sct),
        ("1BE2F+sct", EtaB_1BE2F_sct),
        ("1BE3F+sct", EtaB_1BE3F_sct),
    )
    baseline_solvers = (
        ("DME", ulysses.EtaB_1DME),
        ("DME (Lambda_mu=0)", EtaB_1DME_LambdaMuZero),
        ("1BE1F", ulysses.EtaB_1BE1F),
        ("1BE2F", ulysses.EtaB_1BE2F),
        ("1BE3F", ulysses.EtaB_1BE3F),
    )
    result = {}
    panels = paper_fig1_points(base_params)
    total = len(panels) * len(masses)
    count = 0
    for panel, benchmark in panels.items():
        if baseline_scans is None:
            curves = {"M1_no_sct": masses.copy()}
            curves.update({name: np.empty_like(masses)
                           for name, _ in baseline_solvers})
        else:
            previous = baseline_scans[panel]
            curves = {"M1_no_sct": np.asarray(previous["M1"]).copy()}
            curves.update({name: np.asarray(previous[name]).copy()
                           for name, _ in baseline_solvers})
        curves["M1_sct"] = masses.copy()
        curves.update({name: np.empty_like(masses)
                       for name, _ in scattering_solvers})
        for i, mass in enumerate(masses):
            point = scale_heavy_masses(benchmark, mass)
            if baseline_scans is None:
                for name, cls in baseline_solvers:
                    model = cls()
                    model.setParams(point)
                    with redirect_stdout(StringIO()):
                        curves[name][i] = np.real(model.EtaB)
            for name, cls in scattering_solvers:
                model = cls()
                model.setParams(point)
                with redirect_stdout(StringIO()):
                    curves[name][i] = np.real(model.EtaB)
            count += 1
            if progress:
                print(f"{count:>3}/{total}  panel={panel}, M1={mass:.2e}", end="\r")
        result[panel] = curves
    if progress:
        print()
    return result


def plot_paper_fig1_with_scattering(scans: dict):
    """Four-panel overlay: dashed original, solid scattering extension."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.8), sharex=True, sharey=True,
                             facecolor="white")
    methods = (
        ("DME", "tab:blue"),
        ("DME (Lambda_mu=0)", "#62a0d1"),
        ("1BE1F", "tab:orange"),
        ("1BE2F", "tab:green"),
        ("1BE3F", "tab:red"),
    )
    for row_i, row in enumerate(("top", "bottom")):
        for col_i, column in enumerate(("left", "right")):
            ax = axes[row_i, col_i]
            scan = scans[(row, column)]
            mass_no_sct = scan["M1_no_sct"]
            mass_sct = scan["M1_sct"]
            for method, color in methods:
                ax.plot(mass_no_sct, np.abs(scan[method]) * 1e10,
                        color=color, ls="--", lw=1.7,
                        label=f"{method}, no sct")
                ax.plot(mass_sct, np.abs(scan[f"{method}+sct"]) * 1e10,
                        color=color, ls="-", lw=2.0,
                        label=f"{method}, +sct")
            ax.axhline(ETA_B_OBS * 1e10, color="0.55", lw=0.9)
            ax.axvline(1e12, color="0.25", lw=0.9)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e8, 1e14)
            ax.set_ylim(1e-3, 1e5)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if row_i == 1:
                ax.set_xlabel(r"$M_1$ [GeV]", fontsize=14)
            if col_i == 0:
                ax.set_ylabel(r"$|\eta_B|\times10^{10}$", fontsize=14)
            sign = "-" if row == "top" else "+"
            y3 = 30 if column == "left" else 0
            ax.text(0.04, 0.05, rf"$x_i:{sign},\ y_3={y3}^\circ$",
                    transform=ax.transAxes, fontsize=8.5)
            if row_i == 0 and col_i == 0:
                clean_legend(ax, fontsize=7.0, ncol=2)
    fig.suptitle("Fig. 1 benchmarks with and without scattering", fontsize=14)
    fig.text(0.5, 0.005, "Dashed: original (no scattering); solid: with scattering",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    return fig, axes


def _format_fig1_panel(ax, row, column, *, row_i, col_i):
    """Shared axes formatting for the split scattering figures."""
    ax.axhline(ETA_B_OBS * 1e10, color="0.55", lw=0.9)
    ax.axvline(1e12, color="0.25", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e8, 1e14)
    ax.set_ylim(1e-3, 1e5)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if row_i == 1:
        ax.set_xlabel(r"$M_1$ [GeV]", fontsize=14)
    if col_i == 0:
        ax.set_ylabel(r"$|\eta_B|\times10^{10}$", fontsize=14)
    sign = "-" if row == "top" else "+"
    y3 = 30 if column == "left" else 0
    ax.text(0.04, 0.05, rf"$x_i:{sign},\ y_3={y3}^\circ$",
            transform=ax.transAxes, fontsize=8.5)


def _figure_legend(fig, axes, *, ncol):
    """Place one deduplicated legend above a four-panel figure."""
    handles, labels = [], []
    seen = set()
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               frameon=False, ncol=ncol, fontsize=8.5)


def plot_paper_fig1_scattering_split(scans: dict):
    """Return separate four-panel DME and Boltzmann scattering figures."""
    fig_dme, axes_dme = plt.subplots(
        2, 2, figsize=(8.6, 7.5), sharex=True, sharey=True, facecolor="white"
    )
    fig_be, axes_be = plt.subplots(
        2, 2, figsize=(8.6, 7.5), sharex=True, sharey=True, facecolor="white"
    )
    dme_methods = (
        ("DME", "tab:blue"),
        ("DME (Lambda_mu=0)", "#62a0d1"),
    )
    be_methods = (
        ("1BE1F", "tab:orange"),
        ("1BE2F", "tab:green"),
        ("1BE3F", "tab:red"),
    )
    for row_i, row in enumerate(("top", "bottom")):
        for col_i, column in enumerate(("left", "right")):
            scan = scans[(row, column)]
            mass_plain, mass_sct = scan["M1_no_sct"], scan["M1_sct"]
            ax_dme, ax_be = axes_dme[row_i, col_i], axes_be[row_i, col_i]
            for method, color in dme_methods:
                ax_dme.plot(mass_plain, np.abs(scan[method]) * 1e10,
                            color=color, ls="--", lw=1.8,
                            label=f"{method}, no scattering")
                ax_dme.plot(mass_sct, np.abs(scan[f"{method}+sct"]) * 1e10,
                            color=color, ls="-", lw=2.1,
                            label=f"{method}, with scattering")
            for method, color in be_methods:
                ax_be.plot(mass_plain, np.abs(scan[method]) * 1e10,
                           color=color, ls="--", lw=1.8,
                           label=f"{method}, no scattering")
                ax_be.plot(mass_sct, np.abs(scan[f"{method}+sct"]) * 1e10,
                           color=color, ls="-", lw=2.1,
                           label=f"{method}, with scattering")
            _format_fig1_panel(ax_dme, row, column, row_i=row_i, col_i=col_i)
            _format_fig1_panel(ax_be, row, column, row_i=row_i, col_i=col_i)

    fig_dme.suptitle("Density-matrix equations: scattering comparison",
                     fontsize=14, y=0.995)
    fig_be.suptitle("Boltzmann equations: scattering comparison",
                    fontsize=14, y=0.995)
    _figure_legend(fig_dme, axes_dme, ncol=2)
    _figure_legend(fig_be, axes_be, ncol=3)
    for fig in (fig_dme, fig_be):
        fig.text(0.5, 0.008, "Dashed: no scattering; solid: with scattering",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.035, 1, 0.88))
    return (fig_dme, axes_dme), (fig_be, axes_be)


@dataclass
class I2Result:
    kappa1: np.ndarray
    vanilla_via: np.ndarray
    vanilla_tia: np.ndarray
    scattering_via: np.ndarray
    scattering_tia: np.ndarray


def _i2_at_kappa(kappa, z, d_unit, w_unit, neq, ss_unit, st_unit,
                 *, thermal, scattering):
    """Integrate Eqs. (49), (57), (59), (61) as four coupled ODEs."""
    def interp(values, x):
        return np.interp(np.log(x), np.log(z), values)

    def rhs(x, state):
        n1, coherence, i1, i2 = state
        neq_x = interp(neq, x)
        d = kappa * interp(d_unit, x)
        w = kappa * interp(w_unit, x)
        if scattering:
            ss = kappa * interp(ss_unit, x)
            st = kappa * interp(st_unit, x)
            d_total = d + ss + st
            w_total = washout_sct(w, d, n1, neq_x, ss, st)
        else:
            d_total, w_total = d, w
        # Scattering changes N1 production and washout, not the CP source.
        departure = n1 - neq_x
        return (-d_total * departure,
                d * departure - 0.5 * w_total * coherence,
                coherence,
                w_total * (i1 - i2))

    initial_n1 = neq[0] if thermal else 0.0
    solution = solve_ivp(rhs, (z[0], z[-1]), [initial_n1, 0.0, 0.0, 0.0],
                         method="LSODA", rtol=2e-7, atol=1e-10)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[3, -1]


def scan_i2(kappa_values=None, *, m1_gev=1e12, grid_size=360) -> I2Result:
    """Reproduce Fig. 2 and repeat it with the repository's scatterings."""
    kappas = np.asarray(
        np.r_[np.linspace(0.0, 10.0, 61), np.linspace(11.0, 100.0, 90)]
        if kappa_values is None else kappa_values, dtype=float
    )
    z = np.geomspace(1e-3, 1e3, grid_size)
    base = ulysses.ULSBase(zmin=z[0], zmax=z[-1], zsteps=grid_size)
    d_unit = np.array([np.real(base.D1(1.0, x)) for x in z])
    w_unit = np.array([np.real(base.W1(1.0, x)) for x in z])
    neq = np.array([base.N1Eq(x) for x in z])
    # Rates are linear in kappa.  Evaluate the expensive thermal integrals once.
    ss_unit = np.array([scat_Ss(1.0, x) for x in z])
    st_unit = np.array([scat_St(1.0, x, m1_gev, base.MH) for x in z])

    curves = []
    for scattering in (False, True):
        for thermal in (False, True):
            curves.append(np.array([
                _i2_at_kappa(k, z, d_unit, w_unit, neq, ss_unit, st_unit,
                             thermal=thermal, scattering=scattering)
                for k in kappas
            ]))
    return I2Result(kappas, curves[0], curves[1], curves[2], curves[3])


def plot_i2(result: I2Result):
    """Reproduce paper Fig. 2 without scattering."""
    fig, ax = plt.subplots(figsize=(7.2, 5.3), facecolor="white")
    ax.plot(result.kappa1, result.vanilla_via, color="tab:blue", lw=2,
            label=r"$N_{N_1}(z_0)=0$ (VIA)")
    ax.plot(result.kappa1, result.vanilla_tia, color="tab:orange", lw=2,
            label=r"$N_{N_1}(z_0)=N_{N_1}^{\rm eq}(z_0)$ (TIA)")
    ax.axhline(0.0, color="black", lw=1.5, ls="--", zorder=1)
    ax.axhline(-0.13, color="0.55", lw=1.2, zorder=1)
    style_axes(ax, xlim=(0, 100), ylim=(-1.0, 1.5),
               xlabel=r"$\kappa_1$", ylabel=r"$I_2(\kappa_1;z_f)$",
               title=r"Reproduction of Fig. 2 ($z_f=1000$)")
    ax.set_xticks(np.r_[0, 5, np.arange(10, 101, 10)])
    clean_legend(ax, fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_i2_scattering(result: I2Result):
    """Separate extension of Fig. 2 including Delta-L=1 scattering."""
    fig, ax = plt.subplots(figsize=(7.2, 5.3), facecolor="white")
    ax.plot(result.kappa1, result.vanilla_via, color="tab:blue", lw=2,
            ls="--", label="VIA, no scattering")
    ax.plot(result.kappa1, result.vanilla_tia, color="tab:orange", lw=2,
            ls="--", label="TIA, no scattering")
    ax.plot(result.kappa1, result.scattering_via, color="tab:blue", lw=2,
            ls="-", label="VIA + scattering")
    ax.plot(result.kappa1, result.scattering_tia, color="tab:orange", lw=2,
            ls="-", label="TIA + scattering")
    ax.axhline(0, color="0.2", lw=1)
    ax.axhline(-0.13, color="0.55", lw=1.2, ls=":")
    style_axes(ax, xlim=(0, 100), ylim=(-1.0, 1.5),
               xlabel=r"$\kappa_1$", ylabel=r"$I_2(\kappa_1;z_f)$",
               title=r"Extension of Fig. 2: $\Delta L=1$ scattering")
    ax.set_xticks(np.arange(0, 101, 10))
    clean_legend(ax, fontsize=10, ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_i2_side_by_side(result: I2Result):
    """Compare paper Fig. 2 directly with its scattering extension."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, sharey=True,
                             facecolor="white")
    panels = (
        (axes[0], result.vanilla_via, result.vanilla_tia,
         "Original: no scattering"),
        (axes[1], result.scattering_via, result.scattering_tia,
         r"With $\Delta L=1$ scattering"),
    )
    for ax, via, tia, title in panels:
        ax.plot(result.kappa1, via, color="tab:blue", lw=2.2,
                label=r"$N_{N_1}(z_0)=0$ (VIA)")
        ax.plot(result.kappa1, tia, color="tab:orange", lw=2.2,
                label=r"$N_{N_1}(z_0)=N_{N_1}^{\rm eq}(z_0)$ (TIA)")
        ax.axhline(0.0, color="black", lw=1.4, ls="--", zorder=1)
        ax.axhline(-0.13, color="0.55", lw=1.2, zorder=1)
        style_axes(ax, xlim=(0, 100), ylim=(-1.0, 1.5),
                   xlabel=r"$\kappa_1$", title=title, ticksize=13)
        ax.set_xticks(np.r_[0, 5, np.arange(10, 101, 10)])
        clean_legend(ax, fontsize=8.5)
    axes[0].set_ylabel(r"$I_2(\kappa_1;z_f)$", fontsize=18)
    fig.suptitle(r"Sign of $I_2$ at $z_f=1000$: scattering comparison",
                 fontsize=14)
    fig.tight_layout()
    return fig, axes
