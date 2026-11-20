"""Generate the paper figures and LaTeX tables from measurements_all.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/paper_assets")
OUT.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["van_der_pol", "rossler", "halvorsen", "lorenz", "chen"]
LABELS = {
    "van_der_pol": "van der Pol",
    "rossler": "Rossler",
    "halvorsen": "Halvorsen",
    "lorenz": "Lorenz",
    "chen": "Chen",
}
PARAMS = {
    "chronos-bolt-tiny": 9e6,
    "chronos-bolt-mini": 21e6,
    "chronos-bolt-small": 48e6,
    "chronos-bolt-base": 205e6,
}
STYLE = {
    "delay_embedding": ("#1f4e79", "o", "-", 2.0, "delay embedding"),
    "persistence": ("#7f7f7f", "s", "--", 1.4, "persistence"),
    "chronos-bolt-tiny": ("#f4a582", "^", "-", 1.2, "Chronos-Bolt tiny (9M)"),
    "chronos-bolt-mini": ("#e06c5a", "^", "-", 1.2, "Chronos-Bolt mini (21M)"),
    "chronos-bolt-small": ("#ca0020", "^", "-", 1.2, "Chronos-Bolt small (48M)"),
    "chronos-bolt-base": ("#8b0016", "^", "-", 1.2, "Chronos-Bolt base (205M)"),
}

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "figure.dpi": 200})


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.5)


def save(figure, stem):
    for suffix in ("pdf", "png"):
        figure.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(figure)


def figure_precision_sweep(frame):
    figure, axes = plt.subplots(1, 5, figsize=(13.0, 2.9), sharex=True)
    for axis, system in zip(axes, SYSTEMS):
        subset = frame[frame.system == system]
        floor = float(subset.ks_entropy.iloc[0])
        for predictor, (colour, marker, line, width, label) in STYLE.items():
            group = subset[subset.predictor == predictor].sort_values("precision", ascending=False)
            axis.plot(
                1.0 / group.precision,
                group.information_rate,
                color=colour,
                marker=marker,
                linestyle=line,
                linewidth=width,
                markersize=3.5,
                label=label,
            )
        axis.axhline(max(floor, 1e-3), color="black", linestyle=":", linewidth=1.3)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(f"{LABELS[system]}  ($h_{{KS}}={floor:.2f}$)", fontsize=8)
        axis.set_xlabel("inverse precision $1/\\delta$")
        style_axis(axis)
    axes[0].set_ylabel("information rate (nats/time)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.09))
    figure.tight_layout()
    save(figure, "fig1_precision_sweep")


def figure_scaling(frame):
    chronos = frame[frame.predictor.isin(PARAMS)].copy()
    chronos["logp"] = np.log10(chronos.predictor.map(PARAMS))

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))

    for index, system in enumerate(SYSTEMS):
        group = chronos[chronos.system == system].groupby("logp", as_index=False).coding_gap.median()
        axes[0].plot(
            10 ** group.logp,
            group.coding_gap,
            marker="o",
            markersize=4,
            linewidth=1.3,
            label=LABELS[system],
            color=plt.cm.viridis(index / 4.0),
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("parameters")
    axes[0].set_ylabel("excess bits over floor (nats/time)")
    axes[0].set_title("Excess information does not fall with scale", fontsize=8)
    axes[0].legend(frameon=False, fontsize=6)
    style_axis(axes[0])

    fits = {}
    for system in SYSTEMS:
        group = chronos[chronos.system == system]
        slope, intercept = np.polyfit(group.logp, group.coding_gap, 1)
        fits[system] = {"slope": float(slope), "zero_at_log10": float((0.0 - intercept) / slope)}
    slope, intercept = np.polyfit(chronos.logp, chronos.coding_gap, 1)
    fits["pooled"] = {"slope": float(slope), "zero_at_log10": float((0.0 - intercept) / slope)}

    observed = chronos.groupby("logp", as_index=False).coding_gap.median()
    axes[1].scatter(observed.logp, observed.coding_gap, color="#ca0020", s=26, zorder=3, label="observed")
    span = np.linspace(6.5, fits["pooled"]["zero_at_log10"] + 0.5, 200)
    axes[1].plot(span, slope * span + intercept, color="#1f4e79", linewidth=1.4, label="pooled fit")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.2)
    axes[1].axvline(fits["pooled"]["zero_at_log10"], color="grey", linestyle="--", linewidth=1.0)
    axes[1].annotate(
        f"zero excess at $10^{{{fits['pooled']['zero_at_log10']:.1f}}}$ parameters",
        xy=(fits["pooled"]["zero_at_log10"], 0.0),
        xytext=(fits["pooled"]["zero_at_log10"] - 8.0, float(slope) * 8 + float(intercept) * 0.5),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "linewidth": 0.7},
    )
    axes[1].set_xlabel("$\\log_{10}$ parameters")
    axes[1].set_ylabel("excess bits over floor (nats/time)")
    axes[1].set_title("Extrapolation to the floor", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7)
    style_axis(axes[1])

    figure.tight_layout()
    save(figure, "fig2_scaling")
    return fits


def precision_slopes(frame):
    rows = {}
    for predictor in STYLE:
        rows[predictor] = {
            system: float(np.polyfit(np.log2(1.0 / group.precision), group.information_rate, 1)[0])
            for system, group in frame[frame.predictor == predictor].groupby("system")
        }
    return pd.DataFrame(rows).T[SYSTEMS]


def latex_tables(frame, fits):
    excess = frame.pivot_table(index="predictor", columns="system", values="coding_gap", aggfunc="median")
    excess = excess.loc[list(STYLE), SYSTEMS]
    slopes = precision_slopes(frame).loc[list(STYLE)]
    savings = frame.groupby("predictor").forward_pass_saving.median()
    entropies = frame.groupby("system").ks_entropy.first()[SYSTEMS]

    lines = [
        "% Table 1: median excess bits over the information floor",
        "\\begin{tabular}{l" + "r" * len(SYSTEMS) + "}",
        "\\toprule",
        "Predictor & " + " & ".join(LABELS[s] for s in SYSTEMS) + " \\\\",
        "$h_{KS}$ (nats/time) & " + " & ".join(f"{entropies[s]:.3f}" for s in SYSTEMS) + " \\\\",
        "\\midrule",
    ]
    for predictor in STYLE:
        values = " & ".join(f"{excess.loc[predictor, s]:.2f}" for s in SYSTEMS)
        lines.append(f"{STYLE[predictor][4]} & {values} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", ""]

    lines += [
        "% Table 2: precision slope and forward-pass saving",
        "\\begin{tabular}{l" + "r" * len(SYSTEMS) + "r}",
        "\\toprule",
        "Predictor & " + " & ".join(LABELS[s] for s in SYSTEMS) + " & Fwd. saving \\\\",
        "\\midrule",
    ]
    for predictor in STYLE:
        values = " & ".join(f"{slopes.loc[predictor, s]:.2f}" for s in SYSTEMS)
        lines.append(f"{STYLE[predictor][4]} & {values} & {savings[predictor]:.1f}$\\times$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", ""]

    lines += [
        "% Table 3: scaling fit",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "System & Slope (nats/time per $10\\times$) & Zero excess at \\\\",
        "\\midrule",
    ]
    for system in SYSTEMS:
        lines.append(
            f"{LABELS[system]} & {fits[system]['slope']:.2f} & $10^{{{fits[system]['zero_at_log10']:.1f}}}$ \\\\"
        )
    lines += [
        "\\midrule",
        f"Pooled & {fits['pooled']['slope']:.2f} & $10^{{{fits['pooled']['zero_at_log10']:.1f}}}$ \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]

    (OUT / "tables.tex").write_text("\n".join(lines), encoding="utf-8")
    excess.to_csv(OUT / "table1_excess_bits.csv")
    slopes.to_csv(OUT / "table2_precision_slopes.csv")


def headline(frame, fits):
    excess = frame.pivot_table(index="predictor", columns="system", values="coding_gap", aggfunc="median")
    wins = sum(excess.loc["persistence", s] < min(excess.loc[p, s] for p in PARAMS) for s in SYSTEMS)
    slopes = precision_slopes(frame)
    delay = frame[frame.predictor == "delay_embedding"]
    report = [
        f"persistence beats every Chronos size on {wins}/{len(SYSTEMS)} systems",
        f"pooled scaling slope {fits['pooled']['slope']:.2f} nats/time per decade of parameters",
        f"extrapolated zero excess at 10^{fits['pooled']['zero_at_log10']:.1f} parameters",
        "per-system extrapolation range 10^%.1f to 10^%.1f"
        % (
            min(fits[s]["zero_at_log10"] for s in SYSTEMS),
            max(fits[s]["zero_at_log10"] for s in SYSTEMS),
        ),
        f"delay embedding precision slope negative on {int((slopes.loc['delay_embedding'] < 0).sum())}/5 systems, zero on the limit cycle",
        f"every Chronos precision slope is positive on all 5 systems",
        f"delay embedding excess on van der Pol: {excess.loc['delay_embedding', 'van_der_pol']:.4f} nats/time (floor is 0.0000)",
        f"delay embedding forward-pass saving {delay.forward_pass_saving.median():.1f}x median, "
        f"{delay[delay.system == 'van_der_pol'].forward_pass_saving.max():.0f}x on the limit cycle",
        f"Chronos forward-pass saving {frame[frame.predictor.isin(PARAMS)].forward_pass_saving.median():.2f}x median",
    ]
    text = "\n".join(f"- {line}" for line in report)
    (OUT / "headline_numbers.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    frame = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/measurements_all.csv")
    figure_precision_sweep(frame)
    fits = figure_scaling(frame)
    latex_tables(frame, fits)
    headline(frame, fits)
    print(f"\nwritten to {OUT.resolve()}")


if __name__ == "__main__":
    main()
