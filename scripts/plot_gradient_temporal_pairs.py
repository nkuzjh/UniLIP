#!/usr/bin/env python3
"""Plot paired checkpoint gradient observations without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {0: "#0072B2", 128: "#D55E00", 256: "#009E73"}
STYLES = {17: ("o", "-"), 29: ("^", "--")}
EXPERIMENTS = ["exp32_3maps_B1_S3_J8", "exp32_3maps_B2_S3_J8"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Preview only; marked incomplete")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    if not data.get("complete") and not args.allow_partial:
        raise ValueError("Complete 12-pair analysis required; outputs untouched")
    pairs = data["pairs"]
    keys = [(r["experiment"], r["source_offset"], r["noise_seed"]) for r in pairs]
    expected = {(e, o, s) for e in EXPERIMENTS for o in COLORS for s in STYLES}
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise ValueError("Duplicate or unexpected pair identities")
    if data.get("complete") and set(keys) != expected:
        raise ValueError("Complete analysis does not contain the expected 12 pairs")
    plotted = []
    for row in pairs:
        for task in ("gen", "loc"):
            field = f"{task}_aux_effect_vs_main_descent_pct"
            values = [row[f"step{step}"][field] for step in (1200, 1800)]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError("Undefined auxiliary effect: inspect the main-direction condition before plotting")
            plotted.append({"experiment": row["experiment"], "source_offset": row["source_offset"],
                            "noise_seed": row["noise_seed"], "task": task, "values": values})
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for i, experiment in enumerate(EXPERIMENTS):
        for j, task in enumerate(("gen", "loc")):
            ax = axes[i, j]
            for row in plotted:
                if row["experiment"] != experiment or row["task"] != task:
                    continue
                marker, linestyle = STYLES[row["noise_seed"]]
                ax.plot([1200, 1800], row["values"], color=COLORS[row["source_offset"]],
                        marker=marker, linestyle=linestyle, linewidth=1.4, markersize=5, alpha=.85)
            ax.axhline(0, color="#444444", linewidth=.8)
            ax.set(title=f"B{i + 1}: {'generation' if task == 'gen' else 'localization'}",
                   xticks=[1200, 1800], xlim=(1100, 1900), xlabel="Stage 2 checkpoint",
                   ylabel="Auxiliary effect / main descent (%)")
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=color, label=f"Window offset {offset}", linewidth=2)
               for offset, color in COLORS.items()]
    handles += [Line2D([], [], color="#555555", marker=m, linestyle=ls,
                       label=f"Diagnostic seed {seed}") for seed, (m, ls) in STYLES.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    status = "Paired 128-source windows" if data.get("complete") else f"PARTIAL PREVIEW: {len(pairs)}/12 pairs"
    fig.suptitle(status + ": step 1200 to 1800", y=.995)
    fig.text(.5, .015, "Positive: offsets descent; negative: helps descent. Local gradient estimates, not evaluation-metric changes.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .865), h_pad=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output_dir / f"temporal_auxiliary_effect.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    provenance = {"source": str(args.input), "source_sha256": hashlib.sha256(raw).hexdigest(),
                  "complete": data.get("complete", False), "pairs": len(pairs), "plotted": plotted,
                  "interpretation": "Paired window/seed observations; not independent training repeats or a pooled 384-source gradient."}
    (args.output_dir / "temporal_plot_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Rendered {len(pairs)} pairs; complete={data.get('complete', False)}")


if __name__ == "__main__":
    main()
