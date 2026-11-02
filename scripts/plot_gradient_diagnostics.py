#!/usr/bin/env python3
"""Plot the six checkpoint-1800 FP32 confirm gradient diagnostics (CPU only)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np


EXPERIMENTS = (
    ("S0_J8", "exp32_3maps_stage2_control_j8"),
    ("P3", "exp32_3maps_P3_S1_J8"),
    ("P4", "exp32_3maps_P4_S1_J8"),
    ("U2", "exp32_3maps_U2_S2_J8"),
    ("B1", "exp32_3maps_B1_S3_J8"),
    ("B2", "exp32_3maps_B2_S3_J8"),
)
OFFSETS = (0, 128, 256)
SEEDS = (17, 29)
LAYERS = tuple(range(24))
WINDOW_COLORS = {0: "#2376b7", 128: "#e38924", 256: "#36966a"}
SEED_MARKERS = {17: "o", 29: "^"}


def finite(value, where):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: expected finite number, got {value!r}")
    return float(value)


def parse_observation(result, label):
    shared = result["shared_language_layers"]
    layers = shared["layers"]
    if set(layers) != {str(i) for i in LAYERS}:
        raise ValueError(f"{label}: expected layers 0..23, found {sorted(layers)}")
    denominator = finite(shared["global_alignment_denominator"], f"{label}.denominator")
    if denominator <= 0:
        raise ValueError(f"{label}: nonpositive alignment denominator")
    global_dot = finite(shared["global_shared_weighted_gen_loc_dot"], f"{label}.global_dot")
    cosine = global_dot / denominator
    reported_cosine = finite(
        result["groups"]["shared_language"]["pairwise_weighted_cosine"]["gen"]["loc"],
        f"{label}.reported_cosine",
    )
    np.testing.assert_allclose(cosine, reported_cosine, rtol=1e-9, atol=1e-12)
    layer_cosines = [finite(layers[str(i)]["weighted_gen_loc_cosine"], f"{label}.layer{i}.cos") for i in LAYERS]
    layer_contributions = [
        finite(layers[str(i)]["weighted_gen_loc_dot"], f"{label}.layer{i}.dot") / denominator
        for i in LAYERS
    ]
    # Every layer contributes to the numerator, with one common global denominator.
    np.testing.assert_allclose(sum(layer_contributions), cosine, rtol=1e-8, atol=1e-11)
    effects = {}
    for task in ("gen", "loc"):
        direction = result["combined"]["primary_directions"][task]
        total = finite(direction["raw_descent_first_order_total"], f"{label}.{task}.total")
        aux = finite(direction["raw_descent_first_order_aux_increment"], f"{label}.{task}.aux")
        main = total - aux
        effects[task] = {
            "raw_descent_first_order_total": total,
            "raw_descent_first_order_aux_increment": aux,
            "main_raw_descent_first_order": main,
            "aux_effect_vs_main_descent_pct": 100 * aux / abs(main) if main < 0 else None,
        }
    return {
        "global_gen_loc_cosine": cosine,
        "global_alignment_denominator": denominator,
        "layer_weighted_gen_loc_cosine": layer_cosines,
        "layer_global_cosine_contribution": layer_contributions,
        "effects": effects,
    }


def collect(input_root):
    rows = []
    for short, experiment in EXPERIMENTS:
        observations = []
        missing = []
        for offset in OFFSETS:
            path = input_root / "confirm" / f"{experiment}.step1800.offset{offset}.json"
            if not path.is_file():
                missing.extend({"offset": offset, "noise_seed": seed, "source_file": str(path)} for seed in SEEDS)
                continue
            with path.open(encoding="utf-8") as handle:
                source = json.load(handle)
            metadata = source["metadata"]
            expected = {
                "experiment": experiment,
                "checkpoint_step": 1800,
                "source_offset": offset,
                "stage": "effective",
                "measurement_mode": "component_sum",
                "coordinate": "shared",
                "noise_seeds": list(SEEDS),
            }
            for field, wanted in expected.items():
                actual = metadata[field] if field in ("experiment", "checkpoint_step") else source[field]
                if actual != wanted:
                    raise ValueError(f"{path}: {field}={actual!r}, expected {wanted!r}")
            if metadata.get("precision_reference") is not True or metadata.get("probe_batch_float_dtype") != "float32":
                raise ValueError(f"{path}: not an FP32 reference observation")
            if len(source["source_indices"]) != 128:
                raise ValueError(f"{path}: expected exactly 128 source indices")
            if set(source["results"]) != {str(seed) for seed in SEEDS}:
                raise ValueError(f"{path}: expected exactly noise seeds 17 and 29")
            for seed in SEEDS:
                result = source["results"][str(seed)]
                parsed = parse_observation(result, f"{experiment}.offset{offset}.seed{seed}")
                observations.append({"offset": offset, "noise_seed": seed, "source_file": str(path.resolve()), **parsed})
        complete = len(observations) == len(OFFSETS) * len(SEEDS)
        row = {
            "label": short,
            "experiment": experiment,
            "checkpoint_step": 1800,
            "complete": complete,
            "observation_count": len(observations),
            "missing": missing,
            "observations": observations,
        }
        if complete:
            cosines = np.asarray([obs["layer_weighted_gen_loc_cosine"] for obs in observations])
            contributions = np.asarray([obs["layer_global_cosine_contribution"] for obs in observations])
            row["mean_layer_weighted_gen_loc_cosine"] = cosines.mean(axis=0).tolist()
            row["mean_layer_global_cosine_contribution"] = contributions.mean(axis=0).tolist()
            row["mean_global_gen_loc_cosine"] = float(np.mean([obs["global_gen_loc_cosine"] for obs in observations]))
            np.testing.assert_allclose(
                sum(row["mean_layer_global_cosine_contribution"]),
                row["mean_global_gen_loc_cosine"],
                rtol=1e-8, atol=1e-11,
            )
        rows.append(row)
    return rows


def save_figure(fig, output_dir, stem):
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def heatmap(rows, output_dir):
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=(15.2, 8.8), layout="constrained")
    labels = [f"{row['label']}  n=6" if row["complete"] else f"{row['label']}  pending ({row['observation_count']}/6)" for row in rows]
    fields = (
        ("mean_layer_weighted_gen_loc_cosine", "Layer weighted gen–loc cosine", "mean cosine"),
        ("mean_layer_global_cosine_contribution", "Layer contribution to global gen–loc cosine", "mean contribution"),
    )
    cmap = plt.get_cmap("RdBu").copy()  # red negative, blue positive
    cmap.set_bad("#d7d7d7")
    for ax, (field, title, color_label) in zip(axes, fields):
        matrix = np.asarray([row[field] if row["complete"] else [np.nan] * 24 for row in rows], dtype=float)
        valid = matrix[np.isfinite(matrix)]
        bound = float(max(np.max(np.abs(valid)), 1e-12)) if valid.size else 1.0
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
        ax.set_yticks(range(len(rows)), labels=labels)
        ax.set_xticks(range(24), labels=[str(i) for i in LAYERS])
        ax.set_xlabel("Shared language layer")
        ax.set_title(title, fontsize=13, pad=9)
        ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        cbar = fig.colorbar(image, ax=ax, shrink=.83, pad=.015)
        cbar.set_label(color_label)
        for i, row in enumerate(rows):
            if not row["complete"]:
                ax.text(11.5, i, "pending", ha="center", va="center", color="#555555", fontsize=11)
    fig.suptitle("FP32 reference · 128-source windows · mean across windows and noise · checkpoint 1800", fontsize=15)
    save_figure(fig, output_dir, "layer_alignment_heatmap")


def scatter(rows, output_dir):
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.1))
    fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.27, wspace=.25)
    metrics = (
        ("global_gen_loc_cosine", "Global gen–loc cosine", "cosine"),
        ("gen", "Gen: aux effect on main descent", "% of |main descent|"),
        ("loc", "Loc: aux effect on main descent", "% of |main descent|"),
    )
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for i, row in enumerate(rows):
            if not row["complete"]:
                continue
            values = []
            for obs in row["observations"]:
                value = obs["global_gen_loc_cosine"] if metric == "global_gen_loc_cosine" else obs["effects"][metric]["aux_effect_vs_main_descent_pct"]
                if value is None:
                    continue
                values.append(value)
                # Offsets are deliberately separated; seed symbols have fixed, small jitter.
                jitter = (SEEDS.index(obs["noise_seed"]) - .5) * .055
                x = i + (OFFSETS.index(obs["offset"]) - 1) * .14 + jitter
                ax.scatter(x, value, c=WINDOW_COLORS[obs["offset"]], marker=SEED_MARKERS[obs["noise_seed"]], s=58, edgecolors="white", linewidths=.6, zorder=3)
            if values:
                median = float(np.median(values))
                ax.plot([i - .095, i + .095], [median, median], color="black", linewidth=2, zorder=4)
        ax.axhline(0, color="#555555", linewidth=.85, zorder=1)
        ax.set_xticks(range(len(rows)), [row["label"] if row["complete"] else f"{row['label']}\npending" for row in rows])
        ax.set_xlim(-.55, len(rows) - .45)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12.5)
        ax.grid(axis="y", color="#e4e4e4", linewidth=.6)
        ax.set_axisbelow(True)
    legend = [Line2D([], [], marker="o", linestyle="", color=color, label=f"offset {offset}") for offset, color in WINDOW_COLORS.items()]
    legend.extend(Line2D([], [], marker=marker, linestyle="", color="#555555", label=f"noise {seed}") for seed, marker in SEED_MARKERS.items())
    legend.append(Line2D([], [], color="black", linewidth=2, label="median"))
    fig.legend(handles=legend, loc="lower center", ncol=6, bbox_to_anchor=(.5, .025), frameon=False)
    fig.text(.5, .945, "FP32 reference · 128-source windows · checkpoint 1800 · each point = one window × noise", ha="center", fontsize=14)
    fig.text(.5, .14, "Aux effect = 100 × aux increment / |main descent| when main descent < 0; positive offsets main descent", ha="center", fontsize=10)
    save_figure(fig, output_dir, "gradient_observations_scatter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs_diagnostics/gradient_seen3_fp32"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_diagnostics/gradient_seen3_fp32_figures"))
    parser.add_argument("--require-complete", action="store_true", help="Fail unless all six models have all three windows and both noise seeds.")
    args = parser.parse_args()
    rows = collect(args.input_root)
    incomplete = [row["label"] for row in rows if not row["complete"]]
    if args.require_complete and incomplete:
        parser.error("incomplete confirm observations: " + ", ".join(incomplete))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "FP32 reference, checkpoint 1800, 128-source confirm windows; six observations per complete model",
        "input_root": str(args.input_root.resolve()),
        "offsets": list(OFFSETS),
        "noise_seeds": list(SEEDS),
        "layers": list(LAYERS),
        "aux_effect_definition": "100 * raw_descent_first_order_aux_increment / abs(raw_descent_first_order_total - raw_descent_first_order_aux_increment), only if the main term is negative; positive means it offsets the main descent",
        "rows": rows,
    }
    (args.output_dir / "gradient_plot_data.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    heatmap(rows, args.output_dir)
    scatter(rows, args.output_dir)
    print(json.dumps({"complete_models": [row["label"] for row in rows if row["complete"]], "incomplete_models": incomplete, "observations": sum(row["observation_count"] for row in rows), "missing_groups": sum(len(row["missing"]) for row in rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
