#!/usr/bin/env python3
"""Analyze Experiment 1: four GKNN regularization conditions.

    python plot_exp1.py --root results/exp1 --layer 1 2
        [--kind raw scaled pooled_raw pooled_scaled]
        [--select-by val_cross_entropy|val_accuracy]  [--weight logdet=1 ...]

Reads exported validation diagnostics; it never trains or touches test data.

1. Weight selection. Each method's lambda is chosen separately on validation
   (default: lowest mean validation cross-entropy, ties -> higher accuracy),
   among weights that cover the largest set of fold/seed runs. ``--weight``
   overrides the choice. The tuning table lists every (method, lambda).
2. Comparison. With the selected lambdas, only fold/seed combinations where
   ALL methods are present enter summaries and figures (same initial weights
   and same validation graphs are verified).

Because lambda is selected on the same validation graphs that are then
summarized, validation numbers are optimistic for every regularized method;
final claims need the held-out test folds. Error bars are sample SDs across
matched runs, not confidence intervals. Higher diversity is reported next to
accuracy; no relation between the two is assumed.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ("none", "jsd_code", "jsd_paper", "logdet")
LABELS = {"none": "No regularizer", "jsd_code": "JSD (code)",
          "jsd_paper": "JSD (paper)", "logdet": "LogDet (ours)"}
SHORT = {"none": "None", "jsd_code": "JSD code", "jsd_paper": "JSD paper", "logdet": "LogDet"}
COLORS = {"none": "#6B6B6B", "jsd_code": "#D98032", "jsd_paper": "#3477AD", "logdet": "#B8336A"}
KINDS = ("raw", "scaled", "pooled_raw", "pooled_scaled")
KIND_TEXT = {
    "raw": "node level, raw GKernel responses",
    "scaled": "node level, responses x mask gain",
    "pooled_raw": "graph level, add-pooled raw responses",
    "pooled_scaled": "graph level, add-pooled scaled responses (MLP input)",
}
METRICS = (
    "mean_abs_corr", "mean_sq_corr", "effective_rank", "participation_rank",
    "rank_fraction_total", "active_filters", "constant_filters",
    "val_accuracy_percent", "val_cross_entropy", "train_seconds", "mean_mask_wl",
)


def number(value):
    """Unavailable values remain unavailable instead of becoming zero."""
    if value is None:
        return float("nan")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Diagnostics contain a nonfinite numeric value.")
    return result


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def finite_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def write_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            clean = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                clean[key] = value
            writer.writerow(clean)


# --------------------------------------------------------------------------- reading
def read_runs(root, layer, kind):
    runs = []
    for path in sorted(root.rglob("diagnostics.json")):
        relative = path.relative_to(root)
        if any(part.startswith("analysis") for part in relative.parts[:-1]):
            continue
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        metadata = document["metadata"]
        method = metadata["reg_type"]
        if method not in METHODS:
            raise ValueError(f"Unknown reg_type {method!r} in {path}.")
        selected = [item for item in document["layers"] if item["layer"] == layer]
        if len(selected) != 1:
            raise ValueError(f"Expected one layer {layer} in {path}.")
        if kind not in selected[0]:
            raise ValueError(f"View {kind!r} missing in {path}; rerun run_exp1.py "
                             "with the current exp1_diagnostics.py.")
        stats = selected[0][kind]
        classification = document["classification"]
        for required in ("protocol_id", "initial_state_sha256", "dataset", "seed", "fold"):
            if not metadata.get(required) and metadata.get(required) != 0:
                raise ValueError(f"Missing metadata field {required!r} in {path}.")
        run = {
            "dataset": metadata["dataset"], "reg_type": method,
            "jsd_weight": float(metadata["jsd_weight"]),
            "fold": int(metadata["fold"]), "seed": int(metadata["seed"]),
            "protocol_id": metadata["protocol_id"],
            "initial_state_sha256": metadata["initial_state_sha256"],
            "layer": layer, "kind": kind,
            "total_filters": int(stats["total_filters"]),
            "active_filters": int(stats["active_filters"]),
            "constant_filters": int(stats["constant_filters"]),
            "n_rows": int(stats["n_nodes"]),
            "rank_ceiling": int(stats["rank_ceiling"]),
            "n_graphs": int(classification["n_graphs"]),
            "val_accuracy_percent": 100 * number(classification["accuracy"]),
            "val_cross_entropy": number(classification["cross_entropy"]),
            "train_seconds": number(metadata.get("train_seconds")),
            "checkpoint_epoch": metadata.get("checkpoint_epoch"),
            "checkpoint_path": metadata.get("checkpoint_path"),
            "weighting": document["weighting"],
            "std_tol": float(document["std_tol"]),
            "diagnostics_path": str(path), "selected": False, "is_matched": False,
        }
        for metric in METRICS:
            if metric not in run:
                run[metric] = number(stats.get(metric))
        prefix = f"{kind}_layer{layer}_"
        with np.load(path.parent / "correlations.npz", allow_pickle=False) as arrays:
            run["_correlation"] = np.array(arrays[prefix + "correlation"], dtype=float)
            run["_spectrum"] = np.array(arrays[prefix + "normalized_eigenvalues"], dtype=float)
        m = run["total_filters"]
        if run["_correlation"].shape != (m, m) or run["_spectrum"].shape != (m,):
            raise ValueError(f"Unexpected matrix/spectrum dimensions in {path.parent}.")
        if not np.isfinite(run["_spectrum"]).all() or (run["_spectrum"] < 0).any():
            raise ValueError(f"Invalid normalized spectrum in {path.parent}.")
        expected_sum = 1.0 if run["active_filters"] else 0.0
        if not np.isclose(float(run["_spectrum"].sum()), expected_sum, atol=1e-6):
            raise ValueError(f"Spectrum has unexpected sum in {path.parent}.")
        with np.load(path.parent / "responses.npz", allow_pickle=False) as arrays:
            run["_graph_ids"] = np.array(arrays["graph_ids"])
        structural = selected[0].get("structural")
        run["_structural"] = None
        if structural is not None:
            run["mean_mask_wl"] = number(structural.get("mean_mask_wl"))
            if int(structural["total_masks"]) != m:
                raise ValueError(f"Structural/response mask counts differ in {path}.")
            with np.load(path.parent / "structural.npz", allow_pickle=False) as arrays:
                similarity = np.array(arrays[f"layer{layer}_wl_similarity"], dtype=float)
            if similarity.shape != (m, m) or not np.isfinite(similarity).all():
                raise ValueError(f"Invalid WL similarity matrix in {path.parent}.")
            if (similarity < -1e-8).any() or (similarity > 1 + 1e-8).any():
                raise ValueError(f"WL similarities outside [0, 1] in {path.parent}.")
            if not np.allclose(similarity, similarity.T, atol=1e-8):
                raise ValueError(f"WL similarity matrix is not symmetric in {path.parent}.")
            run["_structural"] = np.clip(similarity, 0, 1)
        runs.append(run)
    if not runs:
        raise ValueError(f"No diagnostics.json files found under {root}.")
    for key in ("dataset", "protocol_id", "total_filters", "weighting", "std_tol"):
        values = {run[key] for run in runs}
        if len(values) != 1:
            raise ValueError(f"Several {key} values found: {sorted(values, key=str)}. "
                             "Use a results folder containing one comparison protocol.")
    keys = [(r["reg_type"], r["jsd_weight"], r["fold"], r["seed"]) for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/weight/fold/seed runs. Use a narrower --root.")
    if "none" in {r["reg_type"] for r in runs} and \
            {r["jsd_weight"] for r in runs if r["reg_type"] == "none"} != {0.0}:
        raise ValueError("Runs with reg_type none must have weight 0.")
    structural_present = [run["_structural"] is not None for run in runs]
    if any(structural_present) and not all(structural_present):
        raise ValueError("Structural WL diagnostics are present for only some runs.")
    return runs


# ------------------------------------------------------------------ weight selection
def tuning_rows(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["reg_type"], run["jsd_weight"])].append(run)
    rows = []
    for (method, weight), group in sorted(groups.items(),
                                          key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])):
        row = {"reg_type": method, "jsd_weight": weight, "n_runs": len(group),
               "fold_seed": sorted((r["fold"], r["seed"]) for r in group), "_runs": group}
        for metric in METRICS:
            row["mean_" + metric] = finite_mean([r[metric] for r in group])
            row["sd_" + metric] = finite_sd([r[metric] for r in group])
        rows.append(row)
    return rows


def select_weights(rows, select_by, overrides):
    """Per method: among weights with the widest fold/seed coverage, pick the best."""
    chosen, notes = {}, {}
    for method in METHODS:
        candidates = [row for row in rows if row["reg_type"] == method]
        if not candidates:
            continue
        if method in overrides:
            match = [row for row in candidates if math.isclose(row["jsd_weight"], overrides[method])]
            if not match:
                raise ValueError(f"--weight {method}={overrides[method]:g} has no runs.")
            chosen[method] = match[0]
            notes[method] = "set by --weight"
            continue
        widest = max(row["n_runs"] for row in candidates)
        pool = [row for row in candidates if row["n_runs"] == widest]
        if select_by == "val_cross_entropy":
            key = lambda r: (r["mean_val_cross_entropy"], -r["mean_val_accuracy_percent"])
        else:
            key = lambda r: (-r["mean_val_accuracy_percent"], r["mean_val_cross_entropy"])
        best = min(pool, key=key)
        chosen[method] = best
        notes[method] = (f"best mean {select_by} among {len(pool)} weight(s) with "
                         f"{widest} runs" + ("" if len(pool) == len(candidates)
                                             else f"; {len(candidates) - len(pool)} "
                                                  "weight(s) with fewer runs ignored"))
    if "none" not in chosen:
        raise ValueError("The baseline reg_type none is required.")
    return chosen, notes


def match_sets(chosen):
    methods = [m for m in METHODS if m in chosen]
    indexed = {(r["fold"], r["seed"], m): r for m in methods for r in chosen[m]["_runs"]}
    sets, unmatched = [], []
    for fold, seed in sorted({(f, s) for f, s, _ in indexed}):
        group = [indexed.get((fold, seed, m)) for m in methods]
        if any(r is None for r in group):
            unmatched.extend(r for r in group if r is not None)
            continue
        base = group[0]
        for other in group[1:]:
            if other["initial_state_sha256"] != base["initial_state_sha256"]:
                raise ValueError(f"Different initial weights for fold={fold}, seed={seed}.")
            if not np.array_equal(other["_graph_ids"], base["_graph_ids"]):
                raise ValueError(f"Different validation graphs for fold={fold}, seed={seed}.")
            if other["n_rows"] != base["n_rows"] or other["n_graphs"] != base["n_graphs"]:
                raise ValueError(f"Different probe sizes for fold={fold}, seed={seed}.")
        for r in group:
            r["is_matched"] = True
        sets.append(dict(zip(methods, group)))
    return methods, sets, unmatched


def aggregate_rows(methods, sets):
    records = []
    for method in methods:
        selected = [s[method] for s in sets]
        for metric in METRICS:
            values = [r[metric] for r in selected]
            records.append({
                "reg_type": method, "jsd_weight": selected[0]["jsd_weight"],
                "layer": selected[0]["layer"], "kind": selected[0]["kind"],
                "metric": metric, "n_matched_runs": len(selected),
                "n_finite": int(np.isfinite(np.asarray(values, dtype=float)).sum()),
                "mean": finite_mean(values), "sample_std": finite_sd(values),
            })
    return records


def difference_rows(methods, sets):
    rows = []
    for s in sets:
        base = s["none"]
        for method in methods:
            if method == "none":
                continue
            row = {"fold": base["fold"], "seed": base["seed"], "reg_type": method,
                   "jsd_weight": s[method]["jsd_weight"]}
            row.update({"delta_" + k: s[method][k] - base[k] for k in METRICS})
            rows.append(row)
    return rows


# ------------------------------------------------------------------------ figures
def save_figure(fig, out, name):
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(out / (name + ".png"), dpi=200, bbox_inches="tight")


def method_label(method, chosen):
    if method == "none":
        return LABELS[method]
    return f"{LABELS[method]}, lambda={chosen[method]['jsd_weight']:g}"


def _ticks(ax, m):
    ticks = np.arange(m) if m <= 16 else np.linspace(0, m - 1, 8, dtype=int)
    ax.set_xticks(ticks, ticks + 1)
    ax.set_yticks(ticks, ticks + 1)


def make_figures(methods, sets, chosen, all_runs, out, figure_fold, figure_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    first = sets[0][methods[0]]
    layer, kind, dataset, m = first["layer"], first["kind"], first["dataset"], first["total_filters"]
    subtitle = f"{dataset} | layer {layer} | {KIND_TEXT[kind]}"
    k = len(methods)
    candidates = [s for s in sets
                  if (figure_fold is None or s["none"]["fold"] == figure_fold)
                  and (figure_seed is None or s["none"]["seed"] == figure_seed)]
    if not candidates:
        raise ValueError("Requested heatmap fold/seed is not among the matched runs.")
    shown = candidates[0]
    where = f"fold {shown['none']['fold']}, seed {shown['none']['seed']}"

    # 1. Correlation matrices (same color scale; gray = constant filter).
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#D9D9D9")
    fig, axes = plt.subplots(1, k, figsize=(3.3 * k + 1, 3.9), layout="constrained", squeeze=False)
    for ax, method in zip(axes[0], methods):
        run = shown[method]
        im = ax.imshow(np.ma.masked_invalid(np.abs(run["_correlation"])), vmin=0, vmax=1,
                       cmap=cmap, interpolation="nearest")
        ma = run["mean_abs_corr"]
        ax.set_title(f"{method_label(method, chosen)}\n{run['active_filters']}/{m} active, "
                     f"mean|r|={'n/a' if not math.isfinite(ma) else f'{ma:.2f}'}", fontsize=9)
        _ticks(ax, m)
        ax.set_xlabel("Filter")
        ax.set_ylabel("Filter")
    fig.colorbar(im, ax=list(axes[0]), shrink=0.8, label="|Pearson correlation|")
    fig.suptitle(f"{subtitle}\n{where}. Gray: constant filter (correlation undefined). "
                 "Filter indices are not aligned across models.", fontsize=10)
    save_figure(fig, out, "correlation_heatmaps")
    plt.close(fig)

    # 2. Structural WL similarity of decoded masks.
    if first["_structural"] is not None:
        fig, axes = plt.subplots(1, k, figsize=(3.3 * k + 1, 3.9), layout="constrained", squeeze=False)
        for ax, method in zip(axes[0], methods):
            run = shown[method]
            im = ax.imshow(run["_structural"], vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
            ax.set_title(f"{method_label(method, chosen)}\nmean WL={run['mean_mask_wl']:.2f}", fontsize=9)
            _ticks(ax, m)
            ax.set_xlabel("Mask")
            ax.set_ylabel("Mask")
        fig.colorbar(im, ax=list(axes[0]), shrink=0.8, label="Normalized WL similarity")
        note = "input atom alphabet" if layer == 1 else "learned hidden alphabet"
        fig.suptitle(f"{dataset} | layer {layer} | decoded structural masks ({note})\n{where}. "
                     "WL similarity 1 is not an isomorphism test.", fontsize=10)
        save_figure(fig, out, "structural_wl_heatmaps")
        plt.close(fig)

    # 3. Normalized eigenvalue spectra.
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    x = np.arange(1, m + 1)
    for method in methods:
        spectra = np.stack([s[method]["_spectrum"] for s in sets])
        mean = spectra.mean(axis=0)
        ax.plot(x, mean, color=COLORS[method], marker="o", markersize=3.5, linewidth=1.8,
                label=method_label(method, chosen))
        if len(sets) > 1:
            sd = spectra.std(axis=0, ddof=1)
            ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd),
                            color=COLORS[method], alpha=0.13)
    ax.set_xlabel("Eigenvalue index (sorted within each run)")
    ax.set_ylabel("Eigenvalue / trace (active filters)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.7, m + 0.3)
    if m <= 16:
        ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title(f"{subtitle}\n{len(sets)} matched run(s) per method"
                 + ("; band: sample SD" if len(sets) > 1 else ""), fontsize=10)
    save_figure(fig, out, "normalized_eigenspectra")
    plt.close(fig)

    # 4. Paired metric panels (lines join runs sharing fold, seed and initial weights).
    ceiling = max(s[mm]["rank_ceiling"] for s in sets for mm in methods)
    panels = [
        ("mean_abs_corr", "Mean |correlation|", (0, 1.02)),
        ("mean_sq_corr", "Mean squared correlation", (0, 1.02)),
        ("effective_rank", f"Effective rank (ceiling {ceiling})", (0, m + 0.5)),
        ("constant_filters", "Inactive (constant) filters", (-0.5, m + 0.5)),
        ("val_accuracy_percent", "Validation accuracy (%)", (0, 102)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 4.0), layout="constrained")
    for ax, (metric, title, limits) in zip(axes, panels):
        for s in sets:
            values = [s[mm][metric] for mm in methods]
            ax.plot(range(k), values, color="#999999", alpha=0.35, linewidth=0.8, zorder=1)
        for i, method in enumerate(methods):
            values = np.array([s[method][metric] for s in sets], dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                offsets = np.linspace(-0.08, 0.08, len(finite)) if len(finite) > 1 else [0]
                ax.scatter(i + np.asarray(offsets), finite, color=COLORS[method], s=22, alpha=0.85, zorder=3)
                if len(finite) > 1:
                    ax.errorbar(i, finite.mean(), yerr=finite.std(ddof=1), fmt="D", color="black",
                                markersize=4.5, capsize=3, zorder=4)
                else:
                    ax.plot(i, finite.mean(), "D", color="black", markersize=4.5, zorder=4)
            else:
                ax.text(i, sum(limits) / 2, "n/a", ha="center", va="center", color="#666666")
        ax.set_xlim(-0.4, k - 0.6)
        ax.set_ylim(*limits)
        ax.set_xticks(range(k), [SHORT[mm] for mm in methods], rotation=30)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{subtitle}\nGray lines: same fold/seed/initial weights; "
                 "diamond: mean; bar: sample SD", fontsize=10)
    save_figure(fig, out, "paired_metrics")
    plt.close(fig)

    # 5. Accuracy versus diversity: selected-lambda runs filled, other lambdas hollow.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, metric, xlabel in ((axes[0], "mean_abs_corr", "Mean |correlation| (lower = less redundant)"),
                               (axes[1], "effective_rank", "Effective rank (higher = more diverse)")):
        for method in methods:
            others = [r for r in all_runs if r["reg_type"] == method and not r["selected"]]
            if others:
                ax.scatter([r[metric] for r in others], [r["val_accuracy_percent"] for r in others],
                           facecolors="none", edgecolors=COLORS[method], s=26, alpha=0.6)
            sel = [s[method] for s in sets]
            ax.scatter([r[metric] for r in sel], [r["val_accuracy_percent"] for r in sel],
                       color=COLORS[method], s=30, alpha=0.9, label=method_label(method, chosen))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation accuracy (%)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{subtitle}\nEach point is one run. Filled: selected lambda; hollow: other lambdas. "
                 "Descriptive only; no causal or monotone relation implied.", fontsize=10)
    save_figure(fig, out, "accuracy_vs_diversity")
    plt.close(fig)

    # 6. Lambda tuning curves (validation), one line per regularized method.
    regularized = [mm for mm in methods if mm != "none"]
    if regularized:
        base_runs = [r for r in all_runs if r["reg_type"] == "none"]
        tune_panels = [("val_cross_entropy", "Validation cross-entropy"),
                       ("val_accuracy_percent", "Validation accuracy (%)"),
                       ("mean_abs_corr", "Mean |correlation|"),
                       ("effective_rank", "Effective rank")]
        fig, axes = plt.subplots(1, len(tune_panels), figsize=(3.4 * len(tune_panels), 3.9),
                                 layout="constrained")
        for ax, (metric, title) in zip(axes, tune_panels):
            base = finite_mean([r[metric] for r in base_runs])
            if math.isfinite(base):
                ax.axhline(base, color=COLORS["none"], linestyle="--", linewidth=1.2,
                           label="No regularizer")
            for method in regularized:
                by_weight = defaultdict(list)
                for r in all_runs:
                    if r["reg_type"] == method:
                        by_weight[r["jsd_weight"]].append(r[metric])
                ws = sorted(by_weight)
                means = [finite_mean(by_weight[w]) for w in ws]
                sds = [finite_sd(by_weight[w]) for w in ws]
                ax.errorbar(ws, means, yerr=[0 if not math.isfinite(s) else s for s in sds],
                            color=COLORS[method], marker="o", markersize=4, capsize=2,
                            linewidth=1.4, label=LABELS[method])
                w_sel = chosen[method]["jsd_weight"]
                ax.scatter([w_sel], [finite_mean(by_weight[w_sel])], s=90, facecolors="none",
                           edgecolors=COLORS[method], linewidths=1.6, zorder=5)
            ax.set_xscale("log")
            ax.set_xlabel("lambda (log scale)")
            ax.set_title(title, fontsize=9.5)
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{subtitle}\nValidation tuning: mean and SD over runs per lambda; "
                     "ring: selected lambda. Lambda scales are not comparable across methods.",
                     fontsize=10)
        save_figure(fig, out, "lambda_tuning")
        plt.close(fig)
    return {"fold": shown["none"]["fold"], "seed": shown["none"]["seed"]}


# --------------------------------------------------------------------------- main
def analyze(root, layer, kind, select_by, overrides, figure_fold, figure_seed, out):
    runs = read_runs(root, layer, kind)
    rows = tuning_rows(runs)
    chosen, notes = select_weights(rows, select_by, overrides)
    for method, row in chosen.items():
        for r in row["_runs"]:
            r["selected"] = True
    methods, sets, unmatched = match_sets(chosen)
    if not sets:
        raise ValueError("No fold/seed has all methods at their selected lambda.")
    out.mkdir(parents=True, exist_ok=True)
    run_cols = ["dataset", "reg_type", "jsd_weight", "selected", "is_matched", "fold", "seed",
                "layer", "kind", "protocol_id", "initial_state_sha256", "n_graphs", "n_rows",
                "rank_ceiling", "total_filters", *METRICS, "weighting", "std_tol",
                "checkpoint_epoch", "checkpoint_path", "diagnostics_path"]
    write_csv(out / "metrics.csv", runs, run_cols)
    tune_cols = ["reg_type", "jsd_weight", "n_runs"] + [p + m for m in METRICS for p in ("mean_", "sd_")]
    write_csv(out / "tuning.csv", rows, tune_cols)
    write_csv(out / "summary.csv", aggregate_rows(methods, sets),
              ["reg_type", "jsd_weight", "layer", "kind", "metric", "n_matched_runs",
               "n_finite", "mean", "sample_std"])
    write_csv(out / "paired_differences.csv", difference_rows(methods, sets),
              ["fold", "seed", "reg_type", "jsd_weight", *("delta_" + m for m in METRICS)])
    heatmap = make_figures(methods, sets, chosen, runs, out, figure_fold, figure_seed)
    report = {
        "dataset": runs[0]["dataset"], "protocol_id": runs[0]["protocol_id"],
        "layer": layer, "kind": kind, "view": KIND_TEXT[kind],
        "methods": methods, "n_matched_sets": len(sets),
        "selected_weights": {m: chosen[m]["jsd_weight"] for m in methods},
        "selection_rule": select_by, "selection_notes": notes,
        "unmatched_runs_excluded_from_summaries": [r["diagnostics_path"] for r in unmatched],
        "heatmap_run": heatmap,
        "difference_direction": "method minus none, same fold/seed/initial weights",
        "error_bars": "Sample SD across matched runs; not confidence intervals.",
        "caveats": [
            "Lambda was selected on the same validation graphs: validation metrics are optimistic.",
            "Correlation-based diversity is linear; it says nothing about predictive usefulness.",
            "Constant filters are excluded from correlations/effective rank and counted separately.",
            "Pairwise correlations are evaluation metrics only; no method is trained on them.",
            "Graph-level effective rank is capped by the number of validation graphs minus one.",
        ],
    }
    with (out / "analysis_info.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[layer {layer}, {kind}] {len(sets)} matched set(s) of {len(methods)} methods; "
          f"lambda: " + ", ".join(f"{m}={chosen[m]['jsd_weight']:g}" for m in methods)
          + f"; {len(unmatched)} unmatched run(s) excluded -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/exp1"))
    parser.add_argument("--layer", type=int, nargs="+", default=[1])
    parser.add_argument("--kind", choices=KINDS, nargs="+", default=list(KINDS))
    parser.add_argument("--select-by", choices=("val_cross_entropy", "val_accuracy"),
                        default="val_cross_entropy")
    parser.add_argument("--weight", action="append", default=[], metavar="METHOD=LAMBDA",
                        help="Force a lambda for a method, e.g. --weight logdet=10.")
    parser.add_argument("--figure-fold", type=int, default=None)
    parser.add_argument("--figure-seed", type=int, default=None)
    parser.add_argument("--out", type=Path, help="Parent output folder (default: --root).")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Results directory does not exist: {args.root}")
    overrides = {}
    for item in args.weight:
        method, _, value = item.partition("=")
        if method not in METHODS or method == "none" or not value:
            parser.error(f"Bad --weight {item!r}; use e.g. logdet=10.")
        overrides[method] = float(value)
    select_by = "val_accuracy_percent" if args.select_by == "val_accuracy" else args.select_by
    try:
        for layer in args.layer:
            if layer < 1:
                parser.error("Layer numbers start at 1.")
            for kind in args.kind:
                out = (args.out or args.root) / f"analysis_layer{layer}_{kind}"
                analyze(args.root, layer, kind, select_by, overrides,
                        args.figure_fold, args.figure_seed, out)
        print("Validation diagnostics only; no test data were used.")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
