"""Response diagnostics for the first GKNN redundancy experiment.

The NumPy statistics can be tested without the GKNN dependencies::

    python exp1_diagnostics.py --self-test

The exporter expects the user's Model: ``model(data)`` returns
``(graph_logits, scaled_layer_responses, auxiliary_loss)`` and
``model.conv_layers`` contains the graph-kernel layers. It observes raw
responses with temporary hooks and does not change the training loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def response_statistics(R, std_tol: float = 1e-8):
    """Return ``(statistics, arrays)`` for one full-probe response matrix.

    Rows are observed nodes, columns are filters. Every node receives equal
    weight. A column is active when its population standard deviation is
    strictly greater than ``std_tol``. Pairwise statistics use active
    columns only; a constant filter is never counted as decorrelated.

    ``effective_rank`` is the entropy rank of the *correlation eigenvalues*,
    not of the singular values of the unstandardized response matrix.
    Correlations are undefined when fewer than two filters are active.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] < 1 or R.shape[1] < 1:
        raise ValueError("R must have shape (N, m), with N >= 1 and m >= 1.")
    if not np.isfinite(R).all():
        raise ValueError("Responses contain NaN or infinity; inspect training first.")
    if not np.isfinite(std_tol) or std_tol < 0:
        raise ValueError("std_tol must be finite and nonnegative.")

    n_nodes, total = R.shape
    centered = R - R.mean(axis=0, keepdims=True)
    std = np.sqrt(np.mean(centered * centered, axis=0))
    if not np.isfinite(std).all():
        raise ValueError("Response scale overflowed the variance calculation.")
    active_mask = std > std_tol
    active = int(active_mask.sum())

    full_correlation = np.full((total, total), np.nan, dtype=np.float64)
    eigenvalues = np.zeros(total, dtype=np.float64)
    normalized_eigenvalues = np.zeros(total, dtype=np.float64)
    mean_abs_corr = None
    mean_sq_corr = None
    effective_rank = 0.0
    participation_rank = 0.0

    if active:
        U = centered[:, active_mask]
        U = U / np.linalg.norm(U, axis=0, keepdims=True)
        C = U.T @ U
        C = (C + C.T) / 2.0
        np.fill_diagonal(C, 1.0)
        full_correlation[np.ix_(active_mask, active_mask)] = C
        eig = np.linalg.eigvalsh(C)[::-1]
        # A Gram matrix is PSD. Small negative values are roundoff, not
        # negative variance. A materially negative eigenvalue is an error.
        if eig.min() < -1e-9 * max(1, active):
            raise ArithmeticError("Correlation matrix is unexpectedly non-PSD.")
        eig = np.maximum(eig, 0.0)
        eigenvalues[:active] = eig
        p = eig / eig.sum()
        normalized_eigenvalues[:active] = p
        positive = p[p > 0]
        rank_ceiling = min(active, n_nodes - 1)
        effective_rank = min(
            float(np.exp(-np.sum(positive * np.log(positive)))),
            float(rank_ceiling),
        )
        participation_rank = min(
            float(1.0 / np.sum(p * p)), float(rank_ceiling)
        )
        if active >= 2:
            off = C[~np.eye(active, dtype=bool)]
            mean_abs_corr = float(np.mean(np.abs(off)))
            mean_sq_corr = float(np.mean(off * off))

    stats = {
        "n_nodes": int(n_nodes),
        "total_filters": int(total),
        "active_filters": active,
        "constant_filters": int(total - active),
        "active_fraction": float(active / total),
        "mean_abs_corr": mean_abs_corr,
        "mean_sq_corr": mean_sq_corr,
        "effective_rank": effective_rank,
        "participation_rank": participation_rank,
        "rank_fraction_total": float(effective_rank / total),
        "rank_fraction_active": float(effective_rank / active) if active else None,
        "rank_ceiling": int(min(active, max(0, n_nodes - 1))),
    }
    arrays = {
        "correlation": full_correlation,
        "active_mask": active_mask,
        "std": std,
        "eigenvalues": eigenvalues,
        "normalized_eigenvalues": normalized_eigenvalues,
    }
    return stats, arrays


def pool_by_graph(R, graph_position, n_graphs):
    """Sum node rows per graph (the model's ``global_add_pool``).

    ``graph_position[k]`` is the 0-based position, in validation order, of the
    graph that node row k belongs to. Returns an ``(n_graphs, m)`` matrix.
    """
    R = np.asarray(R, dtype=np.float64)
    graph_position = np.asarray(graph_position)
    if graph_position.shape != (R.shape[0],):
        raise ValueError("One graph position per node row is required.")
    if graph_position.min() < 0 or graph_position.max() >= n_graphs:
        raise ValueError("Graph positions out of range.")
    pooled = np.zeros((n_graphs, R.shape[1]), dtype=np.float64)
    np.add.at(pooled, graph_position, R)
    return pooled


# Views exported per layer. Node-level: one row per validation node.
# Graph-level: one row per validation graph, after add pooling; "pooled_scaled"
# is exactly the per-layer block of the representation fed to the MLP.
VIEWS = ("raw", "scaled", "pooled_raw", "pooled_scaled")


def _json_value(value: Any):
    """Keep metadata JSON-readable without silently stringifying objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def export_response_diagnostics(
    model,
    loader,
    directory,
    graph_ids,
    metadata,
    std_tol: float = 1e-8,
):
    """Export one checkpoint's validation responses and diagnostics.

    ``loader`` MUST be nonshuffled, without dropped batches, and visit
    exactly the graphs in ``graph_ids`` in that order. The model and data
    must use CPU, matching the provided GraKeL implementation. Use a single
    graph kernel so each response column corresponds to one mask.

    Files written under ``directory``:
      * responses.npz: raw/scaled responses, graph identifiers, predictions;
      * correlations.npz: correlations and spectra for every layer/view;
      * diagnostics.json: metadata, validation performance and statistics.

    This function concatenates responses before computing correlations. It
    neither evaluates the training regularizer nor reads a test loader.
    """
    # Keep metric-only tests usable without PyTorch/Lightning/PyG installed.
    import torch
    from torch.nn import functional as F
    from torch.utils.data import RandomSampler

    graph_ids = np.asarray(graph_ids)
    if graph_ids.ndim != 1 or graph_ids.size < 1:
        raise ValueError("graph_ids must be a nonempty one-dimensional sequence.")
    if not np.issubdtype(graph_ids.dtype, np.integer):
        raise ValueError("graph_ids must contain integer dataset indices.")
    graph_ids = graph_ids.astype(np.int64, copy=False)
    if np.unique(graph_ids).size != graph_ids.size:
        raise ValueError("Validation graph_ids contain repeated graphs.")
    if isinstance(getattr(loader, "sampler", None), RandomSampler):
        raise ValueError("Diagnostics require shuffle=False on the validation loader.")
    if bool(getattr(loader, "drop_last", False)):
        raise ValueError("Diagnostics require drop_last=False.")
    if not hasattr(model, "conv_layers") or len(model.conv_layers) == 0:
        raise ValueError("Expected the user's model.conv_layers graph-kernel modules.")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("This probe targets the provided CPU GKNN implementation.")
    hparams = getattr(model, "hparams", {})
    kernel_name = (
        hparams.get("kernel", "wl")
        if hasattr(hparams, "get")
        else getattr(hparams, "kernel", "wl")
    )
    if "+" in str(kernel_name):
        raise ValueError("Use one kernel, e.g. --kernel wl, for this first experiment.")

    n_layers = len(model.conv_layers)
    raw_chunks = [[] for _ in range(n_layers)]
    scaled_chunks = [[] for _ in range(n_layers)]
    node_graph_chunks = []
    node_position_chunks = []
    label_chunks = []
    logit_chunks = []
    captured = [None] * n_layers
    hooks = []
    prior_modes = [(module, module.training) for module in model.modules()]
    graph_offset = 0
    total_ce = 0.0

    def make_hook(layer_index):
        def capture_raw(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 2:
                raise ValueError("Each GKernel must return a two-dimensional tensor.")
            if captured[layer_index] is not None:
                raise ValueError("A GKernel was called twice in one model forward.")
            # Copy now, before downstream code could modify the tensor.
            captured[layer_index] = output.detach().cpu().clone()
        return capture_raw

    try:
        for layer_index, layer in enumerate(model.conv_layers):
            hooks.append(layer.register_forward_hook(make_hook(layer_index)))
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to("cpu")
                captured[:] = [None] * n_layers
                logits, scaled_responses, _ = model(data)
                labels = data.y.reshape(-1).long()
                n_graphs = int(labels.numel())
                if n_graphs == 0:
                    raise ValueError("An empty validation batch was received.")
                if logits.ndim != 2 or logits.shape[0] != n_graphs:
                    raise ValueError("Expected graph-classification logits with shape (G, C).")
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite logits: the checkpoint is not valid.")
                if len(scaled_responses) != n_layers:
                    raise ValueError("The responses list does not match conv_layers.")
                node_batch = data.batch.detach().cpu().numpy()
                if node_batch.ndim != 1 or not np.array_equal(
                    np.unique(node_batch), np.arange(n_graphs)
                ):
                    raise ValueError("Expected contiguous batch indices and nonempty graphs.")
                if graph_offset + n_graphs > graph_ids.size:
                    raise ValueError("The validation loader visits more graphs than graph_ids.")
                batch_graph_ids = graph_ids[graph_offset:graph_offset + n_graphs]
                node_graph_chunks.append(batch_graph_ids[node_batch])
                node_position_chunks.append(graph_offset + node_batch)
                for layer_index, scaled in enumerate(scaled_responses):
                    raw = captured[layer_index]
                    if raw is None:
                        raise ValueError("A graph-kernel layer produced no captured output.")
                    if not torch.is_tensor(scaled) or scaled.ndim != 2:
                        raise ValueError("Expected a matrix for every scaled layer response.")
                    if raw.shape != scaled.shape or raw.shape[0] != node_batch.size:
                        raise ValueError(
                            "Responses must contain one row per input node. "
                            "Node subsampling needs a separate index mapping."
                        )
                    raw_chunks[layer_index].append(raw.to(torch.float64).numpy())
                    scaled_chunks[layer_index].append(
                        scaled.detach().cpu().to(torch.float64).numpy().copy()
                    )
                label_chunks.append(labels.cpu().numpy().copy())
                logit_chunks.append(logits.detach().cpu().to(torch.float64).numpy().copy())
                total_ce += float(F.cross_entropy(logits, labels, reduction="sum").item())
                graph_offset += n_graphs
    finally:
        for hook in hooks:
            hook.remove()
        # Restore each module's exact mode, including any deliberately mixed modes.
        for module, was_training in prior_modes:
            module.training = was_training

    if graph_offset != graph_ids.size:
        raise ValueError(
            "Validation graph count mismatch: loader visited %d, graph_ids contains %d."
            % (graph_offset, graph_ids.size)
        )

    labels = np.concatenate(label_chunks)
    logits = np.concatenate(logit_chunks)
    response_arrays = {
        "graph_id_per_node": np.concatenate(node_graph_chunks),
        "graph_ids": graph_ids,
        "labels": labels,
        "logits": logits,
    }
    if getattr(model, "mask", None) is not None:
        response_arrays["mask_gains"] = model.mask.detach().cpu().numpy().copy()
    node_position = np.concatenate(node_position_chunks)
    diagnostic_arrays = {}
    layer_summaries = []
    for layer_index in range(n_layers):
        layer_summary = {"layer": layer_index + 1}
        node_matrices = {
            "raw": np.concatenate(raw_chunks[layer_index], axis=0),
            "scaled": np.concatenate(scaled_chunks[layer_index], axis=0),
        }
        for view in VIEWS:
            if view.startswith("pooled_"):
                matrix = pool_by_graph(node_matrices[view[len("pooled_"):]],
                                       node_position, graph_offset)
            else:
                matrix = node_matrices[view]
            prefix = "%s_layer%d" % (view, layer_index + 1)
            response_arrays[prefix] = matrix
            stats, arrays = response_statistics(matrix, std_tol=std_tol)
            layer_summary[view] = stats
            for name, array in arrays.items():
                diagnostic_arrays[prefix + "_" + name] = array
        layer_summaries.append(layer_summary)

    summary = {
        "format_version": 1,
        "metadata": _json_value(metadata),
        "classification": {
            "accuracy": float(np.mean(logits.argmax(axis=1) == labels)),
            "cross_entropy": float(total_ce / graph_offset),
            "n_graphs": int(graph_offset),
        },
        "weighting": "equal_per_node",
        "views": {
            "raw": "node level, GKernel output before the learned mask gain",
            "scaled": "node level, after the mask gain (as used by the model)",
            "pooled_raw": "graph level, add pooling of raw responses",
            "pooled_scaled": "graph level, add pooling of scaled responses (MLP input block)",
        },
        "std_tol": float(std_tol),
        "layers": layer_summaries,
    }
    # Validate JSON before creating output files; NaN metrics must not be hidden.
    summary_text = json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "responses.npz", **response_arrays)
    np.savez_compressed(directory / "correlations.npz", **diagnostic_arrays)
    (directory / "diagnostics.json").write_text(summary_text + "\n", encoding="utf-8")
    return summary


def _self_test():
    """Known mathematical cases; these are not GKNN experimental results."""
    x = np.asarray([-3.0, -1.0, 1.0, 3.0])
    stats, arrays = response_statistics(np.column_stack([x, x, np.ones(4)]))
    assert stats["active_filters"] == 2 and stats["constant_filters"] == 1
    np.testing.assert_allclose(stats["mean_abs_corr"], 1.0, atol=1e-12)
    np.testing.assert_allclose(stats["effective_rank"], 1.0, atol=1e-12)
    assert np.isnan(arrays["correlation"][2]).all()
    assert np.isnan(arrays["correlation"][:, 2]).all()

    stats, arrays = response_statistics(np.column_stack([x, -x, 2 * x + 7]))
    np.testing.assert_allclose(stats["mean_abs_corr"], 1.0, atol=1e-12)
    np.testing.assert_allclose(stats["participation_rank"], 1.0, atol=1e-12)
    np.testing.assert_allclose(arrays["correlation"][0, 1], -1.0, atol=1e-12)

    orthogonal = np.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    stats, arrays = response_statistics(orthogonal)
    np.testing.assert_allclose(stats["mean_abs_corr"], 0.0, atol=1e-12)
    np.testing.assert_allclose(stats["effective_rank"], 2.0, atol=1e-12)
    np.testing.assert_allclose(arrays["normalized_eigenvalues"], [0.5, 0.5])

    stats, arrays = response_statistics(np.ones((5, 3)))
    assert stats["active_filters"] == 0 and stats["effective_rank"] == 0
    assert stats["mean_abs_corr"] is None and np.isnan(arrays["correlation"]).all()
    assert not arrays["eigenvalues"].any()

    stats, _ = response_statistics(x[:, None])
    assert stats["active_filters"] == 1 and stats["mean_sq_corr"] is None
    np.testing.assert_allclose(stats["effective_rank"], 1.0)
    stats, _ = response_statistics(np.array([[1.0, 2.0]]))
    assert stats["effective_rank"] == 0 and stats["constant_filters"] == 2

    for invalid in (np.asarray([]), np.array([[float("nan")]]), np.ones((0, 2))):
        try:
            response_statistics(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input did not raise ValueError.")

    # Correlation/participation-rank identity for an arbitrary active matrix.
    rng = np.random.default_rng(19)
    stats, _ = response_statistics(rng.normal(size=(40, 5)))
    expected = 5 / (1 + 4 * stats["mean_sq_corr"])
    np.testing.assert_allclose(stats["participation_rank"], expected, atol=1e-12)

    # Add pooling: rows of graph k summed, empty-graph-free positions.
    R = np.arange(12, dtype=float).reshape(6, 2)
    pooled = pool_by_graph(R, np.array([0, 0, 1, 2, 2, 2]), 3)
    np.testing.assert_allclose(pooled, [[2, 4], [4, 5], [24, 27]])
    # Node responses exactly uncorrelated, pooled ones perfectly correlated:
    # add pooling makes both filters track graph size. The views differ.
    block = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)  # corr 0
    node = np.vstack([block] * 6)                     # graphs with 1, 2, 3 blocks
    pos = np.repeat([0, 1, 1, 2, 2, 2], 4)
    s_node, _ = response_statistics(node)
    s_pool, _ = response_statistics(pool_by_graph(node, pos, 3))
    np.testing.assert_allclose(s_node["mean_abs_corr"], 0.0, atol=1e-12)
    np.testing.assert_allclose(s_pool["mean_abs_corr"], 1.0, atol=1e-12)
    try:
        pool_by_graph(R, np.array([0, 0, 1, 2, 2, 3]), 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range graph position did not raise.")
    print("PASS: duplicate, antipodal, affine, orthogonal, constant, shape, rank and pooling checks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run NumPy metric checks.")
    arguments = parser.parse_args()
    if arguments.self_test:
        _self_test()
    else:
        parser.print_help()
