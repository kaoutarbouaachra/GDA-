"""Structural WL similarity of the effective discrete GKNN masks.

Uses the supplied layers.max_comp decoder, including its empty-edge
fallback, rather than introducing a different connectivity rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def decode_masks_from_arrays(adjacencies, label_scores, component_function):
    """Decode already-refreshed P/X arrays; return kernel and JSON graphs."""
    adjacencies = np.asarray(adjacencies)
    label_scores = np.asarray(label_scores)
    if (adjacencies.ndim != 3 or adjacencies.shape[1] != adjacencies.shape[2]
            or adjacencies.shape[0] < 1 or adjacencies.shape[1] < 1):
        raise ValueError("Expected P with shape (masks, nodes, nodes).")
    if label_scores.shape[0] != adjacencies.shape[0]:
        raise ValueError("P and X contain different numbers of masks.")
    if not np.isfinite(adjacencies).all() or not np.isfinite(label_scores).all():
        raise ValueError("Mask parameters contain NaN or infinity.")
    kernel_graphs, json_graphs = [], []
    for mask_index, adjacency in enumerate(adjacencies):
        labels = label_scores[mask_index].argmax(axis=-1).reshape(-1)
        if labels.size != adjacency.shape[0]:
            raise ValueError("Mask node-label count does not match its adjacency.")
        edges = {(int(i), int(j)) for i, j in zip(*np.nonzero(adjacency))}
        node_labels = {int(i): int(label) for i, label in enumerate(labels)}
        effective_edges, effective_labels = component_function(edges, node_labels)
        effective_edges = [(int(i), int(j)) for i, j in effective_edges]
        effective_labels = {int(i): int(label) for i, label in effective_labels.items()}
        if not effective_labels:
            raise ValueError("The mask decoder returned an empty graph.")
        if any(i not in effective_labels or j not in effective_labels
               for i, j in effective_edges):
            raise ValueError("Decoded edges reference vertices without labels.")
        kernel_graphs.append((effective_edges, effective_labels))
        json_graphs.append({
            "mask": mask_index + 1,
            "node_labels": {str(i): label for i, label in sorted(effective_labels.items())},
            # Store each undirected edge once; retain decoder-generated self-loops.
            "edges": [list(edge) for edge in sorted({
                (min(i, j), max(i, j)) for i, j in effective_edges
            })],
        })
    return kernel_graphs, json_graphs


def export_structural_diagnostics(model, directory):
    """Call AFTER the validation probe has refreshed every layer's P and X.

    Returns one record per layer, for merging into diagnostics.json. WL is
    computed independently of the validation graphs and their target labels.
    Similarity 1 is not an isomorphism certificate.
    """
    from grakel.kernels import WeisfeilerLehman
    from layers import max_comp

    records, arrays, decoded_layers = [], {}, []
    for layer_number, layer in enumerate(model.conv_layers, start=1):
        if not hasattr(layer, "P") or not hasattr(layer, "X"):
            raise ValueError("Expected the supplied GKNN layer's discrete P and X.")
        graphs, decoded = decode_masks_from_arrays(
            layer.P.detach().cpu().numpy(),
            layer.X.detach().cpu().numpy(),
            max_comp,
        )
        # Matches the user's provided WL factory (default base VertexHistogram).
        kernel = WeisfeilerLehman(n_iter=3, normalize=True)
        similarity = np.asarray(kernel.fit_transform(graphs), dtype=np.float64)
        m = len(graphs)
        if similarity.shape != (m, m) or not np.isfinite(similarity).all():
            raise ValueError("WL returned an invalid mask similarity matrix.")
        if (not np.allclose(similarity, similarity.T, atol=1e-8)
                or not np.allclose(np.diag(similarity), 1.0, atol=1e-8)
                or similarity.min() < -1e-8 or similarity.max() > 1 + 1e-8):
            raise ValueError("Unexpected values for a normalized WL similarity matrix.")
        similarity = np.clip((similarity + similarity.T) / 2, 0.0, 1.0)
        off_diagonal = similarity[~np.eye(m, dtype=bool)]
        records.append({"layer": layer_number, "structural": {
            "mean_mask_wl": float(off_diagonal.mean()) if m >= 2 else None,
            "total_masks": m,
            "wl_iterations": 3,
            "wl_normalized": True,
            "effective_node_counts": [len(graph[1]) for graph in graphs],
            "decoder": "layers.max_comp on the current discrete P and X",
        }})
        arrays["layer%d_wl_similarity" % layer_number] = similarity
        decoded_layers.append({"layer": layer_number, "masks": decoded})

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "structural.npz", **arrays)
    document = {
        "format_version": 1,
        "decoding": "Exact layers.max_comp decoder from the supplied implementation.",
        "label_semantics": (
            "Layer 1 labels are input feature-category indices. Later-layer labels "
            "are learned channel indices and must not be read as atom identities."
        ),
        "edge_format": "Undirected edges; decoder-generated self-loops are retained.",
        "interpretation": "Normalized WL similarity 1 does not prove graph isomorphism.",
        "layers": decoded_layers,
    }
    (directory / "masks.json").write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return records
