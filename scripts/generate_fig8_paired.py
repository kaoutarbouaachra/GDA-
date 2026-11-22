#!/usr/bin/env python3
"""Approximate Cosmo et al. Fig. 8 data protocol, with paired grouped splits.

Creates five independent single-motif binary tasks (NOT a Ring+Wheel task).
Each positive-negative pair shares an Erdos-Renyi base, node count and
attachment edges; only the inserted motif's internal edges differ.

The motif definitions match scripts/generate_motifs.py used in this project.
They are not claimed to be the paper authors' original data files.

Outputs JSONL compatible with src/synthetic_data.py and 10-fold JSON splits
compatible with run_tracked.py's data/folds/<DATASET>_folds_10.txt reader.

Integrated from FIG8_GKNN_CAM_JSD_code.zip. Changes: dataset names use
--prefix (default FIG8, so they never collide with older SYNTH_* runs);
existing fold/data files are never overwritten unless --overwrite; each
motif record stores its graph radius (egonet radius needed to contain it).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

MOTIFS = ("ring", "wheel", "grid", "ladder", "cliques")


def make_motif(name: str) -> nx.Graph:
    if name == "ring":
        g = nx.cycle_graph(8)
    elif name == "wheel":
        g = nx.wheel_graph(8)
    elif name == "grid":
        g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    elif name == "ladder":
        g = nx.ladder_graph(5)
    elif name == "cliques":
        g = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
        g.add_edge(3, 4)
    else:
        raise ValueError(f"Unsupported motif: {name}")
    return nx.convert_node_labels_to_integers(g)


def connected_matched_foil(template: nx.Graph, rng: random.Random) -> nx.Graph:
    """Negative control with identical node/edge counts, non-isomorphic."""
    n, m = template.number_of_nodes(), template.number_of_edges()
    for _ in range(10000):
        g = nx.gnm_random_graph(n, m, seed=rng.randrange(1 << 30))
        if nx.is_connected(g) and not nx.is_isomorphic(g, template):
            return g
    raise RuntimeError(f"Could not find a valid foil for n={n}, m={m}")


def build_record(base: nx.Graph, insert: nx.Graph, template: nx.Graph,
                 attach: list[tuple[int, int]], present: bool,
                 motif_name: str, pair_id: int) -> dict:
    base_n = base.number_of_nodes()
    total_n = base_n + insert.number_of_nodes()
    g = nx.Graph()
    g.add_nodes_from(range(total_n))
    g.add_edges_from(base.edges())
    g.add_edges_from((base_n + u, base_n + v) for u, v in insert.edges())
    g.add_edges_from(attach)
    block = list(range(base_n, total_n))
    return {
        "n_nodes": total_n,
        "n_edges": g.number_of_edges(),
        "edges": [list(e) for e in sorted(tuple(sorted(e)) for e in g.edges())],
        "label": int(present),
        "n_base_nodes": base_n,
        "base_pair_id": pair_id,
        "motifs": {
            motif_name: {
                "present": bool(present),
                "nodes": block if present else [],
                "block_nodes": block,
                "template_nodes": template.number_of_nodes(),
                "template_edges": template.number_of_edges(),
                "radius": int(nx.radius(template)),
            }
        },
    }


def make_dataset(name: str, n_graphs: int, seed: int, total_min: int,
                 total_max: int, p_base: float, p_attach: float) -> list[dict]:
    rng = random.Random(seed)
    template = make_motif(name)
    n_m = template.number_of_nodes()
    rows: list[dict] = []
    for pair_id in range(n_graphs // 2):
        total_n = rng.randint(total_min, total_max)
        base_n = total_n - n_m
        if base_n < 2:
            raise ValueError("Total graph size too small for the selected motif")
        base = nx.fast_gnp_random_graph(base_n, p_base,
                                        seed=rng.randrange(1 << 30))
        attach = [(u, base_n + v) for u in range(base_n)
                  for v in range(n_m) if rng.random() < p_attach]
        if not attach:
            attach = [(rng.randrange(base_n), base_n + rng.randrange(n_m))]
        foil = connected_matched_foil(template, rng)
        pos = build_record(base, template, template, attach, True, name, pair_id)
        neg = build_record(base, foil, template, attach, False, name, pair_id)
        if (pos["n_nodes"], pos["n_edges"]) != (neg["n_nodes"], neg["n_edges"]):
            raise AssertionError("Positive and negative size mismatch")
        rows.extend((pos, neg))
    rng.shuffle(rows)
    for i, record in enumerate(rows):
        record["graph_id"] = i
    return rows


def make_grouped_folds(rows: list[dict], n_folds: int, seed: int,
                       val_fraction: float = 0.10) -> list[list[list[int]]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row["base_pair_id"])].append(i)
    if any(len(indices) != 2 for indices in groups.values()):
        raise AssertionError("Expected exactly two samples per base pair")
    pair_ids = sorted(groups)
    if len(pair_ids) < n_folds:
        raise ValueError("Fewer base pairs than folds")
    random.Random(seed).shuffle(pair_ids)
    result: list[list[list[int]]] = []
    for fold in range(n_folds):
        test_groups = set(pair_ids[fold::n_folds])
        candidates = [g for g in pair_ids if g not in test_groups]
        random.Random(seed + 10000 + fold).shuffle(candidates)
        n_val = max(1, round(val_fraction * len(candidates)))
        val_groups = set(candidates[:n_val])
        train_groups = set(candidates[n_val:])
        parts = [train_groups, val_groups, test_groups]
        indexes = [sorted(i for g in part for i in groups[g]) for part in parts]
        tr, va, te = map(set, indexes)
        if tr & va or tr & te or va & te or len(tr | va | te) != len(rows):
            raise AssertionError("Split overlap or missing rows")
        for part in indexes:
            counts = Counter(rows[i]["label"] for i in part)
            if counts[0] != counts[1]:
                raise AssertionError("Class imbalance in a grouped split")
        result.append(indexes)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="synthetic")
    p.add_argument("--fold_dir", default="data/folds")
    p.add_argument("--prefix", default="FIG8", help="Dataset names: <PREFIX>_<MOTIF>")
    p.add_argument("--motifs", nargs="+", default=list(MOTIFS), choices=MOTIFS)
    p.add_argument("--n_graphs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--total_min", type=int, default=30)
    p.add_argument("--total_max", type=int, default=50)
    p.add_argument("--p_base", type=float, default=0.10)
    p.add_argument("--p_attach", type=float, default=0.02)
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing data/fold files (never needed for a first run)")
    args = p.parse_args()
    if args.n_graphs < 2 * args.n_folds or args.n_graphs % 2:
        p.error("n_graphs must be even and >= 2 * n_folds")
    if not (0 <= args.p_base <= 1 and 0 <= args.p_attach <= 1):
        p.error("Edge probabilities must be in [0, 1]")
    if args.total_min > args.total_max:
        p.error("total_min must be <= total_max")
    if args.total_min <= max(make_motif(n).number_of_nodes() for n in args.motifs) + 1:
        p.error("total_min is too small for the motifs")

    outdir, folddir = Path(args.out_dir), Path(args.fold_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    folddir.mkdir(parents=True, exist_ok=True)
    targets = {name: (outdir / f"{args.prefix.lower()}_{name}.jsonl",
                      folddir / f"{args.prefix}_{name.upper()}_folds_{args.n_folds}.txt")
               for name in args.motifs}
    existing = [str(f) for pair in targets.values() for f in pair if f.exists()]
    if existing and not args.overwrite:
        p.error("refusing to overwrite existing files (use --overwrite only if intended): "
                + ", ".join(existing))
    mapping = {}
    for name in args.motifs:
        rows = make_dataset(name, args.n_graphs, args.seed, args.total_min,
                            args.total_max, args.p_base, args.p_attach)
        ds = f"{args.prefix}_{name.upper()}"
        path, split_path = targets[name]
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                        encoding="utf-8")
        split = make_grouped_folds(rows, args.n_folds, seed=args.seed + 123)
        split_path.write_text(json.dumps(split), encoding="utf-8")
        mapping[ds] = str(path)
        train_n, val_n, test_n = map(len, split[0])
        print(f"{ds:14s}: {len(rows)} graphs | {len(rows)//2} pairs | "
              f"fold0={train_n}/{val_n}/{test_n} | motif={make_motif(name).number_of_nodes()} "
              f"nodes, radius {nx.radius(make_motif(name))}")
    map_path = outdir / f"{args.prefix.lower()}_datasets.json"
    map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (outdir / f"{args.prefix.lower()}_protocol.json").write_text(json.dumps({
        "reproduction_status": "reconstruction, not the authors' raw dataset",
        "motifs": list(args.motifs), "seed": args.seed,
        "n_graphs_per_motif": args.n_graphs,
        "paired_base_graphs": True, "grouped_train_val_test": True,
        "total_nodes_min": args.total_min, "total_nodes_max": args.total_max,
        "p_base": args.p_base, "p_attach": args.p_attach,
        "folds": args.n_folds,
        "negative_control": "connected, nonisomorphic, same nodes and edges",
    }, indent=2), encoding="utf-8")
    print("Wrote", map_path)


if __name__ == "__main__":
    main()
