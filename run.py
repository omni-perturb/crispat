#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import muon as mu
import crispat

METHODS: dict[str, tuple[Callable, int]] = {
    "pgmm": (crispat.ga_poisson_gauss, 500),
    "gauss": (crispat.ga_gauss, 250),
    "2beta": (crispat.ga_2beta, 500),
    "3beta": (crispat.ga_3beta, 500),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="crispat mixture model guide assignment"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="path to input h5mu (MuData with rna and crispr modalities)",
    )
    parser.add_argument("--output_dir", "-o", required=True, help="output directory")
    parser.add_argument("--name", "-n", required=True, help="output file prefix")
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHODS.keys()),
        help="guide assignment method",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=None,
        help="SVI training steps (default: 500 for pgmm/2beta/3beta, 250 for gauss)",
    )
    parser.add_argument(
        "--umi_threshold",
        type=int,
        default=0,
        help="post-assignment UMI filter (default: 0)",
    )
    return parser.parse_args()


def build_assignment_layer(
    assignments_df: pd.DataFrame, adata: ad.AnnData
) -> sp.csr_matrix:
    """Build a binary sparse cells x guides matrix from the long-format assignments CSV."""
    cell_idx = pd.Index(adata.obs_names)
    guide_idx = pd.Index(adata.var_names)

    rows = cell_idx.get_indexer(assignments_df["cell"].values)
    cols = guide_idx.get_indexer(assignments_df["gRNA"].values)

    mask = (rows >= 0) & (cols >= 0)
    rows, cols = rows[mask], cols[mask]

    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(adata.n_obs, adata.n_vars),
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fn, default_n_iter = METHODS[args.method]
    n_iter = args.n_iter if args.n_iter is not None else default_n_iter

    print("Loading input h5mu ...", file=sys.stderr)
    mdata = mu.read_h5mu(args.input)
    adata_guides = mdata["crispr"]
    print(
        f"  crispr modality: {adata_guides.n_obs} x {adata_guides.n_vars}",
        file=sys.stderr,
    )

    if adata_guides.n_vars == 0:
        raise ValueError("crispr modality is empty")

    tmp_path = os.path.join(args.output_dir, "_crispr_counts.h5ad")
    crispat_out = os.path.join(args.output_dir, args.method)
    print(
        f"Running crispat {args.method} (n_iter={n_iter}, umi_threshold={args.umi_threshold}) ...",
        file=sys.stderr,
    )
    try:
        adata_guides.write_h5ad(tmp_path)
        fn(tmp_path, crispat_out, n_iter=n_iter, UMI_threshold=args.umi_threshold)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    os.unlink(tmp_path)

    print("Loading assignments ...", file=sys.stderr)
    assignments = pd.read_csv(os.path.join(crispat_out, "assignments.csv"))

    n_assigned = assignments["cell"].nunique()
    n_total = adata_guides.n_obs
    print(
        f"  {n_assigned}/{n_total} cells assigned ({100 * n_assigned / n_total:.1f}%)",
        file=sys.stderr,
    )

    adata_guides.layers["assigned"] = build_assignment_layer(assignments, adata_guides)

    per_cell = (
        assignments.groupby("cell")["gRNA"]
        .apply(lambda g: ",".join(sorted(g)))
        .rename("guide_identity")
        .to_frame()
    )
    per_cell["n_guides_assigned"] = per_cell["guide_identity"].str.count(",") + 1
    adata_guides.obs = adata_guides.obs.join(per_cell, how="left")
    adata_guides.obs["n_guides_assigned"] = (
        adata_guides.obs["n_guides_assigned"].fillna(0).astype(int)
    )

    adata_guides.uns["guide_assignment_method"] = f"crispat_{args.method}"
    adata_guides.uns["guide_assignment_params"] = {
        "method": args.method,
        "n_iter": n_iter,
        "umi_threshold": args.umi_threshold,
    }

    out_path = os.path.join(args.output_dir, f"{args.name}.h5ad")
    print(f"Writing {out_path} ...", file=sys.stderr)
    adata_guides.write_h5ad(out_path)


if __name__ == "__main__":
    main()
