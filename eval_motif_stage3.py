#!/usr/bin/env python3
# ─── Project root anchor ─────────────────────────────────────────────────────
# Allows this script to be run from any working directory; ensures stage
# scripts (s1_static_triage.py etc.) and relative output paths resolve from root.
import sys as _sys, os as _os
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
del _ROOT, _sys, _os, _P
# ─────────────────────────────────────────────────────────────────────────────

"""
Stage 3 Evaluation: Clustering Quality on MOTIF Ground Truth

Measures how well your HDBSCAN/PROUD-MAL clusters align with
the true malware family labels from motif_dataset.jsonl.

Metrics computed:
  - Adjusted Rand Index (ARI)   — main headline metric
  - Normalized Mutual Info (NMI)
  - Homogeneity / Completeness / V-measure
  - Per-cluster purity table
  - Per-family fragmentation table (how many clusters a family is split into)

Usage:
    python eval_motif_stage3.py \
        --stage3-dir  data/MOTIF/results/stage3 \
        --stage2-dir  data/MOTIF/results/stage2 \
        --manifest    data/MOTIF/motif_manifest.json

    # Quick mode: use the cluster assignments already written into Stage 2 JSONs
    python eval_motif_stage3.py --stage2-dir data/MOTIF/results/stage2 \
                                --manifest data/MOTIF/motif_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy sklearn import (not in requirements but commonly available)
# ─────────────────────────────────────────────────────────────────────────────

def _import_sklearn():
    try:
        from sklearn import metrics
        return metrics
    except ImportError:
        logger.error(
            "scikit-learn is required for evaluation metrics.\n"
            "Install it with:  pip install scikit-learn"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> dict:
    """Build multi-key index (sha256/md5/MOTIF_<md5>) → manifest entry."""
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    index = {}
    for e in data.get("entries", []):
        sha256 = (e.get("sha256") or "").lower()
        md5 = (e.get("md5") or "").lower()

        if sha256:
            index[sha256] = e
        if md5:
            index[md5] = e
            index[f"motif_{md5}"] = e

        # Fallback: derive MOTIF_<md5> from path
        path = Path(e.get("path", ""))
        if path.stem.lower().startswith("motif_"):
            index[path.stem.lower()] = e
    logger.info(f"Manifest: {len(index)} entries")
    return index


def load_cluster_assignments_from_stage2(stage2_dir: Path) -> List[dict]:
    """
    Read cluster_id from each Stage 2 JSON (Stage 3 writes the stage3 block
    back into the Stage 2 JSON in-place).

    Returns list of {sha256, file_name, cluster_id, pseudo_label, is_outlier}
    """
    records = []
    json_files = list(stage2_dir.glob("*_stage2.json")) + list(stage2_dir.glob("*.json"))
    # Deduplicate
    seen_paths = set()
    unique_files = []
    for p in json_files:
        if p not in seen_paths:
            seen_paths.add(p)
            unique_files.append(p)

    skipped = 0
    for jp in unique_files:
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.debug(f"Skip {jp.name}: {e}")
            skipped += 1
            continue

        stage3 = data.get("stage3")
        if not stage3:
            skipped += 1
            continue

        # Stage 2 files are typically named: MOTIF_<md5>_stage2.json
        # Keep a flexible sample key for matching against manifest (sha256 or md5 or MOTIF_<md5>)
        sample_key = (data.get("sha256") or data.get("file_hash") or "").lower()
        if not sample_key:
            stem = jp.stem.lower()
            if stem.startswith("motif_") and stem.endswith("_stage2"):
                md5 = stem[len("motif_"):-len("_stage2")]
                sample_key = f"motif_{md5}"
            elif stem.startswith("motif_"):
                sample_key = stem
            else:
                sample_key = stem

        records.append({
            "sample_key": sample_key,
            "file_name": data.get("file_name") or jp.name,
            "cluster_id": stage3.get("cluster_id", -1),
            "pseudo_label": stage3.get("pseudo_label", "Unknown"),
            "is_outlier": stage3.get("is_outlier", False),
        })

    logger.info(
        f"Stage 2 JSONs scanned: {len(unique_files)} | "
        f"with stage3 block: {len(records)} | skipped: {skipped}"
    )
    return records


def load_cluster_assignments_from_stage3_report(stage3_dir: Path) -> Optional[dict]:
    """
    Load the stage3_batch_report.json for cluster-level metadata.
    Returns the parsed report or None if not found.
    """
    candidates = list(stage3_dir.glob("stage3_batch_report.json"))
    if not candidates:
        return None
    with open(candidates[0], encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_family(name: str) -> str:
    """Lowercase + strip trailing digits used in variant names."""
    return name.lower().strip()


def build_aligned_arrays(
    records: List[dict], manifest: dict
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """
    Align predicted cluster labels with ground truth family labels.
    Only files present in BOTH the clustering output and the manifest with
    a known family are included.

    Returns:
        y_pred  : list of cluster_id (int)
        y_true  : list of encoded family label (int)
        families: list of family name strings (parallel to y_true)
        sha256s : list of SHA-256 strings (for debugging)
    """
    family_encoder: Dict[str, int] = {}
    y_pred, y_true, families, sha256s = [], [], [], []

    unknown_count = 0
    for rec in records:
        sample_key = (rec.get("sample_key") or "").lower()
        gt = manifest.get(sample_key)

        # Secondary fallback using file_name (e.g., MOTIF_<md5>.exe)
        if not gt:
            file_name = (rec.get("file_name") or "").lower()
            file_stem = Path(file_name).stem
            gt = manifest.get(file_stem)

        # Tertiary fallback if key is motif_<md5>, try md5 directly
        if not gt and sample_key.startswith("motif_"):
            gt = manifest.get(sample_key[len("motif_"):])

        if not gt:
            unknown_count += 1
            continue

        family = normalise_family(gt.get("family") or gt.get("reported_family") or "")
        if not family or family in ("unknown", ""):
            unknown_count += 1
            continue

        if family not in family_encoder:
            family_encoder[family] = len(family_encoder)

        y_pred.append(int(rec["cluster_id"]))
        y_true.append(family_encoder[family])
        families.append(family)
        sha256s.append(sample_key)

    logger.info(
        f"Aligned {len(y_pred)} samples | "
        f"{len(family_encoder)} distinct families | "
        f"{unknown_count} dropped (no GT family)"
    )
    return y_pred, y_true, families, sha256s


def compute_ari_metrics(y_pred: List[int], y_true: List[int]) -> dict:
    """Compute ARI, NMI, Homogeneity, Completeness, V-measure."""
    skm = _import_sklearn()

    ari  = skm.adjusted_rand_score(y_true, y_pred)
    nmi  = skm.normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    hom, com, vme = skm.homogeneity_completeness_v_measure(y_true, y_pred)

    return {
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "homogeneity": round(hom, 4),
        "completeness": round(com, 4),
        "v_measure": round(vme, 4),
        "n_samples": len(y_pred),
        "n_true_families": len(set(y_true)),
        "n_pred_clusters": len(set(y_pred)),
    }


def cluster_purity_table(
    y_pred: List[int], families: List[str]
) -> List[dict]:
    """
    For each cluster, report:
      - dominant family
      - purity (dominant / total)
      - total members
    """
    cluster_to_families: Dict[int, List[str]] = defaultdict(list)
    for cid, fam in zip(y_pred, families):
        cluster_to_families[cid].append(fam)

    rows = []
    for cid, fams in sorted(cluster_to_families.items()):
        counter = Counter(fams)
        dominant, dominant_count = counter.most_common(1)[0]
        purity = dominant_count / len(fams)
        rows.append({
            "cluster_id": cid,
            "size": len(fams),
            "dominant_family": dominant,
            "dominant_count": dominant_count,
            "purity": round(purity, 4),
            "outlier_cluster": cid == -1,
            "family_breakdown": dict(counter.most_common()),
        })
    rows.sort(key=lambda r: r["purity"], reverse=True)
    return rows


def family_fragmentation_table(
    y_pred: List[int], families: List[str]
) -> List[dict]:
    """
    For each family, report how many distinct clusters it was split across.
    A perfectly-clustered family appears in exactly 1 cluster.
    """
    family_to_clusters: Dict[str, set] = defaultdict(set)
    family_count: Dict[str, int] = defaultdict(int)
    for cid, fam in zip(y_pred, families):
        family_to_clusters[fam].add(cid)
        family_count[fam] += 1

    rows = []
    for fam, clusters in sorted(family_to_clusters.items()):
        rows.append({
            "family": fam,
            "total_samples": family_count[fam],
            "n_clusters": len(clusters),
            "cluster_ids": sorted(clusters),
            "fragmented": len(clusters) > 1,
        })
    rows.sort(key=lambda r: r["n_clusters"], reverse=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics: dict, purity_table: list, frag_table: list):
    print()
    print("=" * 60)
    print("  Stage 3 Evaluation: Clustering Quality (MOTIF GT)")
    print("=" * 60)
    print(f"  Samples evaluated  : {metrics['n_samples']}")
    print(f"  True families      : {metrics['n_true_families']}")
    print(f"  Predicted clusters : {metrics['n_pred_clusters']}")
    print()
    print(f"  Adjusted Rand Index  (ARI) : {metrics['adjusted_rand_index']:>7.4f}")
    print(f"  Norm. Mutual Info   (NMI)  : {metrics['normalized_mutual_info']:>7.4f}")
    print(f"  Homogeneity                : {metrics['homogeneity']:>7.4f}")
    print(f"  Completeness               : {metrics['completeness']:>7.4f}")
    print(f"  V-Measure                  : {metrics['v_measure']:>7.4f}")
    print()

    # Interpretation guide
    ari = metrics["adjusted_rand_index"]
    if ari >= 0.75:
        grade = "EXCELLENT — families are tightly grouped"
    elif ari >= 0.50:
        grade = "GOOD — most families clustered together"
    elif ari >= 0.25:
        grade = "FAIR — some fragmentation"
    else:
        grade = "POOR — clusters do not align with families"
    print(f"  ARI interpretation : {grade}")
    print()

    # Top 15 clusters by purity
    print("  Top 15 Clusters by Purity:")
    print(f"  {'ClusterID':>10}  {'Size':>6}  {'Purity':>7}  Dominant Family")
    print("  " + "-" * 55)
    for row in purity_table[:15]:
        outlier_tag = " [outliers]" if row["outlier_cluster"] else ""
        print(
            f"  {row['cluster_id']:>10}  "
            f"{row['size']:>6}  "
            f"{row['purity']:>7.2%}  "
            f"{row['dominant_family']}{outlier_tag}"
        )
    print()

    # Most fragmented families
    fragmented = [r for r in frag_table if r["fragmented"]]
    if fragmented:
        print(f"  Most Fragmented Families (top 10):")
        print(f"  {'Family':<30}  {'Samples':>7}  {'Clusters':>8}")
        print("  " + "-" * 50)
        for row in fragmented[:10]:
            print(
                f"  {row['family']:<30}  "
                f"{row['total_samples']:>7}  "
                f"{row['n_clusters']:>8}"
            )
    print("=" * 60)
    print()


def save_results(output_path: Path, metrics: dict, purity_table: list, frag_table: list):
    results = {
        "evaluation": "stage3_clustering_quality",
        "dataset": "MOTIF",
        "metrics": metrics,
        "cluster_purity_table": purity_table,
        "family_fragmentation_table": frag_table,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 3 clustering quality against MOTIF ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval_motif_stage3.py \\
        --stage2-dir data/MOTIF/results/stage2 \\
        --manifest   data/MOTIF/motif_manifest.json

    # Also load Stage 3 batch report for extra cluster metadata
    python eval_motif_stage3.py \\
        --stage2-dir  data/MOTIF/results/stage2 \\
        --stage3-dir  data/MOTIF/results/stage3 \\
        --manifest    data/MOTIF/motif_manifest.json \\
        --output      data/MOTIF/eval_stage3.json
        """,
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("data/MOTIF/results/stage2"),
        help="Directory containing Stage 2 JSONs (with stage3 block written in-place)",
    )
    parser.add_argument(
        "--stage3-dir",
        type=Path,
        default=None,
        help="Optional: Stage 3 output dir (for batch report metadata)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/MOTIF/motif_manifest.json"),
        help="motif_manifest.json from download_motif.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/MOTIF/eval_stage3.json"),
        help="Where to save the evaluation JSON (default: data/MOTIF/eval_stage3.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    for p, name in [(args.stage2_dir, "--stage2-dir"), (args.manifest, "--manifest")]:
        if not p.exists():
            logger.error(f"{name} not found: {p}")
            sys.exit(1)

    # Load data
    manifest = load_manifest(args.manifest)
    records = load_cluster_assignments_from_stage2(args.stage2_dir)

    if not records:
        logger.error("No cluster assignment records found. Has Stage 3 been run?")
        sys.exit(1)

    # Align predictions with ground truth
    y_pred, y_true, families, sha256s = build_aligned_arrays(records, manifest)
    if len(y_pred) < 10:
        logger.error(
            f"Only {len(y_pred)} aligned samples — too few to compute meaningful metrics. "
            "Check that the manifest sha256s match the Stage 2 JSON filenames."
        )
        sys.exit(1)

    # Compute metrics
    metrics = compute_ari_metrics(y_pred, y_true)
    purity_table = cluster_purity_table(y_pred, families)
    frag_table = family_fragmentation_table(y_pred, families)

    # Print & save
    print_summary(metrics, purity_table, frag_table)
    save_results(args.output, metrics, purity_table, frag_table)

    # Return non-zero exit code if ARI is very poor (useful in CI)
    if metrics["adjusted_rand_index"] < 0.1:
        logger.warning("ARI < 0.10 — clustering quality is very poor.")
        sys.exit(2)


if __name__ == "__main__":
    main()
