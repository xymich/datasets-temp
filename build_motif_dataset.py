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
Build a labeled MOTIF dataset from Stage 2 outputs.

This script joins:
- MOTIF manifest entries (ground-truth family/label metadata)
- Stage 2 JSON outputs from `motif_pipeline.py`

and writes JSONL records for model training/evaluation.

Usage:
  python build_motif_dataset.py
  python build_motif_dataset.py --stage2-dir data/MOTIF/results_full/stage2 --output data/MOTIF/motif_stage2_dataset.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> Dict[str, dict]:
    """Return md5 -> manifest entry."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    md5_to_entry = {}
    for entry in entries:
        md5 = entry.get("md5")
        if not md5:
            file_path = Path(entry.get("path", ""))
            stem = file_path.stem
            if stem.startswith("MOTIF_"):
                md5 = stem[6:]
        if md5:
            md5_to_entry[md5.lower()] = entry

    logger.info(f"Loaded {len(md5_to_entry)} manifest entries")
    return md5_to_entry


def load_stage2_records(stage2_dir: Path) -> List[Tuple[str, dict]]:
    """Load Stage 2 outputs as (md5, stage2_json)."""
    records = []
    for fp in sorted(stage2_dir.glob("MOTIF_*_stage2.json")):
        stem = fp.stem  # MOTIF_<md5>_stage2
        if not stem.startswith("MOTIF_") or not stem.endswith("_stage2"):
            continue
        md5 = stem[len("MOTIF_"):-len("_stage2")].lower()
        try:
            with open(fp, encoding="utf-8") as f:
                obj = json.load(f)
            records.append((md5, obj))
        except Exception as e:
            logger.warning(f"Failed to load {fp.name}: {e}")

    logger.info(f"Loaded {len(records)} Stage 2 records from {stage2_dir}")
    return records


def build_split(records: List[dict], train_ratio: float, seed: int) -> None:
    """Assign split=train/val stratified by family."""
    by_family: Dict[str, List[int]] = {}
    for idx, rec in enumerate(records):
        fam = rec.get("family", "unknown")
        by_family.setdefault(fam, []).append(idx)

    rng = random.Random(seed)
    for fam, idxs in by_family.items():
        rng.shuffle(idxs)
        n_train = int(len(idxs) * train_ratio)
        for i, idx in enumerate(idxs):
            records[idx]["split"] = "train" if i < n_train else "val"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MOTIF labeled dataset from Stage 2 outputs")
    parser.add_argument("--manifest", type=Path, default=Path("data/MOTIF/motif_manifest.json"), help="Path to motif manifest")
    parser.add_argument("--stage2-dir", type=Path, default=Path("data/MOTIF/results_full/stage2"), help="Path to Stage 2 output directory")
    parser.add_argument("--output", type=Path, default=Path("data/MOTIF/motif_stage2_dataset.jsonl"), help="Output JSONL path")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--require-gt", action="store_true", help="Only keep samples with ground truth labels")
    args = parser.parse_args()

    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        return 1
    if not args.stage2_dir.exists():
        logger.error(f"Stage 2 directory not found: {args.stage2_dir}")
        return 1

    md5_to_entry = load_manifest(args.manifest)
    stage2_records = load_stage2_records(args.stage2_dir)

    dataset = []
    missing_manifest = 0
    unknown_family = 0

    for md5, stage2 in stage2_records:
        meta = md5_to_entry.get(md5)
        if not meta:
            missing_manifest += 1
            continue

        family = meta.get("family") or meta.get("reported_family") or "unknown"
        if family == "unknown":
            unknown_family += 1
            if args.require_gt:
                continue

        rec = {
            "sample_id": f"MOTIF_{md5}",
            "md5": md5,
            "sha256": meta.get("sha256", ""),
            "family": family,
            "reported_family": meta.get("reported_family", family),
            "aliases": meta.get("aliases", []),
            "label": meta.get("label", "unknown"),
            "source": "MOTIF",
            "features": stage2,
        }
        dataset.append(rec)

    if not dataset:
        logger.error("No records matched between Stage 2 outputs and manifest")
        return 1

    build_split(dataset, args.train_ratio, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_count = sum(1 for r in dataset if r.get("split") == "train")
    val_count = len(dataset) - train_count

    logger.info("=" * 60)
    logger.info("MOTIF dataset build complete")
    logger.info(f"  Output:           {args.output}")
    logger.info(f"  Total records:    {len(dataset)}")
    logger.info(f"  Train:            {train_count}")
    logger.info(f"  Val:              {val_count}")
    logger.info(f"  Missing manifest: {missing_manifest}")
    logger.info(f"  Unknown family:   {unknown_family}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
