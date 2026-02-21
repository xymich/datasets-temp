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
Stage 4 Evaluation: Family Classification Accuracy on MOTIF Ground Truth

Measures how accurately the LLM forensic verdict identifies the correct
malware family name, using the alias table in motif_families.csv to handle
variant names (e.g., "WannaCry" == "WannaCrypt").

Metrics computed:
  - Exact-match family accuracy
  - Alias-match family accuracy  (headline metric per prompt10-motif.md)
  - Top-k accuracy (is the correct family anywhere in the verdict text?)
  - Per-family accuracy breakdown
  - Hallucination rate (from verification block)
  - Avg judge score (from verification.judge block)
  - Confusion matrix (top N families)

Usage:
    python eval_motif_stage4.py \
        --verdicts-dir data/MOTIF/results/stage4 \
        --manifest     data/MOTIF/motif_manifest.json \
        --repo-dir     data/MOTIF/repo
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Alias table loading
# ─────────────────────────────────────────────────────────────────────────────

def load_alias_table(repo_dir: Path) -> Dict[str, List[str]]:
    """
    Load motif_families.csv (or motif_families.json) and build a mapping:
      canonical_name → [alias1, alias2, ...]
    and
      alias → canonical_name
    Both in lowercase.
    """
    # Try CSV first
    csv_candidates = list(repo_dir.rglob("motif_families.csv"))
    json_candidates = list(repo_dir.rglob("motif_families.json"))

    canonical_to_aliases: Dict[str, Set[str]] = defaultdict(set)

    if csv_candidates:
        path = csv_candidates[0]
        logger.info(f"Loading alias table from {path}")
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Common column names across MOTIF versions
                canonical = (
                    row.get("family") or row.get("canonical") or
                    row.get("name") or row.get("Family") or ""
                ).strip().lower()
                aliases_raw = (
                    row.get("aliases") or row.get("Aliases") or
                    row.get("alias") or ""
                ).strip()
                if not canonical:
                    continue
                canonical_to_aliases[canonical].add(canonical)
                if aliases_raw:
                    for a in re.split(r"[,;|]", aliases_raw):
                        a = a.strip().lower()
                        if a:
                            canonical_to_aliases[canonical].add(a)

    elif json_candidates:
        path = json_candidates[0]
        logger.info(f"Loading alias table from {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Handle both list and dict representations
        if isinstance(data, list):
            for entry in data:
                canonical = (entry.get("family") or entry.get("name") or "").lower().strip()
                if not canonical:
                    continue
                canonical_to_aliases[canonical].add(canonical)
                for a in entry.get("aliases", []):
                    canonical_to_aliases[canonical].add(str(a).lower().strip())
        elif isinstance(data, dict):
            for canonical, aliases in data.items():
                canonical = canonical.lower().strip()
                canonical_to_aliases[canonical].add(canonical)
                for a in (aliases if isinstance(aliases, list) else [aliases]):
                    canonical_to_aliases[canonical].add(str(a).lower().strip())
    else:
        logger.warning(
            "motif_families.csv / motif_families.json not found in repo. "
            "Alias matching will be disabled (exact match only)."
        )
        return {}

    total_aliases = sum(len(v) for v in canonical_to_aliases.values())
    logger.info(
        f"Alias table: {len(canonical_to_aliases)} families, "
        f"{total_aliases} total aliases"
    )
    return {k: sorted(v) for k, v in canonical_to_aliases.items()}


def build_alias_lookup(alias_table: Dict[str, List[str]]) -> Dict[str, str]:
    """alias (lower) → canonical_family (lower)"""
    lookup = {}
    for canonical, aliases in alias_table.items():
        for a in aliases:
            lookup[a.lower()] = canonical.lower()
        lookup[canonical.lower()] = canonical.lower()
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Verdict loading
# ─────────────────────────────────────────────────────────────────────────────

def load_verdicts(verdicts_dir: Path) -> List[dict]:
    """Load all *_verdict.json files from the Stage 4 output directory."""
    verdict_files = list(verdicts_dir.glob("*_verdict.json")) + \
                    list(verdicts_dir.glob("*verdict*.json"))
    # Deduplicate
    verdict_files = list({p.resolve(): p for p in verdict_files}.values())

    records = []
    skipped = 0
    for vf in verdict_files:
        try:
            with open(vf, encoding="utf-8") as f:
                data = json.load(f)
            records.append(data)
        except Exception as e:
            logger.debug(f"Skip {vf.name}: {e}")
            skipped += 1

    logger.info(f"Verdicts loaded: {len(records)} | skipped: {skipped}")
    return records


def extract_sha256_from_verdict(v: dict) -> Optional[str]:
    """Pull SHA-256 from verdict JSON (several possible key names)."""
    for key in ("file_hash", "sha256", "hash"):
        h = v.get(key)
        if h and len(h) >= 32:
            return h.lower()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Family extraction from verdict text
# ─────────────────────────────────────────────────────────────────────────────

def extract_predicted_family(verdict: dict) -> str:
    """
    Pull the predicted family name from the Stage 4 LLM verdict.
    Checked fields (in priority order):
      1. verdict.cluster_consistency
      2. verdict.verdict_summary
      3. verdict.behavioral_reasoning[-1]  (VERDICT step)
      4. reasoning_chain[-1]
      5. verdict.verdict_summary (fallback text scan against alias table)
    Returns lowercase string or "unknown".
    """
    inner = verdict.get("verdict") or {}
    if isinstance(inner, str):
        # Some schemas return verdict as a string "MALWARE"/"BENIGN"
        inner = {}

    # 1. cluster_consistency often contains family name
    cc = inner.get("cluster_consistency") or ""
    family = _parse_family_from_text(cc)
    if family:
        return family

    # 2. verdict_summary
    vs = inner.get("verdict_summary") or ""
    family = _parse_family_from_text(vs)
    if family:
        return family

    # 3. Last reasoning step (VERDICT step)
    for field in ("behavioral_reasoning", "reasoning_chain"):
        chain = inner.get(field) or []
        if chain:
            last_step = chain[-1] if isinstance(chain[-1], str) else str(chain[-1])
            family = _parse_family_from_text(last_step)
            if family:
                return family

    # 4. Full text sweep of all string fields
    full_text = json.dumps(inner).lower()
    family = _parse_family_from_text(full_text)
    if family:
        return family

    return "unknown"


# Common family name patterns found in LLM output
_FAMILY_PATTERNS = [
    # Direct "identified as X" / "classified as X" / "family X"
    r"(?:identified as|classified as|family[:\s]+|label[:\s]+|malware family[:\s]+)\s*['\"]?([a-z][a-z0-9\-_\.]{2,30})['\"]?",
    # Colon form: "family: Trickbot"
    r"family[:\s]+([a-zA-Z][a-zA-Z0-9\-_\.]{2,30})",
    # "Trickbot malware" / "Wannacry ransomware"
    r"\b([a-z][a-z0-9\-_\.]{3,20})\s+(?:malware|ransomware|trojan|worm|botnet|banker|loader|dropper|rat|backdoor|stealer|virus)\b",
    # Pseudo-label format "Pseudo:Malware:FamilyName"
    r"pseudo:[^:]+:([a-z][a-z0-9\-_\.]{2,30})",
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _FAMILY_PATTERNS]


def _parse_family_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in _COMPILED_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).lower().strip().rstrip(".")
            # Filter out generic words
            if candidate not in {
                "malware", "benign", "file", "sample", "binary",
                "unknown", "suspicious", "packed", "code", "this",
                "that", "with", "from", "into", "using", "based",
            }:
                return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

def is_alias_match(predicted: str, true_family: str, alias_lookup: Dict[str, str]) -> bool:
    """Return True if predicted resolves to the same canonical name as true_family."""
    if not alias_lookup:
        return predicted.lower() == true_family.lower()

    canonical_pred = alias_lookup.get(predicted.lower(), predicted.lower())
    canonical_true = alias_lookup.get(true_family.lower(), true_family.lower())
    return canonical_pred == canonical_true


def is_text_contains_match(verdict: dict, true_family: str, alias_lookup: Dict[str, str]) -> bool:
    """
    Looser check: does the full verdict JSON text contain the true family name
    or any of its aliases?
    """
    full_text = json.dumps(verdict).lower()
    names_to_check = {true_family.lower()}
    # Add aliases
    canonical = alias_lookup.get(true_family.lower(), true_family.lower())
    for alias, can in alias_lookup.items():
        if can == canonical:
            names_to_check.add(alias)
    return any(name in full_text for name in names_to_check if len(name) > 2)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    verdicts: List[dict],
    manifest: Dict[str, dict],
    alias_lookup: Dict[str, str],
) -> dict:
    """
    Core evaluation loop.
    Returns a comprehensive result dict.
    """
    total = 0
    exact_match = 0
    alias_match_count = 0
    contains_match_count = 0
    unknown_pred = 0

    # Verification aggregates
    hallucination_rates = []
    judge_scores = []

    per_family_stats: Dict[str, dict] = defaultdict(
        lambda: {"total": 0, "exact": 0, "alias": 0, "contains": 0}
    )

    confusion: List[Tuple[str, str]] = []  # (true, predicted)

    for v in verdicts:
        sha256 = extract_sha256_from_verdict(v)
        if not sha256:
            continue

        gt = manifest.get(sha256)
        if not gt:
            continue

        true_family = (gt.get("family") or gt.get("reported_family") or "").lower().strip()
        if not true_family or true_family == "unknown":
            continue

        total += 1
        predicted = extract_predicted_family(v)
        if predicted == "unknown":
            unknown_pred += 1

        exact = predicted.lower() == true_family.lower()
        alias = is_alias_match(predicted, true_family, alias_lookup)
        contains = is_text_contains_match(v, true_family, alias_lookup)

        if exact:
            exact_match += 1
        if alias:
            alias_match_count += 1
        if contains:
            contains_match_count += 1

        per_family_stats[true_family]["total"] += 1
        if exact:
            per_family_stats[true_family]["exact"] += 1
        if alias:
            per_family_stats[true_family]["alias"] += 1
        if contains:
            per_family_stats[true_family]["contains"] += 1

        confusion.append((true_family, predicted if predicted != "unknown" else "<unknown>"))

        # Verification metrics
        verification = v.get("verification") or {}
        hall = (verification.get("hallucination") or {}).get("hallucination_rate")
        if hall is not None:
            hallucination_rates.append(float(hall))
        judge_score = (verification.get("judge") or {}).get("score")
        if judge_score is not None:
            try:
                judge_scores.append(float(judge_score))
            except (TypeError, ValueError):
                pass

    # Per-family breakdown
    family_breakdown = []
    for fam, stats in sorted(per_family_stats.items()):
        n = stats["total"]
        family_breakdown.append({
            "family": fam,
            "total": n,
            "alias_accuracy": round(stats["alias"] / n, 4) if n else 0.0,
            "exact_accuracy": round(stats["exact"] / n, 4) if n else 0.0,
            "contains_accuracy": round(stats["contains"] / n, 4) if n else 0.0,
        })
    family_breakdown.sort(key=lambda r: r["alias_accuracy"])

    # Top-N confusion (most common misclassifications)
    errors = [(t, p) for t, p in confusion if t != p and p != "<unknown>"]
    error_counter = Counter(errors)
    top_confusions = [
        {"true": t, "predicted": p, "count": c}
        for (t, p), c in error_counter.most_common(20)
    ]

    metrics = {
        "total_evaluated": total,
        "exact_match_accuracy": round(exact_match / total, 4) if total else 0.0,
        "alias_match_accuracy": round(alias_match_count / total, 4) if total else 0.0,
        "contains_match_accuracy": round(contains_match_count / total, 4) if total else 0.0,
        "unknown_prediction_rate": round(unknown_pred / total, 4) if total else 0.0,
        "avg_hallucination_rate": round(sum(hallucination_rates) / len(hallucination_rates), 4)
            if hallucination_rates else None,
        "avg_judge_score": round(sum(judge_scores) / len(judge_scores), 4)
            if judge_scores else None,
        "n_distinct_true_families": len(per_family_stats),
    }

    return {
        "metrics": metrics,
        "per_family_accuracy": family_breakdown,
        "top_confusions": top_confusions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    m = results["metrics"]
    print()
    print("=" * 60)
    print("  Stage 4 Evaluation: Family Classification (MOTIF GT)")
    print("=" * 60)
    print(f"  Samples evaluated     : {m['total_evaluated']}")
    print(f"  Distinct families     : {m['n_distinct_true_families']}")
    print()
    print(f"  Alias-match accuracy  : {m['alias_match_accuracy']:>7.2%}  ← headline metric")
    print(f"  Exact-match accuracy  : {m['exact_match_accuracy']:>7.2%}")
    print(f"  Text-contains check   : {m['contains_match_accuracy']:>7.2%}  (upper-bound)")
    print(f"  Unknown predictions   : {m['unknown_prediction_rate']:>7.2%}")
    if m["avg_hallucination_rate"] is not None:
        print(f"  Avg hallucination rate: {m['avg_hallucination_rate']:>7.2%}")
    if m["avg_judge_score"] is not None:
        print(f"  Avg judge score (1-5) : {m['avg_judge_score']:>7.2f}")
    print()

    # Interpretation
    acc = m["alias_match_accuracy"]
    if acc >= 0.70:
        grade = "STRONG — LLM correctly identifies most families"
    elif acc >= 0.40:
        grade = "MODERATE — identifies common families, misses rare ones"
    elif acc >= 0.15:
        grade = "WEAK — limited family identification capability"
    else:
        grade = "VERY POOR — LLM not reliably predicting family names"
    print(f"  Interpretation        : {grade}")
    print()

    # Worst 10 families by alias accuracy
    bottom = sorted(results["per_family_accuracy"], key=lambda r: r["alias_accuracy"])[:10]
    print("  10 Hardest Families (lowest alias accuracy):")
    print(f"  {'Family':<25}  {'Samples':>7}  {'Alias Acc':>9}")
    print("  " + "-" * 46)
    for row in bottom:
        print(
            f"  {row['family']:<25}  "
            f"{row['total']:>7}  "
            f"{row['alias_accuracy']:>9.2%}"
        )
    print()

    # Top-10 confusions
    if results["top_confusions"]:
        print("  Top-10 Misclassifications:")
        print(f"  {'True Family':<25}  {'Predicted':<25}  {'Count':>5}")
        print("  " + "-" * 58)
        for row in results["top_confusions"][:10]:
            print(
                f"  {row['true']:<25}  "
                f"{row['predicted']:<25}  "
                f"{row['count']:>5}"
            )
    print("=" * 60)
    print()


def save_results(output_path: Path, results: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"evaluation": "stage4_family_classification", "dataset": "MOTIF", **results},
            f, indent=2
        )
    logger.info(f"Results saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 4 family classification accuracy against MOTIF ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval_motif_stage4.py \\
        --verdicts-dir data/MOTIF/results/stage4 \\
        --manifest     data/MOTIF/motif_manifest.json \\
        --repo-dir     data/MOTIF/repo

    # Without alias table (exact-match only)
    python eval_motif_stage4.py \\
        --verdicts-dir data/MOTIF/results/stage4 \\
        --manifest     data/MOTIF/motif_manifest.json
        """,
    )
    parser.add_argument(
        "--verdicts-dir",
        type=Path,
        default=Path("data/MOTIF/results/stage4"),
        help="Directory containing Stage 4 *_verdict.json files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/MOTIF/motif_manifest.json"),
        help="motif_manifest.json from download_motif.py",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("data/MOTIF/repo"),
        help="Cloned MOTIF repo directory (for motif_families.csv alias table)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/MOTIF/eval_stage4.json"),
        help="Where to save evaluation results JSON",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    for p, name in [(args.verdicts_dir, "--verdicts-dir"), (args.manifest, "--manifest")]:
        if not p.exists():
            logger.error(f"{name} not found: {p}")
            sys.exit(1)

    # Load alias table
    alias_table: Dict[str, List[str]] = {}
    alias_lookup: Dict[str, str] = {}
    if args.repo_dir.exists():
        alias_table = load_alias_table(args.repo_dir)
        alias_lookup = build_alias_lookup(alias_table)
    else:
        logger.warning(
            f"Repo dir not found ({args.repo_dir}). Alias matching disabled. "
            "Pass --repo-dir data/MOTIF/repo to enable it."
        )

    # Load data
    manifest = {e["sha256"].lower(): e for e in json.load(open(args.manifest))["entries"]}
    verdicts = load_verdicts(args.verdicts_dir)

    if not verdicts:
        logger.error("No verdict JSON files found. Has Stage 4 been run?")
        sys.exit(1)

    # Evaluate
    results = run_evaluation(verdicts, manifest, alias_lookup)

    # Report
    print_summary(results)
    save_results(args.output, results)


if __name__ == "__main__":
    main()
