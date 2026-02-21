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
MOTIF End-to-End Pipeline Runner

Runs all 3,095 MOTIF PE files through the full Stage 1 → 2 → 3 → 4 pipeline,
with parallel workers and resume support (already-processed files are skipped).

Usage:
    python motif_pipeline.py --manifest data/MOTIF/motif_manifest.json
    python motif_pipeline.py --manifest data/MOTIF/motif_manifest.json --workers 2 --stages 1234
    python motif_pipeline.py --manifest data/MOTIF/motif_manifest.json --stages 34  # skip 1+2
    python motif_pipeline.py --manifest data/MOTIF/motif_manifest.json --dry-run

Pipeline stages:
    Stage 1 : Static triage  (s1_static_triage.py)
    Stage 2 : Semantic restoration / Ghidra FCG  (s2_semantic_restoration.py)
    Stage 3 : HDBSCAN clustering  (s3_clustering.py)   -- batch, runs ONCE at end
    Stage 4 : LLM forensic verdict  (s4_reasoning.py)  -- batch, runs ONCE at end

Stages 1 and 2 are parallelised per-file.
Stages 3 and 4 are batch operations that run after all per-file work completes.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.tool_config import load_tool_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PYTHON = sys.executable

# Resolve external tool paths once at startup (tools.cfg > env vars > PATH)
_TOOL_PATHS = load_tool_paths()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> list:
    """Load motif_manifest.json produced by download_motif.py."""
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    logger.info(f"Manifest loaded: {len(entries)} files")
    return entries


def run_cmd(cmd: List[str], label: str, timeout: int = 600) -> tuple:
    """Run a subprocess; return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.warning(f"  TIMEOUT ({timeout}s): {label}")
        return False, "", "TimeoutExpired"
    except Exception as e:
        return False, "", str(e)


def write_progress(progress_file: Path, done: set, failed: set):
    """Persist resume state to disk."""
    data = {"done": sorted(done), "failed": sorted(failed)}
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_progress(progress_file: Path) -> tuple:
    """Load resume state from disk."""
    if not progress_file.exists():
        return set(), set()
    with open(progress_file, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("done", [])), set(data.get("failed", []))


# ─────────────────────────────────────────────────────────────────────────────
# Per-file Stage 1 + 2
# ─────────────────────────────────────────────────────────────────────────────

def process_one_file(
    entry: dict,
    out_stage1: Path,
    out_stage2: Path,
    die_path: str,
    ghidra_path: Optional[str],
    ghidra_timeout: int,
    stages: str,
    dry_run: bool,
    stage1_timeout: int = 120,
    stage2_timeout: int = 600,
) -> dict:
    """
    Run Stage 1 and/or Stage 2 on a single PE file.
    Returns a result dict with keys: sha256, success, stage1_ok, stage2_ok, error
    """
    sha256 = entry["sha256"]
    pe_path = Path(entry["path"])
    result = {
        "sha256": sha256,
        "path": str(pe_path),
        "stage1_ok": None,
        "stage2_ok": None,
        "success": False,
        "error": None,
    }

    if not pe_path.exists():
        result["error"] = f"File not found: {pe_path}"
        return result

    # ── Stage 1 ───────────────────────────────────────────────────────
    if "1" in stages:
        # Check if already done (output JSON exists)
        # s1_static_triage.py outputs: {pe_file.stem}_triage.json
        s1_out = out_stage1 / f"{pe_path.stem}_triage.json"
        if not s1_out.exists():
            if dry_run:
                logger.info(f"[DRY] Stage 1: {pe_path.name}")
                result["stage1_ok"] = True
            else:
                # Create temp directory for this file (s1_static_triage.py processes folders)
                temp_dir = out_stage1.parent / "_tmp" / f"s1_{sha256[:16]}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Copy file to temp dir
                    temp_file = temp_dir / f"{pe_path.stem}.exe"
                    shutil.copy2(pe_path, temp_file)
                    
                    cmd = [
                        PYTHON, "s1_static_triage.py",
                        str(temp_dir),
                        str(out_stage1),
                        "--die-path", die_path,
                    ]
                    ok, stdout, stderr = run_cmd(cmd, f"Stage1:{sha256[:8]}", timeout=stage1_timeout + 30)
                    result["stage1_ok"] = ok
                    if not ok:
                        result["error"] = f"Stage1 failed: {stderr[:200]}"
                        return result
                finally:
                    # Cleanup temp dir
                    shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            result["stage1_ok"] = True  # already done

    # ── Stage 2 ───────────────────────────────────────────────────────
    if "2" in stages:
        # Stage 2 reads from Stage 1 output; check if already done
        # Stage 2 output naming: <sha256>_stage2.json or <name>_stage2.json
        existing_s2 = list(out_stage2.glob(f"{sha256}*_stage2.json")) + \
                      list(out_stage2.glob(f"{pe_path.stem}*_stage2.json"))
        if not existing_s2:
            if dry_run:
                logger.info(f"[DRY] Stage 2: {pe_path.name}")
                result["stage2_ok"] = True
            else:
                # Create temp directory for this file
                temp_dir = out_stage2.parent / "_tmp" / f"s2_{sha256[:16]}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Copy file to temp dir
                    temp_file = temp_dir / f"{pe_path.stem}.exe"
                    shutil.copy2(pe_path, temp_file)
                    
                    cmd = [
                        PYTHON, "s2_semantic_restoration.py",
                        str(temp_dir),
                        str(out_stage2),
                        "--stage1-dir", str(out_stage1),
                        "--no-llm",  # skip LLM refinement for speed (still does Ghidra)
                    ]
                    if ghidra_path:
                        cmd += ["--ghidra-path", ghidra_path]
                    cmd += ["--ghidra-timeout", str(ghidra_timeout)]
                    ok, stdout, stderr = run_cmd(cmd, f"Stage2:{sha256[:8]}", timeout=stage2_timeout + 60)
                    result["stage2_ok"] = ok
                    if not ok:
                        result["error"] = f"Stage2 failed: {stderr[:200]}"
                        return result
                finally:
                    # Cleanup temp dir
                    shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            result["stage2_ok"] = True  # already done

    result["success"] = True
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch Stage 3 + 4
# ─────────────────────────────────────────────────────────────────────────────

def run_stage3(out_stage2: Path, out_stage3: Path, dry_run: bool) -> bool:
    """Run HDBSCAN clustering across all Stage 2 outputs."""
    if dry_run:
        logger.info("[DRY] Stage 3: s3_clustering.py")
        return True
    logger.info("Running Stage 3 (HDBSCAN clustering)…")
    cmd = [PYTHON, "s3_clustering.py", str(out_stage2), str(out_stage3), "--mode", "prod"]
    ok, stdout, stderr = run_cmd(cmd, "Stage3", timeout=3600)
    if not ok:
        logger.error(f"Stage 3 failed:\n{stderr[:500]}")
    return ok


def run_stage4(
    out_stage2: Path,
    out_stage4: Path,
    model_path: Optional[str],
    dry_run: bool,
) -> bool:
    """Run LLM forensic verdict across all Stage 2 outputs."""
    if dry_run:
        logger.info("[DRY] Stage 4: s4_reasoning.py")
        return True
    logger.info("Running Stage 4 (LLM forensic verdict)…")
    cmd = [
        PYTHON, "s4_reasoning.py",
        str(out_stage2),
        str(out_stage4),
        "--mode", "prod",
    ]
    if model_path:
        cmd += ["--model", model_path]
    ok, stdout, stderr = run_cmd(cmd, "Stage4", timeout=0)  # no cap — may run for hours
    if not ok:
        logger.error(f"Stage 4 failed:\n{stderr[:500]}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main worker loop
# ─────────────────────────────────────────────────────────────────────────────

def run_per_file_stages(
    entries: list,
    out_stage1: Path,
    out_stage2: Path,
    die_path: str,
    ghidra_path: Optional[str],
    ghidra_timeout: int,
    stages: str,
    workers: int,
    dry_run: bool,
    progress_file: Path,
):
    """Run Stage 1 and/or 2 in parallel over all entries with resume support."""
    done, failed = load_progress(progress_file)

    pending = [e for e in entries if e["sha256"] not in done and e["sha256"] not in failed]
    logger.info(
        f"Per-file work: {len(entries)} total | "
        f"{len(done)} already done | {len(failed)} previously failed | "
        f"{len(pending)} pending"
    )

    if not pending:
        logger.info("All per-file stages already complete.")
        return done, failed

    start = time.time()
    count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_one_file,
                entry,
                out_stage1,
                out_stage2,
                die_path,
                ghidra_path,
                ghidra_timeout,
                stages,
                dry_run,
            ): entry
            for entry in pending
        }

        for future in as_completed(futures):
            entry = futures[future]
            sha256 = entry["sha256"]
            count += 1

            try:
                result = future.result()
            except Exception as exc:
                logger.error(f"  [{count}/{len(pending)}] {sha256[:12]}  EXCEPTION: {exc}")
                failed.add(sha256)
                write_progress(progress_file, done, failed)
                continue

            elapsed = time.time() - start
            rate = count / elapsed * 60  # files/min
            eta_min = (len(pending) - count) / (count / elapsed) / 60 if count > 0 else 0

            if result["success"]:
                done.add(sha256)
                logger.info(
                    f"  [{count}/{len(pending)}] ✓ {sha256[:12]}  "
                    f"| {rate:.1f} files/min | ETA {eta_min:.0f} min"
                )
            else:
                failed.add(sha256)
                logger.warning(
                    f"  [{count}/{len(pending)}] ✗ {sha256[:12]}  "
                    f"error: {result.get('error', 'unknown')}"
                )

            # Persist progress every 10 files
            if count % 10 == 0:
                write_progress(progress_file, done, failed)

    write_progress(progress_file, done, failed)
    logger.info(
        f"Per-file stages complete: {len(done)} succeeded, {len(failed)} failed "
        f"({time.time() - start:.0f}s elapsed)"
    )
    return done, failed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline on the MOTIF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline, 2 parallel workers
    python motif_pipeline.py --manifest data/MOTIF/motif_manifest.json

    # Only Stages 3+4 (Stages 1+2 already done)
    python motif_pipeline.py --manifest data/MOTIF/motif_manifest.json --stages 34

    # Dry run to see what would be processed
    python motif_pipeline.py --manifest data/MOTIF/motif_manifest.json --dry-run
        """,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/MOTIF/motif_manifest.json"),
        help="Path to motif_manifest.json from download_motif.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/MOTIF/results"),
        help="Root output directory (default: data/MOTIF/results)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers for Stage 1+2 (default: 2)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="1234",
        help="Which stages to run, e.g. '34' to skip Stages 1+2 (default: 1234)",
    )
    parser.add_argument(
        "--die-path",
        type=str,
        default=_TOOL_PATHS.die_path,
        help="Path to DIE executable (overrides tools.cfg / PMD_DIE_PATH env var)",
    )
    parser.add_argument(
        "--ghidra-path",
        type=str,
        default=_TOOL_PATHS.ghidra_path,
        help="Path to Ghidra analyzeHeadless script (overrides tools.cfg / PMD_GHIDRA_PATH env var)",
    )
    parser.add_argument(
        "--ghidra-timeout",
        type=int,
        default=300,
        help="Per-file Ghidra timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="GGUF model path override for Stage 4",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without executing anything",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Validate manifest ──────────────────────────────────────────────
    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        logger.error("Run download_motif.py first to generate the manifest.")
        sys.exit(1)

    entries = load_manifest(args.manifest)
    if not entries:
        logger.error("Manifest contains no entries.")
        sys.exit(1)

    # ── Set up output directories ──────────────────────────────────────
    output = args.output
    out_stage1 = output / "stage1"
    out_stage2 = output / "stage2"
    out_stage3 = output / "stage3"
    out_stage4 = output / "stage4"
    for d in [out_stage1, out_stage2, out_stage3, out_stage4]:
        d.mkdir(parents=True, exist_ok=True)

    progress_file = output / "motif_progress.json"

    logger.info("=" * 60)
    logger.info("MOTIF Pipeline Runner")
    logger.info("=" * 60)
    logger.info(f"  Manifest:   {args.manifest}")
    logger.info(f"  Samples:    {len(entries)}")
    logger.info(f"  Stages:     {args.stages}")
    logger.info(f"  Workers:    {args.workers}")
    logger.info(f"  Output:     {output}")
    if args.dry_run:
        logger.info("  MODE:       DRY RUN (no commands executed)")
    logger.info("=" * 60)

    # ── Stages 1 + 2 (per-file, parallel) ─────────────────────────────
    if "1" in args.stages or "2" in args.stages:
        run_per_file_stages(
            entries=entries,
            out_stage1=out_stage1,
            out_stage2=out_stage2,
            die_path=args.die_path,
            ghidra_path=args.ghidra_path,
            ghidra_timeout=args.ghidra_timeout,
            stages=args.stages,
            workers=args.workers,
            dry_run=args.dry_run,
            progress_file=progress_file,
        )

    # ── Stage 3 (batch clustering) ─────────────────────────────────────
    if "3" in args.stages:
        ok = run_stage3(out_stage2, out_stage3, args.dry_run)
        if not ok:
            logger.error("Stage 3 failed. Continuing to Stage 4 if requested.")

    # ── Stage 4 (batch LLM verdict) ────────────────────────────────────
    if "4" in args.stages:
        ok = run_stage4(out_stage2, out_stage4, args.model, args.dry_run)
        if not ok:
            logger.error("Stage 4 failed.")

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline run complete.")
    logger.info(f"  Stage 1/2 outputs → {out_stage2}")
    logger.info(f"  Stage 3 output    → {out_stage3}")
    logger.info(f"  Stage 4 verdicts  → {out_stage4}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  python eval_motif_stage3.py --stage3-dir data/MOTIF/results/stage3 --manifest data/MOTIF/motif_manifest.json")
    logger.info("  python eval_motif_stage4.py --verdicts-dir data/MOTIF/results/stage4 --manifest data/MOTIF/motif_manifest.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
