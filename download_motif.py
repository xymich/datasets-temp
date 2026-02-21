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
MOTIF Dataset Downloader & Preprocessor

Downloads the Booz Allen Hamilton MOTIF dataset, extracts the PE files,
and re-arms disarmed PE headers so Ghidra / pefile can process them.

Steps:
  1. git clone https://github.com/boozallen/MOTIF
  2. Extract MOTIF.7z with 7-Zip using the published password
  3. Scan every extracted file: if first 2 bytes are NOT MZ, restore them
  4. Verify extraction counts against expected 3,095 samples

Usage:
    python download_motif.py
    python download_motif.py --output data/MOTIF --7z "C:/Program Files/7-Zip/7z.exe"
    python download_motif.py --skip-clone              # if repo already cloned
    python download_motif.py --skip-extract            # if already extracted
    python download_motif.py --skip-rearm              # skip header restoration
"""

import argparse
import hashlib
import json
import logging
import shutil
import struct
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MOTIF_REPO_URL = "https://github.com/boozallen/MOTIF.git"
MOTIF_ARCHIVE_PASSWORD = "i_assume_all_risk_opening_malware"
EXPECTED_SAMPLE_COUNT = 3095

# Typical PE magic bytes that tools like DIE may zero out
MZ_MAGIC = b"MZ"
PE_SIGNATURE = b"PE\x00\x00"


def find_7zip() -> str:
    """Locate 7z.exe on PATH or common install locations."""
    # Try PATH first
    if shutil.which("7z"):
        return "7z"
    if shutil.which("7za"):
        return "7za"
    # Common Windows install locations
    candidates = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def clone_repo(repo_url: str, dest: Path) -> bool:
    """Clone the MOTIF repo if not already present."""
    if (dest / ".git").exists():
        logger.info(f"Repo already cloned at {dest}, pulling latest...")
        result = subprocess.run(
            ["git", "-C", str(dest), "pull"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.warning(f"git pull failed (non-fatal): {result.stderr.strip()}")
        return True

    logger.info(f"Cloning {repo_url} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"git clone failed:\n{result.stderr}")
        return False
    logger.info("Clone complete.")
    return True


def find_archive(repo_dir: Path) -> Path:
    """Locate the MOTIF.7z archive inside the cloned repo."""
    candidates = list(repo_dir.rglob("MOTIF.7z"))
    if candidates:
        return candidates[0]
    # Also look for a split archive (.7z.001)
    candidates = list(repo_dir.rglob("MOTIF.7z.001"))
    if candidates:
        return candidates[0]
    return None


def extract_archive(archive_path: Path, dest: Path, sevenz_exe: str, password: str) -> bool:
    """Extract the password-protected 7z archive."""
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path.name} → {dest}  (this may take a few minutes)")

    # Ensure 7z path is properly quoted if it contains spaces
    cmd = [
        str(Path(sevenz_exe)),  # Convert to Path then back to str to normalize
        "x",
        str(archive_path),
        f"-p{password}",
        f"-o{dest}",
        "-aoa",   # overwrite all without prompt
        "-bso0",  # suppress stdout
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, shell=False)
    if result.returncode not in (0, 1):  # 7z returns 1 for warnings which is OK
        logger.error(f"7z extraction failed (code {result.returncode}):\n{result.stderr}")
        return False
    logger.info("Extraction complete.")
    return True


def is_disarmed_pe(file_path: Path) -> bool:
    """Return True if the file looks like a disarmed PE (bad/zeroed MZ magic)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(2)
        # Disarmed if not MZ and file is big enough to be a PE
        return len(header) == 2 and header != MZ_MAGIC and file_path.stat().st_size > 256
    except Exception:
        return False


def rearm_pe(file_path: Path) -> bool:
    """
    Restore MZ magic bytes at offset 0.
    The authors only zeroed the first 2 bytes; the rest of the PE structure
    (e_lfanew pointer → PE signature) is intact.
    """
    try:
        with open(file_path, "r+b") as f:
            f.seek(0)
            f.write(MZ_MAGIC)
        return True
    except Exception as e:
        logger.warning(f"  Could not re-arm {file_path.name}: {e}")
        return False


def rearm_all(pe_dir: Path) -> dict:
    """
    Walk pe_dir, find disarmed files, restore MZ headers.
    Returns statistics dict.
    """
    stats = {"total": 0, "already_ok": 0, "rearmed": 0, "failed": 0}

    all_files = [p for p in pe_dir.rglob("*") if p.is_file()]
    logger.info(f"Scanning {len(all_files)} extracted files for disarmed PE headers...")

    for fp in all_files:
        stats["total"] += 1
        if fp.stat().st_size < 64:
            continue  # Too small to be a PE

        if is_disarmed_pe(fp):
            if rearm_pe(fp):
                stats["rearmed"] += 1
            else:
                stats["failed"] += 1
        else:
            stats["already_ok"] += 1

    return stats


def verify_mz_headers(pe_dir: Path) -> dict:
    """Quick pass to count how many files now have valid MZ headers."""
    valid = 0
    invalid = 0
    for fp in pe_dir.rglob("*"):
        if not fp.is_file() or fp.stat().st_size < 64:
            continue
        try:
            with open(fp, "rb") as f:
                magic = f.read(2)
            if magic == MZ_MAGIC:
                valid += 1
            else:
                invalid += 1
        except Exception:
            invalid += 1
    return {"valid_mz": valid, "invalid_mz": invalid}


def load_motif_ground_truth(repo_dir: Path) -> list:
    """Load motif_dataset.jsonl for ground truth labels."""
    gt_candidates = list(repo_dir.rglob("motif_dataset.jsonl"))
    if not gt_candidates:
        logger.warning("motif_dataset.jsonl not found in repo.")
        return []
    records = []
    with open(gt_candidates[0], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info(f"Loaded {len(records)} ground truth records from motif_dataset.jsonl")
    return records


def build_hash_index(pe_dir: Path, gt_records: list) -> dict:
    """
    Check how many extracted files match MD5 hashes in the ground truth.
    MOTIF files are named MOTIF_<md5>, so we extract MD5 from filename.
    Returns {md5: Path} for matched files.
    """
    logger.info("Building MD5 index of extracted files from filenames...")
    hash_to_path = {}
    for fp in pe_dir.rglob("MOTIF_*"):
        if not fp.is_file() or fp.stat().st_size < 64:
            continue
        # Extract MD5 from filename: MOTIF_<md5>
        filename = fp.stem  # e.g., "MOTIF_001d216ee755f0bc96125892e2fb3e3a"
        if filename.startswith("MOTIF_"):
            md5 = filename[6:]  # Skip "MOTIF_" prefix
            hash_to_path[md5.lower()] = fp

    # Cross-check with ground truth (uses "md5" field)
    gt_hashes = set()
    for rec in gt_records:
        md5 = rec.get("md5", "")
        if md5:
            gt_hashes.add(md5.lower())

    matched = len(gt_hashes & set(hash_to_path.keys()))
    logger.info(
        f"  Extracted files: {len(hash_to_path)}  |  "
        f"GT MD5s: {len(gt_hashes)}  |  "
        f"Matched: {matched}"
    )
    return hash_to_path


def save_manifest(output_dir: Path, pe_dir: Path, gt_records: list, rearm_stats: dict):
    """Save motif_manifest.json for use by motif_pipeline.py."""
    # Build MD5-to-truth lookup using "md5" field from ground truth
    md5_to_truth = {}
    for rec in gt_records:
        md5 = rec.get("md5", "")
        if md5:
            md5_to_truth[md5.lower()] = rec

    entries = []
    for fp in sorted(pe_dir.rglob("MOTIF_*")):
        if not fp.is_file() or fp.stat().st_size < 64:
            continue
        
        # Extract MD5 from filename: MOTIF_<md5>
        filename = fp.stem
        if not filename.startswith("MOTIF_"):
            continue
        md5 = filename[6:].lower()
        
        # Optional: compute SHA-256 for reference
        try:
            sha256 = hashlib.sha256(fp.read_bytes()).hexdigest()
        except Exception:
            sha256 = "error"
        
        # Match against ground truth using MD5
        gt = md5_to_truth.get(md5, {})
        entries.append({
            "path": str(fp.resolve()),
            "md5": md5,
            "sha256": sha256,
            "family": gt.get("reported_family") or "unknown",
            "reported_family": gt.get("reported_family") or "unknown",
            "aliases": gt.get("aliases") or [],
            "label": gt.get("label") or "unknown",
            "gt_available": bool(gt),
        })

    manifest = {
        "total_files": len(entries),
        "rearm_stats": rearm_stats,
        "pe_dir": str(pe_dir.resolve()),
        "entries": entries,
    }
    manifest_path = output_dir / "motif_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved → {manifest_path}  ({len(entries)} entries)")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Download, extract, and pre-process the MOTIF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_motif.py
    python download_motif.py --output data/MOTIF
    python download_motif.py --skip-clone          # repo already present
    python download_motif.py --skip-extract        # 7z already done
    python download_motif.py --no-hash-index       # skip slow hash scan
        """,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/MOTIF"),
        help="Root output directory (default: data/MOTIF)",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=None,
        help="Where to clone the MOTIF repo (default: <output>/repo)",
    )
    parser.add_argument(
        "--7z",
        dest="sevenz",
        type=str,
        default=None,
        help="Path to 7z.exe (auto-detected if not supplied)",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip git clone step (repo already present)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip 7z extraction step (already extracted)",
    )
    parser.add_argument(
        "--skip-rearm",
        action="store_true",
        help="Skip MZ header restoration step",
    )
    parser.add_argument(
        "--no-hash-index",
        action="store_true",
        help="Skip SHA-256 cross-check (faster but no match stats)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir: Path = args.output
    repo_dir: Path = args.repo_dir or (output_dir / "repo")
    pe_dir: Path = output_dir / "samples"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Clone ─────────────────────────────────────────────────
    if not args.skip_clone:
        if not clone_repo(MOTIF_REPO_URL, repo_dir):
            logger.error("Clone failed. Aborting.")
            sys.exit(1)
    else:
        logger.info(f"Skipping clone (--skip-clone). Using {repo_dir}")
        if not repo_dir.exists():
            logger.error(f"Repo directory not found: {repo_dir}")
            sys.exit(1)

    # ── Step 2: Locate archive ────────────────────────────────────────
    archive_path = None
    if not args.skip_extract:
        archive_path = find_archive(repo_dir)
        if archive_path is None:
            logger.error(
                "Could not find MOTIF.7z inside the cloned repo.\n"
                "Check that the clone succeeded and the file exists in the repository."
            )
            sys.exit(1)
        logger.info(f"Found archive: {archive_path}")

    # ── Step 3: Extract ───────────────────────────────────────────────
    if not args.skip_extract and archive_path:
        sevenz = args.sevenz or find_7zip()
        if sevenz is None:
            logger.error(
                "7z.exe not found on PATH or common locations.\n"
                "Install 7-Zip from https://www.7-zip.org/ or pass --7z <path>"
            )
            sys.exit(1)
        logger.info(f"Using 7-Zip: {sevenz}")
        if not extract_archive(archive_path, pe_dir, sevenz, MOTIF_ARCHIVE_PASSWORD):
            logger.error("Extraction failed. Aborting.")
            sys.exit(1)
    elif not args.skip_extract:
        logger.error("Archive not found but extraction was requested.")
        sys.exit(1)
    else:
        logger.info(f"Skipping extraction (--skip-extract). Assuming files are in {pe_dir}")
        if not pe_dir.exists():
            logger.error(f"PE directory not found: {pe_dir}")
            sys.exit(1)

    # ── Step 4: Re-arm PE headers ─────────────────────────────────────
    rearm_stats = {"total": 0, "already_ok": 0, "rearmed": 0, "failed": 0}
    if not args.skip_rearm:
        rearm_stats = rearm_all(pe_dir)
        logger.info(
            f"Re-arm results: {rearm_stats['rearmed']} restored, "
            f"{rearm_stats['already_ok']} already valid, "
            f"{rearm_stats['failed']} failed"
        )
        # Verify
        mz_check = verify_mz_headers(pe_dir)
        logger.info(
            f"MZ header check: {mz_check['valid_mz']} valid, "
            f"{mz_check['invalid_mz']} still invalid"
        )
    else:
        logger.info("Skipping re-arm step (--skip-rearm).")

    # ── Step 5: Load ground truth ─────────────────────────────────────
    gt_records = load_motif_ground_truth(repo_dir)

    # ── Step 6: Optional hash cross-check ────────────────────────────
    if not args.no_hash_index and gt_records:
        build_hash_index(pe_dir, gt_records)

    # ── Step 7: Save manifest ─────────────────────────────────────────
    manifest_path = save_manifest(output_dir, pe_dir, gt_records, rearm_stats)

    # ── Summary ───────────────────────────────────────────────────────
    total_files = sum(1 for p in pe_dir.rglob("*") if p.is_file() and p.stat().st_size > 64)
    logger.info("")
    logger.info("=" * 60)
    logger.info("MOTIF Download & Pre-processing Complete")
    logger.info("=" * 60)
    logger.info(f"  PE files in {pe_dir}: {total_files}")
    logger.info(f"  Expected:              {EXPECTED_SAMPLE_COUNT}")
    if total_files < EXPECTED_SAMPLE_COUNT * 0.9:
        logger.warning(
            f"  WARNING: Extracted fewer files than expected "
            f"({total_files} vs {EXPECTED_SAMPLE_COUNT})"
        )
    logger.info(f"  Ground truth records:  {len(gt_records)}")
    logger.info(f"  Manifest:              {manifest_path}")
    logger.info("")
    logger.info("Next step:")
    logger.info("  python motif_pipeline.py --manifest data/MOTIF/motif_manifest.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
