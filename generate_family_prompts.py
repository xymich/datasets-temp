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
Generate instruction-tuning prompts for malware family classification.

Input: JSONL dataset from build_motif_dataset.py (or compatible schema)
Output: train/val/test JSONL in LLM format for finetune_llm.py

Each output record contains:
  - instruction
  - input
    - output   (structured JSON with reasoning chain + family)
  - family   (ground truth canonical family)
  - aliases  (known aliases for family)
  - sample_id

Usage:
  python generate_family_prompts.py \
      --input data/MOTIF/motif_stage2_dataset_full.jsonl \
      --output-dir splits/llm_family
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def normalize_family(name: str) -> str:
    return (name or "unknown").strip().lower()


def extract_feature_summary(rec: dict, max_landmarks: int = 20, max_pseudocode_chars: int = 1000) -> str:
    features = rec.get("features", {})

    declared = features.get("declared_purpose", {})
    s1 = features.get("stage1_evidence", {})
    entropy = s1.get("entropy_analysis", {})
    imports = s1.get("import_analysis", {})
    packer = s1.get("packer_detection", {})
    stage3 = features.get("stage3", {})

    landmarks = features.get("topological_landmarks", [])
    landmark_names = []
    for lm in landmarks[:max_landmarks]:
        api = lm.get("api")
        if api:
            landmark_names.append(str(api))

    pseudo_code = features.get("pseudo_code_snippets", [])
    pseudo_parts = []
    used = 0
    for snippet in pseudo_code:
        fn_name = snippet.get("function_name", "unknown")
        code = (snippet.get("refined_code") or snippet.get("pseudo_code") or "").strip()
        if not code:
            continue
        remain = max_pseudocode_chars - used
        if remain <= 0:
            break
        take = code[:remain]
        pseudo_parts.append(f"// {fn_name}\n{take}")
        used += len(take)

    summary = {
        "sample_id": rec.get("sample_id") or rec.get("file_name") or rec.get("md5") or "unknown",
        "stage1_classification": s1.get("classification", "unknown"),
        "stage1_score": s1.get("score", 0),
        "packer": packer.get("die_packer_name") or "none",
        "max_entropy": entropy.get("max_section_entropy", 0),
        "high_entropy_sections": entropy.get("high_entropy_sections", []),
        "total_imports": imports.get("total_imports", 0),
        "critical_apis": landmark_names,
        "cluster_id": stage3.get("cluster_id", -1),
        "cluster_label": stage3.get("pseudo_label", "Unknown"),
        "declared_purpose": {
            "product_name": declared.get("product_name"),
            "file_description": declared.get("file_description"),
            "company_name": declared.get("company_name"),
        },
        "code_snippets": "\n\n".join(pseudo_parts) if pseudo_parts else "",
    }

    return json.dumps(summary, ensure_ascii=False)


def _cot_identity(declared: dict) -> str:
    """Step 1: reason about claimed identity from VersionInfo/VERSIONINFO metadata."""
    product = (declared.get("product_name") or "").strip()
    descr   = (declared.get("file_description") or "").strip()
    company = (declared.get("company_name") or "").strip()
    if not any([product, descr, company]):
        return (
            "No VersionInfo metadata present. The binary carries no embedded identity "
            "claim, which is unusual for legitimate software and increases suspicion."
        )
    parts = []
    if product:
        parts.append(f"product_name='{product}'")
    if descr:
        parts.append(f"file_description='{descr}'")
    if company:
        parts.append(f"company_name='{company}'")
    claim = ", ".join(parts)
    # Flag obvious spoofing patterns
    spoofed_keywords = ["microsoft", "adobe", "google", "windows", "update", "setup", "installer"]
    claim_lower = claim.lower()
    if any(kw in claim_lower for kw in spoofed_keywords):
        return (
            f"VersionInfo claims: {claim}. "
            "This name impersonates a well-known vendor — a common malware spoofing technique "
            "to evade user scrutiny and AppLocker policies."
        )
    return f"VersionInfo claims: {claim}. Metadata present but requires corroboration."


def _cot_capabilities(summary: dict) -> str:
    """Step 2: reason about capabilities from entropy, imports, packer, critical APIs."""
    entropy_val  = summary.get("max_entropy", 0.0)
    packer       = summary.get("packer") or "none"
    total_imports = summary.get("total_imports", 0)
    critical_apis = summary.get("critical_apis") or []
    stage1_class  = summary.get("stage1_classification", "unknown")

    parts = []

    # Entropy
    try:
        e = float(entropy_val)
        if e > 7.5:
            parts.append(
                f"Max section entropy is {e:.2f} (>7.5) — strongly indicates packing, "
                "encryption, or compressed payload; typical of shellcode loaders and ransomware."
            )
        elif e > 7.0:
            parts.append(
                f"Max section entropy is {e:.2f} (>7.0) — elevated, suggesting possible "
                "packing or self-modifying code."
            )
        elif e > 6.5:
            parts.append(f"Max section entropy is {e:.2f} — slightly elevated but within range for some legitimate code.")
        else:
            parts.append(f"Max section entropy is {e:.2f} — normal range for compiled code.")
    except (TypeError, ValueError):
        parts.append(f"Entropy: {entropy_val} (could not parse).")

    # Packer
    if packer and packer.lower() not in ("none", "unknown", ""):
        parts.append(f"Packer detected: '{packer}' — confirms obfuscation layer present.")

    # Stage 1
    s1 = stage1_class.lower()
    if "packed" in s1 or "confirmed" in s1:
        parts.append(f"Stage 1 classification '{stage1_class}' confirms packing.")
    elif "suspicious" in s1 or "evasive" in s1:
        parts.append(f"Stage 1 classification '{stage1_class}' flags suspicious/evasive behaviour.")
    elif "benign" in s1:
        parts.append(f"Stage 1 classification '{stage1_class}' — no packing detected.")

    # Imports
    if total_imports == 0:
        parts.append("Zero imports resolved — likely dynamic import resolution (GetProcAddress/LoadLibrary).")
    elif total_imports < 10:
        parts.append(f"Only {total_imports} imports — minimal IAT typical of packers/loaders.")
    else:
        parts.append(f"{total_imports} imports resolved.")

    # Critical APIs
    INJECTION_APIS  = {"VirtualAlloc", "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread",
                       "NtUnmapViewOfSection", "QueueUserAPC", "SetThreadContext"}
    CREDENTIAL_APIS = {"CryptAcquireContext", "CryptEncrypt", "BCryptEncrypt"}
    NETWORK_APIS    = {"InternetOpenUrl", "HttpOpenRequest", "WSAConnect", "socket", "connect",
                       "URLDownloadToFile", "WinHttpOpen"}
    PERSISTENCE_APIS= {"RegSetValueEx", "CreateService", "StartService", "SchtasksCreate"}

    found_inject  = INJECTION_APIS  & set(critical_apis)
    found_cred    = CREDENTIAL_APIS & set(critical_apis)
    found_net     = NETWORK_APIS    & set(critical_apis)
    found_persist = PERSISTENCE_APIS & set(critical_apis)

    if found_inject:
        parts.append(f"Process-injection APIs present: {', '.join(sorted(found_inject))} — strong indicator of code injection / hollow process.")
    if found_net:
        parts.append(f"Network APIs present: {', '.join(sorted(found_net))} — sample likely phones home or downloads secondary payload.")
    if found_persist:
        parts.append(f"Persistence APIs present: {', '.join(sorted(found_persist))} — sample installs itself for persistence.")
    if found_cred:
        parts.append(f"Crypto APIs present: {', '.join(sorted(found_cred))} — possible ransomware or credential encryption.")
    if critical_apis and not any([found_inject, found_net, found_persist, found_cred]):
        parts.append(f"Notable APIs: {', '.join(critical_apis[:8])}.")

    return " ".join(parts)


def _cot_cluster(summary: dict) -> str:
    """Step 3: reason about cluster context from Stage 3 PROUD-MAL output."""
    cluster_id    = summary.get("cluster_id", -1)
    cluster_label = (summary.get("cluster_label") or "Unknown").strip()

    if cluster_id == -1:
        return (
            "Sample was assigned to the noise cluster (id=-1) by PROUD-MAL. "
            "It does not share strong structural similarity with any known family cluster."
        )
    if cluster_label.lower() in ("unknown", ""):
        return (
            f"Cluster {cluster_id} has no pseudo-label yet. "
            "The sample shares structural similarity with other cluster members but "
            "the cluster family identity has not been resolved."
        )
    return (
        f"PROUD-MAL assigned this sample to cluster {cluster_id} "
        f"(pseudo-label: '{cluster_label}'). "
        "Samples in this cluster share similar FCG topology, import profiles, and "
        "entropy distributions — strong evidence for the predicted family."
    )


def _build_evidence_refs(summary: dict) -> list:
    """Collect short evidence token strings for the evidence_refs field."""
    refs = []
    s1 = summary.get("stage1_classification", "")
    if s1 and s1 != "unknown":
        refs.append(f"stage1:{s1}")
    e = summary.get("max_entropy", 0)
    try:
        refs.append(f"entropy:{float(e):.2f}")
    except (TypeError, ValueError):
        pass
    packer = summary.get("packer") or "none"
    if packer.lower() not in ("none", "unknown", ""):
        refs.append(f"packer:{packer}")
    ci = summary.get("cluster_id", -1)
    refs.append(f"cluster:{ci}")
    cl = summary.get("cluster_label") or ""
    if cl and cl.lower() not in ("unknown", ""):
        refs.append(f"cluster_label:{cl}")
    for api in (summary.get("critical_apis") or [])[:5]:
        refs.append(f"api:{api}")
    return refs


def build_prompt_record(rec: dict) -> dict:
    family  = normalize_family(rec.get("family") or rec.get("reported_family"))
    aliases = [normalize_family(a) for a in rec.get("aliases", []) if a]
    aliases = sorted({a for a in aliases if a and a != "unknown"})

    instruction = (
        "You are a malware reverse-engineering analyst. "
        "Given static and semantic PE evidence, perform a 4-step forensic reasoning chain and return STRICT JSON only.\n"
        "REASONING STEPS (must all be present):\n"
        "1) identify_claimed_identity\n"
        "2) analyze_capabilities\n"
        "3) context_from_cluster\n"
        "4) verdict\n\n"
        "OUTPUT SCHEMA (JSON only, no markdown):\n"
        "{\n"
        "  \"reasoning_chain\": {\n"
        "    \"identify_claimed_identity\": \"...\",\n"
        "    \"analyze_capabilities\": \"...\",\n"
        "    \"context_from_cluster\": \"...\",\n"
        "    \"verdict\": \"...\"\n"
        "  },\n"
        "  \"predicted_family\": \"<canonical family lowercase>\",\n"
        "  \"confidence\": <float 0..1>,\n"
        "  \"evidence_refs\": [\"short tokens copied from input evidence\"]\n"
        "}"
    )

    input_text = extract_feature_summary(rec)

    # Parse the summary back so we can drive reasoning from actual values
    try:
        summary = json.loads(input_text)
    except (json.JSONDecodeError, TypeError):
        summary = {}

    features = rec.get("features", {})
    declared = features.get("declared_purpose", {})

    step1 = _cot_identity(declared)
    step2 = _cot_capabilities(summary)
    step3 = _cot_cluster(summary)

    # Confidence: higher when cluster agrees and packer/entropy is clear
    confidence = 0.70
    cluster_label = (summary.get("cluster_label") or "Unknown").lower()
    if cluster_label not in ("unknown", "") and cluster_label in family:
        confidence = 0.95
    elif summary.get("cluster_id", -1) != -1:
        confidence = 0.82

    step4 = (
        f"Combining identity analysis, capability profile, and cluster context, "
        f"the most consistent family attribution is '{family}'. "
        f"Confidence: {confidence:.0%}."
    )

    output_obj = {
        "reasoning_chain": {
            "identify_claimed_identity": step1,
            "analyze_capabilities": step2,
            "context_from_cluster": step3,
            "verdict": step4,
        },
        "predicted_family": family,
        "confidence": round(confidence, 2),
        "evidence_refs": _build_evidence_refs(summary),
    }

    return {
        "instruction": instruction,
        "input": input_text,
        "output": json.dumps(output_obj, ensure_ascii=False),
        "family": family,
        "aliases": aliases,
        "sample_id": rec.get("sample_id") or rec.get("file_name") or rec.get("md5") or "unknown",
        "md5": rec.get("md5", ""),
        "sha256": rec.get("sha256", ""),
        "source": rec.get("source", "unknown"),
    }


def stratified_split(records: List[dict], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    by_family = defaultdict(list)

    for rec in records:
        fam = rec.get("family", "unknown")
        by_family[fam].append(rec)

    train, val, test = [], [], []
    for fam, rows in by_family.items():
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(rows[:n_train])
        val.extend(rows[n_train:n_train + n_val])
        test.extend(rows[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return {"train": train, "val": val, "test": test}


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate family-classification prompts for LLM fine-tuning")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL dataset (e.g., motif_stage2_dataset_full.jsonl)")
    parser.add_argument("--output-dir", type=Path, default=Path("splits/llm_family"), help="Output directory for train/val/test JSONL")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-family-count", type=int, default=2, help="Drop families with fewer samples than this")
    # Benign mixing
    parser.add_argument("--benign-input", type=Path, default=None,
                        help="Optional benign JSONL to mix in (e.g., data/benign_features/benign_features.jsonl). "
                             "Benign records are added as family=\"benign\".")
    parser.add_argument("--max-benign", type=int, default=0,
                        help="Maximum benign records to include (0 = all available, default: %(default)s)")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input dataset not found: {args.input}")
        return 1

    if args.train_ratio + args.val_ratio >= 1.0:
        logger.error("train_ratio + val_ratio must be < 1.0")
        return 1

    raw = load_jsonl(args.input)
    if not raw:
        logger.error("No records found in input dataset")
        return 1

    normalized = []
    fam_counts: Dict[str, int] = defaultdict(int)
    for rec in raw:
        fam = normalize_family(rec.get("family") or rec.get("reported_family"))
        rec["family"] = fam
        fam_counts[fam] += 1
        normalized.append(rec)

    filtered = [r for r in normalized if fam_counts[r["family"]] >= args.min_family_count and r["family"] != "unknown"]
    dropped = len(normalized) - len(filtered)

    # ── Optional benign mixing ────────────────────────────────────────────────
    benign_count = 0
    if args.benign_input:
        if not args.benign_input.exists():
            logger.error(f"Benign input not found: {args.benign_input}")
            return 1
        import random as _random
        ben_raw = load_jsonl(args.benign_input)
        if args.max_benign and args.max_benign < len(ben_raw):
            rng = _random.Random(args.seed + 99)
            ben_raw = rng.sample(ben_raw, args.max_benign)
        for rec in ben_raw:
            rec["family"] = "benign"
            # Ensure sample_id field exists
            if "sample_id" not in rec:
                rec["sample_id"] = rec.get("file_hash") or rec.get("file_name") or "unknown"
            filtered.append(rec)
        benign_count = len(ben_raw)
        logger.info(f"Added {benign_count} benign records from {args.benign_input}")
    # ─────────────────────────────────────────────────────────────────────────

    prompts = [build_prompt_record(r) for r in filtered]
    splits = stratified_split(prompts, args.train_ratio, args.val_ratio, args.seed)

    write_jsonl(args.output_dir / "train.jsonl", splits["train"])
    write_jsonl(args.output_dir / "val.jsonl", splits["val"])
    write_jsonl(args.output_dir / "test.jsonl", splits["test"])

    family_list = sorted({p["family"] for p in prompts})
    metadata = {
        "input": str(args.input),
        "benign_input": str(args.benign_input) if args.benign_input else None,
        "schema_version": "cot_family_v1",
        "total_input_records": len(raw),
        "benign_records_added": benign_count,
        "total_prompt_records": len(prompts),
        "dropped_records": dropped,
        "families": len(family_list),
        "family_list": family_list,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "min_family_count": args.min_family_count,
        "metricisable_fields": [
            "predicted_family",
            "reasoning_chain.identify_claimed_identity",
            "reasoning_chain.analyze_capabilities",
            "reasoning_chain.context_from_cluster",
            "reasoning_chain.verdict",
            "evidence_refs",
            "confidence",
        ],
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("Family prompt dataset generated")
    logger.info(f"  Output dir:    {args.output_dir}")
    logger.info(f"  Records:       {len(prompts)} ({len(prompts) - benign_count} malicious, {benign_count} benign)")
    logger.info(f"  Families:      {metadata['families']}")
    logger.info(f"  Train/Val/Test: {metadata['train']}/{metadata['val']}/{metadata['test']}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
