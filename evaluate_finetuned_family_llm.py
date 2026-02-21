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
Evaluate a fine-tuned causal LLM on malware family classification prompts.

Expected test format: JSONL records with keys
  - instruction
  - input
  - output  (ground-truth canonical family)
  - aliases (optional list)

Usage:
  python evaluate_finetuned_family_llm.py \
      --model-dir models/llm_family \
      --test-file splits/llm_family/test.jsonl \
      --repo-dir data/MOTIF/repo
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_family(s: str) -> str:
    return (s or "").strip().lower().replace('"', "").replace("'", "")


def parse_family_pred(text: str) -> str:
    text = normalize_family(text)
    if not text:
        return "unknown"

    # JSON-like output support: {"family":"xyz"}
    m = re.search(r'"family"\s*:\s*"([a-z0-9_\-\.]+)"', text)
    if m:
        return normalize_family(m.group(1))

    # first likely token
    m = re.search(r"[a-z][a-z0-9_\-\.]{1,40}", text)
    if m:
        return normalize_family(m.group(0))

    return "unknown"


def parse_structured_response(text: str) -> dict:
    """
    Parse model response into a dict if possible.
    Accepts pure JSON or JSON embedded in extra text.
    """
    text = (text or "").strip()
    if not text:
        return {}

    # Try direct parse first
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Try to extract first JSON object span
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        blob = text[start:end + 1]
        try:
            obj = json.loads(blob)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def has_nonempty_str(d: dict, key: str) -> bool:
    val = d.get(key)
    return isinstance(val, str) and bool(val.strip())


def reasoning_metrics(resp: dict, input_text: str) -> dict:
    """Return metricisable checks for CoT schema compliance and grounding."""
    chain = resp.get("reasoning_chain") if isinstance(resp, dict) else None
    if not isinstance(chain, dict):
        chain = {}

    required = [
        "identify_claimed_identity",
        "analyze_capabilities",
        "context_from_cluster",
        "verdict",
    ]
    present = sum(1 for k in required if has_nonempty_str(chain, k))
    full_chain = present == len(required)

    refs = resp.get("evidence_refs", []) if isinstance(resp, dict) else []
    if not isinstance(refs, list):
        refs = []

    input_l = (input_text or "").lower()
    grounded_hits = 0
    for r in refs:
        token = normalize_family(str(r))
        if token and token in input_l:
            grounded_hits += 1

    grounded_ratio = (grounded_hits / len(refs)) if refs else 0.0
    return {
        "has_reasoning_chain": bool(chain),
        "reasoning_steps_present": present,
        "full_reasoning_chain": full_chain,
        "n_evidence_refs": len(refs),
        "evidence_grounded_ratio": grounded_ratio,
    }


def load_alias_lookup(repo_dir: Path) -> Dict[str, str]:
    csv_candidates = list(repo_dir.rglob("motif_families.csv")) if repo_dir.exists() else []
    if not csv_candidates:
        return {}

    lookup: Dict[str, str] = {}
    with open(csv_candidates[0], encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = normalize_family(
                row.get("family") or row.get("canonical") or row.get("name") or row.get("Family") or ""
            )
            if not canonical:
                continue
            lookup[canonical] = canonical
            aliases_raw = row.get("aliases") or row.get("Aliases") or row.get("alias") or ""
            if aliases_raw:
                for a in re.split(r"[,;|]", aliases_raw):
                    a = normalize_family(a)
                    if a:
                        lookup[a] = canonical
    return lookup


def alias_match(pred: str, truth: str, alias_lookup: Dict[str, str], sample_aliases: List[str]) -> bool:
    pred = normalize_family(pred)
    truth = normalize_family(truth)

    if pred == truth:
        return True

    # Sample-local aliases from dataset
    for a in sample_aliases or []:
        if pred == normalize_family(a):
            return True

    if not alias_lookup:
        return False

    pred_can = alias_lookup.get(pred, pred)
    truth_can = alias_lookup.get(truth, truth)
    return pred_can == truth_can


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction or ""
    input_text = input_text or ""
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load_model_and_tokenizer(model_dir: Path):
    import torch
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not found. Install with: pip install transformers")
        raise

    # Supports both normal CausalLM checkpoints and PEFT adapter checkpoints.
    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        try:
            from peft import AutoPeftModelForCausalLM
            logger.info("Detected PEFT adapter checkpoint")
            model = AutoPeftModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            return model, tokenizer
        except Exception as e:
            logger.warning(f"Failed loading as PEFT adapter, falling back to AutoModelForCausalLM: {e}")

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else None, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned family classification LLM")
    parser.add_argument("--model-dir", type=Path, required=True, help="Fine-tuned model/checkpoint directory")
    parser.add_argument("--test-file", type=Path, required=True, help="Test JSONL generated by generate_family_prompts.py")
    parser.add_argument("--repo-dir", type=Path, default=Path("data/MOTIF/repo"), help="MOTIF repo dir for alias table")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples for quick eval (0=all)")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output", type=Path, default=Path("data/MOTIF/eval_family_llm.json"), help="Output JSON metrics")
    args = parser.parse_args()

    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        return 1
    if not args.test_file.exists():
        logger.error(f"Test file not found: {args.test_file}")
        return 1

    rows = load_jsonl(args.test_file)
    if not rows:
        logger.error("No test records found")
        return 1

    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]

    alias_lookup = load_alias_lookup(args.repo_dir)

    try:
        import torch
    except ImportError:
        logger.error("torch not found. Install with: pip install torch")
        return 1

    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    exact = 0
    alias_ok = 0
    unknown = 0
    json_ok = 0
    full_chain_ok = 0
    chain_steps_total = 0
    grounded_ratio_sum = 0.0
    details = []

    logger.info(f"Evaluating {len(rows)} samples on {device}...")

    with torch.no_grad():
        for idx, rec in enumerate(rows, start=1):
            prompt = build_prompt(rec.get("instruction", ""), rec.get("input", ""))
            gt = normalize_family(rec.get("output") or rec.get("family") or "unknown")

            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

            gen_ids = out[0][enc["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            parsed = parse_structured_response(gen_text)
            if parsed:
                json_ok += 1

            pred = normalize_family(parsed.get("predicted_family", "")) if parsed else ""
            if not pred:
                pred = parse_family_pred(gen_text)

            if pred == "unknown":
                unknown += 1

            rm = reasoning_metrics(parsed, rec.get("input", "")) if parsed else {
                "has_reasoning_chain": False,
                "reasoning_steps_present": 0,
                "full_reasoning_chain": False,
                "n_evidence_refs": 0,
                "evidence_grounded_ratio": 0.0,
            }
            if rm["full_reasoning_chain"]:
                full_chain_ok += 1
            chain_steps_total += rm["reasoning_steps_present"]
            grounded_ratio_sum += rm["evidence_grounded_ratio"]

            is_exact = pred == gt
            is_alias = alias_match(pred, gt, alias_lookup, rec.get("aliases", []))

            exact += int(is_exact)
            alias_ok += int(is_alias)

            if idx <= 20:
                details.append({
                    "sample_id": rec.get("sample_id", "unknown"),
                    "ground_truth": gt,
                    "prediction": pred,
                    "exact_match": is_exact,
                    "alias_match": is_alias,
                    "parsed_json": bool(parsed),
                    "reasoning_steps_present": rm["reasoning_steps_present"],
                    "full_reasoning_chain": rm["full_reasoning_chain"],
                    "evidence_grounded_ratio": round(rm["evidence_grounded_ratio"], 4),
                    "raw_response": gen_text,
                })

            if idx % 50 == 0:
                logger.info(f"  {idx}/{len(rows)} processed")

    metrics = {
        "n_samples": len(rows),
        "exact_accuracy": round(exact / len(rows), 4),
        "alias_accuracy": round(alias_ok / len(rows), 4),
        "unknown_prediction_rate": round(unknown / len(rows), 4),
        "json_parse_rate": round(json_ok / len(rows), 4),
        "full_reasoning_chain_rate": round(full_chain_ok / len(rows), 4),
        "avg_reasoning_steps_present": round(chain_steps_total / len(rows), 4),
        "avg_evidence_grounded_ratio": round(grounded_ratio_sum / len(rows), 4),
        "model_dir": str(args.model_dir),
        "test_file": str(args.test_file),
    }

    report = {"metrics": metrics, "examples": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("Family LLM evaluation complete")
    logger.info(f"  Exact accuracy: {metrics['exact_accuracy']:.4f}")
    logger.info(f"  Alias accuracy: {metrics['alias_accuracy']:.4f}")
    logger.info(f"  Unknown rate:   {metrics['unknown_prediction_rate']:.4f}")
    logger.info(f"  Report:         {args.output}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
