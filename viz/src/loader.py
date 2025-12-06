#!/usr/bin/env python3
import argparse, json, sys, os, glob
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"[loader] WARN: could not load tokenizer '{model_name}' ({e}). Falling back to 'gpt2'.", file=sys.stderr)
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("gpt2", use_fast=True)

def _tok_len(tok, text: Optional[str]) -> int:
    if not text:
        return 0
    try:
        return len(tok.encode(text))
    except Exception:
        return len(text.split())

def _read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _rows_from_marius(session_dir: Path, tok) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for runf in sorted(session_dir.glob("run-*.json")):
        j = _read_json(runf)
        rmeta = j.get("meta", {}) if isinstance(j, dict) else {}
        orig_prompt = rmeta.get("original_prompt") or rmeta.get("initial_prompt")
        orig_output = rmeta.get("original_output")
        if orig_prompt is not None:
            rows.append({
                "id": f"{runf.name}-it0",
                "file": runf.name,
                "iteration": 0,
                "prompt": orig_prompt,
                "output": orig_output,
                "original_prompt": orig_prompt,
                "original_output": orig_output,
            })

        hist = j.get("history", [])
        for idx, rec in enumerate(hist, start=1):
            p = rec.get("new_prompt") or rec.get("prompt") or orig_prompt
            o = rec.get("new_output") or rec.get("output")
            it = int(rec.get("iteration", idx))

            row = {
                "id": f"{runf.name}-it{it}",
                "file": runf.name,
                "iteration": it,
                "prompt": p,
                "output": o,
                "original_prompt": orig_prompt,
                "original_output": orig_output,
            }

            for k in ("score", "bert_score", "compression_score"):
                if k in rec:
                    row[k] = rec[k]
            rows.append(row)
    return rows

def _rows_from_karim(run_dir: Path, tok) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta = _read_json(run_dir / "meta.json") if (run_dir / "meta.json").exists() else {}

    orig_prompt = meta.get("original_prompt") or meta.get("initial_prompt")
    orig_output = meta.get("original_output")

    if orig_prompt is not None:
        rows.append({
            "id": "baseline-it0",
            "file": "events.json",
            "iteration": 0,
            "prompt": orig_prompt,
            "output": orig_output,
            "original_prompt": orig_prompt,
            "original_output": orig_output,
        })

    ev_path = run_dir / "events.json"
    if not ev_path.exists():
        return rows

    data = _read_json(ev_path)
    events = data if isinstance(data, list) else data.get("events", [])
    for i, e in enumerate(events, start=1):
        p = e.get("new_prompt") or e.get("prompt") or orig_prompt
        o = e.get("new_output") or e.get("output")
        it = int(e.get("iteration", i))
        row = {
            "id": f"events-it{it}-{i}",
            "file": "events.json",
            "iteration": it,
            "prompt": p,
            "output": o,
            "original_prompt": orig_prompt,
            "original_output": orig_output,
        }
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to one session/run directory")
    ap.add_argument("--out", required=True, help="Output CSV path (prepared.csv)")
    ap.add_argument("--model_fallback", default="gpt2", help="Tokenizer fallback model")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_name = args.model_fallback
    meta_files = list(run_dir.glob("meta.json"))
    if meta_files:
        try:
            m = _read_json(meta_files[0])
            model_name = m.get("model") or model_name
        except Exception:
            pass
    tok = _load_tokenizer(model_name)
    rows: List[Dict[str, Any]] = []
    if list(run_dir.glob("run-*.json")):
        rows = _rows_from_marius(run_dir, tok)
    else:
        rows = _rows_from_karim(run_dir, tok)

    if not rows:
        print(f"[loader] No rows parsed in {run_dir}", file=sys.stderr)
        pd.DataFrame(columns=["iteration", "prompt", "output"]).to_csv(out, index=False)
        return

    df = pd.DataFrame(rows).sort_values(["iteration", "id"]).reset_index(drop=True)

    df["prompt_tokens"] = df["prompt"].apply(lambda s: _tok_len(tok, s))
    init_tokens = (df["original_prompt"].iloc[0] if not df["original_prompt"].isna().all() else "")
    df["init_prompt_tokens"] = _tok_len(tok, init_tokens) if isinstance(init_tokens, str) else 0
    base = df["init_prompt_tokens"].replace(0, 1)
    df["compression"] = df["prompt_tokens"] / base
    df["compression_tokens"] = df["prompt_tokens"]

    df.to_csv(out, index=False)
    print(f"Saved: {out}  (rows={len(df)})")

if __name__ == "__main__":
    main()
