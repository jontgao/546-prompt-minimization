import argparse, json
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

def load_meta(meta_path: Path):
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    for k in ("initial_prompt", "initial_output", "tokenizer_model"):
        if k not in meta:
            raise ValueError(f"meta.json missing required key: {k}")
    return meta

def load_events(events_path: Path):
    rows = []
    with events_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for k in ("iteration", "prompt", "output"):
                if k not in obj:
                    raise ValueError(f"events.jsonl line {i} missing key: {k}")
            rows.append(obj)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to runs/<run_id> folder")
    ap.add_argument("--out", required=True, help="Where to save prepared CSV")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta = load_meta(run_dir / "meta.json")
    events = load_events(run_dir / "events.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer_model"])
    init_prompt = meta["initial_prompt"]

    init_prompt_tok_len = max(1, len(tokenizer.encode(init_prompt)))

    from src.metrics import CompressionTokenScorer
    comp_scorer = CompressionTokenScorer(tokenizer)

    records = []
    for e in events:
        prompt_tok = tokenizer.encode(e["prompt"])
        output_tok = tokenizer.encode(e["output"])
        records.append({
            "iteration": int(e["iteration"]),
            "prompt": e["prompt"],
            "output": e["output"],
            "prompt_tokens": len(prompt_tok),
            "output_tokens": len(output_tok),
            "init_prompt_tokens": init_prompt_tok_len,
            "compression_tokens": float(comp_scorer.compute_score(e["prompt"], init_prompt))
        })

    df = pd.DataFrame(records).sort_values(["iteration", "prompt_tokens"]).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    preview_path = out_path.with_suffix(".head.csv")
    df.head(10).to_csv(preview_path, index=False)
    print(f"Saved: {out_path}  (rows={len(df)})")
    print(f"Preview: {preview_path}")

if __name__ == "__main__":
    main()
