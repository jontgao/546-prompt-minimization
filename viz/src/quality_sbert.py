# viz/src/quality_sbert.py
import argparse, json
from pathlib import Path
import pandas as pd
from src.metrics import SentenceBERTCosineScorer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    init_output = meta["initial_output"]

    df = pd.read_csv(args.in_csv)

    scorer = SentenceBERTCosineScorer(model="all-MiniLM-L6-v2")
    sims = scorer.compute_score(df["output"].tolist(), [init_output]).ravel()

    df["sbert_cosine"] = sims
    df["quality_loss"] = 1.0 - df["sbert_cosine"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()