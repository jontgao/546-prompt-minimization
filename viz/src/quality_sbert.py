import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="run directory (only used for logging)")
    ap.add_argument("--in_csv", required=True, help="prepared.csv")
    ap.add_argument("--out_csv", required=True, help="prepared_quality.csv (output)")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model id")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(args.in_csv)

    # Columns we rely on
    if "output" not in df.columns:
        raise SystemExit(f"'output' column missing in {args.in_csv}")
    
    def clean(txt):
        if pd.isna(txt):
            return ""
        s = str(txt).strip()
        return "" if s.lower() in {"none", "nan", "null"} else s

    have_orig_out = "original_output" in df.columns
    have_orig_prompt = "original_prompt" in df.columns

    refs = []
    for i, row in df.iterrows():
        ref = ""
        if have_orig_out:
            ref = clean(row["original_output"])
        if not ref and have_orig_prompt:
            ref = clean(row["original_prompt"])
        refs.append(ref)

    outs = [clean(x) for x in df["output"].tolist()]

    if all(r == "" for r in refs):
        df["sbert_cosine"] = np.nan
        df["quality_loss"] = np.nan
        df.to_csv(args.out_csv, index=False)
        print(f"[quality_sbert] All references empty in {run_dir}. Wrote NaNs to {args.out_csv}")
        return

    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception as e:
        raise SystemExit(f"sentence-transformers not installed: {e}")

    model = SentenceTransformer(args.sbert)
    outs_emb = model.encode(outs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    refs_emb = model.encode(refs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sims = (outs_emb * refs_emb).sum(axis=1)
    sims = np.asarray(sims, dtype=float)
    sims = np.clip(sims, 0.0, 1.0)

    df["sbert_cosine"] = sims
    df["quality_loss"] = 1.0 - sims

    df.to_csv(args.out_csv, index=False)
    print(f"[quality_sbert] Saved: {args.out_csv}  (rows={len(df)})")

if __name__ == "__main__":
    main()