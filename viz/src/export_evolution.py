#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def _coerce_numeric(df, cols):
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _ensure_losses(df, qloss_col, w_comp, w_qloss):
    if "compression" not in df.columns:
        if {"prompt_tokens", "init_prompt_tokens"} <= set(df.columns):
            df["compression"] = df["prompt_tokens"] / df["init_prompt_tokens"]

    if qloss_col not in df.columns or df[qloss_col].isna().all():
        if "sbert_cosine" in df.columns:
            df[qloss_col] = 1.0 - df["sbert_cosine"].clip(0, 1)
        else:
            df[qloss_col] = np.nan
    else:
        if "sbert_cosine" in df.columns:
            df[qloss_col] = df[qloss_col].fillna(1.0 - df["sbert_cosine"].clip(0, 1))

    if "total_loss" not in df.columns:
        df["total_loss"] = w_comp * df.get("compression") + w_qloss * df.get(qloss_col)

    return df


def _baseline_row(df, qloss_col, w_comp, w_qloss):
    if "original_prompt" in df.columns and df["original_prompt"].notna().any():
        base_prompt = str(df["original_prompt"].dropna().iloc[0])
    else:
        try:
            imin = int(df["iteration"].min())
            base_prompt = str(df.loc[df["iteration"] == imin, "prompt"].dropna().iloc[0])
        except Exception:
            base_prompt = "Original prompt (baseline)"

    if w_qloss > 0:
        ql = max(0.0, (1.0 - w_comp * 1.0) / w_qloss)
    else:
        ql = np.nan

    row = {
        "iteration": 0,
        "compression": 1.0,
        qloss_col: ql,
        "total_loss": 1.0,
        "prompt": base_prompt,
        "original_prompt": base_prompt,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="prepared_quality.csv")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--y_metric", default="total_loss")
    ap.add_argument("--qloss_col", default="quality_loss")
    ap.add_argument("--w_comp", type=float, default=1.0)
    ap.add_argument("--w_qloss", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=3, help="top-K per iteration to include")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    _coerce_numeric(
        df,
        [
            "iteration",
            "compression",
            "prompt_tokens",
            "init_prompt_tokens",
            args.qloss_col,
            "sbert_cosine",
            "total_loss" if "total_loss" in df.columns else "",
        ],
    )
    df = _ensure_losses(df, args.qloss_col, args.w_comp, args.w_qloss)
    df = df[df["iteration"].notna()].copy()
    df["iteration"] = df["iteration"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df[args.y_metric].notna()].copy()

    df = df[df["iteration"] != 0].copy()
    base = _baseline_row(df, args.qloss_col, args.w_comp, args.w_qloss)

    for k in base.keys():
        if k not in df.columns:
            df[k] = np.nan
    df = pd.concat([pd.DataFrame([base]), df], ignore_index=True)

    if df.empty:
        raise SystemExit("No valid rows after filtering.")

    iter_best = (
        df.sort_values(["iteration", args.y_metric])
        .groupby("iteration", as_index=False)
        .first()
        .sort_values("iteration")
        .reset_index(drop=True)
    )

    iter_best["cum_best"] = iter_best[args.y_metric].cummin()
    events = iter_best[iter_best[args.y_metric] == iter_best["cum_best"]].copy()
    overall_best = iter_best.loc[iter_best[args.y_metric].idxmin()]

    topk = (
        df.sort_values(["iteration", args.y_metric])
        .groupby("iteration", as_index=False)
        .head(args.k)
        .copy()
    )

    def _pick_fields(row):
        out = {
            "iteration": int(row.get("iteration", -1)),
            "total_loss": float(row.get("total_loss", np.nan)),
            "compression": float(row.get("compression", np.nan)),
            "quality_loss": float(row.get(args.qloss_col, np.nan)),
            "prompt": None if pd.isna(row.get("prompt")) else str(row.get("prompt")),
        }
        for opt in ("id", "file", "prompt_tokens", "init_prompt_tokens", "sbert_cosine"):
            if opt in row and not pd.isna(row[opt]):
                out[opt] = row[opt] if opt != "sbert_cosine" else float(row[opt])
        return out

    per_iter_best = [ _pick_fields(r) for _, r in iter_best.iterrows() ]
    new_best_events = [ _pick_fields(r) for _, r in events.iterrows() ]
    overall_best_obj = _pick_fields(overall_best)

    topk_list = []
    for it, grp in topk.groupby("iteration", sort=True):
        cand = [ _pick_fields(r) for _, r in grp.iterrows() ]
        topk_list.append({"iteration": int(it), "candidates": cand})

    payload = {
        "meta": {
            "source_csv": str(Path(args.csv)),
            "y_metric": args.y_metric,
            "qloss_col": args.qloss_col,
            "w_comp": args.w_comp,
            "w_qloss": args.w_qloss,
            "top_k": args.k,
        },
        "baseline": {
            "iteration": 0,
            "total_loss": 1.0,
            "compression": 1.0,
            "quality_loss": float(base.get(args.qloss_col, np.nan)),
            "prompt": base.get("prompt"),
        },
        "per_iteration_best": per_iter_best,
        "new_best_events": new_best_events,
        "overall_best": overall_best_obj,
        "top_k_per_iteration": topk_list,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote JSON: {args.out}")

if __name__ == "__main__":
    main()