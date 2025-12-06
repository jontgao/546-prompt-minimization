#!/usr/bin/env python3
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
        df["total_loss"] = w_comp * df["compression"] + w_qloss * df[qloss_col]
    return df


def _truncate_lines(text, max_lines=6, max_chars=120):
    """Keep up to max_lines; each line shortened to max_chars."""
    if text is None:
        return ""
    txt = str(text).replace("\r", "")
    lines = [textwrap.shorten(l.strip(), width=max_chars, placeholder="…")
             for l in txt.split("\n") if l.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["…"]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="prepared_quality.csv")
    ap.add_argument("--out", required=True, help="output png path")

    ap.add_argument("--y_metric", default="total_loss")
    ap.add_argument("--qloss_col", default="quality_loss")
    ap.add_argument("--w_comp", type=float, default=1.0)
    ap.add_argument("--w_qloss", type=float, default=1.0)

    ap.add_argument("--sample_strategy", choices=["topk", "all"], default="topk")
    ap.add_argument("--k", type=int, default=3)

    ap.add_argument("--fig_w", type=float, default=16.0)
    ap.add_argument("--fig_h", type=float, default=10.5)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--ylim_max", type=float, default=None)
    ap.add_argument("--max_events", type=int, default=12)

    ap.add_argument("--right_top", type=float, default=0.98, help="Top y of right panel (axes fraction)")
    ap.add_argument("--lh", type=float, default=0.032, help="Line height step in axes fraction")
    ap.add_argument("--prompt_lines", type=int, default=6, help="Max prompt lines")
    ap.add_argument("--prompt_chars", type=int, default=120, help="Max chars per prompt line")
    ap.add_argument("--timeline_prompt_lines", type=int, default=2, help="Prompt lines per timeline entry")
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
    df = df[df["iteration"].notna()]
    df["iteration"] = df["iteration"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df[args.y_metric].notna()].copy()

    baseline_prompt = None
    if "original_prompt" in df.columns and df["original_prompt"].notna().any():
        baseline_prompt = str(df["original_prompt"].dropna().iloc[0])
    else:
        imin = df["iteration"].min()
        try:
            baseline_prompt = str(
                df.loc[df["iteration"] == imin, "prompt"].dropna().iloc[0]
            )
        except Exception:
            baseline_prompt = "Original prompt (baseline)"

    df = df[df["iteration"] != 0].copy()
    if args.w_qloss > 0:
        baseline_qloss = max(0.0, (1.0 - args.w_comp * 1.0) / args.w_qloss)
    else:
        baseline_qloss = np.nan

    baseline_row = {
        "iteration": 0,
        "compression": 1.0,
        args.qloss_col: baseline_qloss,
        "total_loss": 1.0,
        "prompt": baseline_prompt,
        "original_prompt": baseline_prompt,
    }
    for k in list(baseline_row.keys()):
        if k not in df.columns:
            df[k] = np.nan
    df = pd.concat([pd.DataFrame([baseline_row]), df], ignore_index=True)

    if df.empty:
        raise SystemExit("No valid rows to plot after filtering.")

    if args.sample_strategy == "topk":
        scatter_df = (
            df.sort_values(["iteration", args.y_metric])
            .groupby("iteration", as_index=False)
            .head(args.k)
        )
    else:
        scatter_df = df

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

    fig = plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(
        scatter_df["iteration"].values,
        scatter_df[args.y_metric].values,
        s=20, alpha=0.80, label="candidates (sampled)",
    )
    ax.plot(
        iter_best["iteration"], iter_best[args.y_metric],
        "-o", color="red", lw=2, ms=3, label="per-iteration best",
    )
    ax.scatter(
        events["iteration"], events[args.y_metric],
        marker="*", s=140, color="gold", edgecolor="black",
        zorder=5, label="new best",
    )
    for _, r in events.iterrows():
        ax.annotate(
            f"it {int(r['iteration'])}\n{args.y_metric}={r[args.y_metric]:.3f}",
            xy=(r["iteration"], r[args.y_metric]),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85),
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss" if args.y_metric == "total_loss" else args.y_metric)
    ax.set_title("Evolution View (baseline at iteration 0, loss=1)")
    if args.ylim_max:
        ax.set_ylim(top=args.ylim_max)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    side = fig.add_subplot(gs[0, 1])
    side.axis("off")

    y = args.right_top
    lh = args.lh

    def put_text(txt, fs=10, bold=False, lines=None, dy=None):
        nonlocal y
        if txt is None: 
            return
        kw = {"fontsize": fs, "transform": side.transAxes}
        if bold:
            kw["fontweight"] = "bold"
        side.text(0.02, y, txt, **kw)
        used = dy if dy is not None else (lh * (lines if lines else 1))
        y -= used

    def hbar(label, val, maxv, bar_w=0.42, bar_h=0.03, pad=0.01):
        nonlocal y
        x0 = 0.02
        side.add_patch(Rectangle((x0, y), bar_w, bar_h,
                                 transform=side.transAxes,
                                 facecolor="#e8e8e8", edgecolor="none"))
        w = 0.0 if not np.isfinite(val) else (val / maxv) * bar_w
        side.add_patch(Rectangle((x0, y), w, bar_h,
                                 transform=side.transAxes,
                                 facecolor="#4c78a8", edgecolor="none"))
        side.text(x0, y + bar_h + 0.004, label, fontsize=9, transform=side.transAxes)
        side.text(x0 + bar_w + 0.01, y + bar_h/2, f"{val:.3f}",
                  va="center", fontsize=9, transform=side.transAxes)
        y -= (bar_h + pad)

    put_text("Selected Candidate (overall best)", fs=12, bold=True)
    put_text(f"Iteration: {int(overall_best['iteration'])}   "
             f"Score ({args.y_metric}): {overall_best[args.y_metric]:.3f}",
             fs=10, lines=1)

    comp = float(overall_best.get("compression", np.nan))
    qloss = float(overall_best.get(args.qloss_col, np.nan))
    tot   = float(args.w_comp * comp + args.w_qloss * qloss)
    put_text("Metrics (overall best)", fs=12, bold=True)
    maxv = float(np.nanmax([1.0, tot, comp, qloss]))
    hbar("Total",              tot,   maxv)
    hbar("Compression",        comp,  maxv)
    hbar("Semantic (quality)", qloss, maxv)
    y -= lh * 0.5
    prompt_best = _truncate_lines(
        overall_best.get("prompt", ""),
        max_lines=args.prompt_lines,
        max_chars=args.prompt_chars,
    )
    put_text("Prompt:", fs=10, bold=True)
    n_lines = prompt_best.count("\n") + 1 if prompt_best else 1
    put_text(prompt_best, fs=10, lines=n_lines)

    y -= lh * 0.5
    put_text("New-bests timeline", fs=12, bold=True)
    timeline = events.sort_values("iteration").reset_index(drop=True)
    if len(timeline) > args.max_events:
        timeline = timeline.iloc[-args.max_events:]

    for _, r in timeline.iterrows():
        it = int(r["iteration"])
        score = float(r[args.y_metric])
        put_text(f"it {it}  |  score={score:.3f}", fs=10, bold=True)
        pr = _truncate_lines(r.get("prompt", ""), max_lines=args.timeline_prompt_lines, max_chars=105)
        n = pr.count("\n") + 1 if pr else 1
        put_text(pr, fs=9, lines=n)

    if len(events) > len(timeline):
        put_text(f"… and {len(events) - len(timeline)} earlier events", fs=9, lines=1)

    plt.savefig(args.out, bbox_inches="tight")
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()