import argparse, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="prepared_quality.csv path")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--w_comp", type=float, default=0.5, help="weight for compression")
    ap.add_argument("--w_qloss", type=float, default=0.5, help="weight for quality loss")
    ap.add_argument("--qloss_col", default="quality_loss", help="column name for quality loss (e.g., quality_loss or quality_loss_bert)")
    ap.add_argument("--clusters", type=int, default=3, help="number of 'islands'")
    ap.add_argument("--label_islands", action="store_true", help="annotate island names")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.qloss_col not in df.columns:
        raise SystemExit(f"Column {args.qloss_col} not found in {args.csv}. Columns: {list(df.columns)}")

    df = df.copy()
    df["quality_loss_viz"] = df[args.qloss_col]
    df["combined"] = args.w_comp*df["compression_tokens"] + args.w_qloss*df["quality_loss_viz"]

    X = df[["compression_tokens","quality_loss_viz"]].values
    k = min(args.clusters, len(df))
    if k > 1:
        labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(X)
    else:
        labels = np.zeros(len(df), dtype=int)
    df["island"] = labels

    edges = []
    iters = sorted(df["iteration"].unique())
    for it in iters:
        if it == iters[0]:
            continue
        cur = df[df["iteration"]==it][["compression_tokens","quality_loss_viz"]].values
        prev = df[df["iteration"]==it-1][["compression_tokens","quality_loss_viz"]].values
        for (x1,y1) in cur:
            for (x0,y0) in prev:
                edges.append(((x0,y0),(x1,y1)))

    best_idx = int(df["combined"].idxmin())
    best = df.loc[best_idx].to_dict()

    def nearest_prev(prev_df, x, y):
        if prev_df.empty: return None
        P = prev_df[["compression_tokens","quality_loss_viz"]].values
        d = np.sqrt(((P - np.array([x,y]))**2).sum(axis=1))
        i = int(np.argmin(d))
        return prev_df.iloc[i]

    path_pts = [(best["compression_tokens"], best["quality_loss_viz"])]
    cur_it = int(best["iteration"])
    cur_row = df.loc[best_idx]
    while cur_it > df["iteration"].min():
        prev_df = df[df["iteration"]==cur_it-1]
        prev_row = nearest_prev(prev_df, cur_row["compression_tokens"], cur_row["quality_loss_viz"])
        if prev_row is None: break
        path_pts.append((prev_row["compression_tokens"], prev_row["quality_loss_viz"]))
        cur_row = prev_row
        cur_it -= 1
    path_pts = path_pts[::-1]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    ax = fig.add_subplot(gs[0,0])
    info = fig.add_subplot(gs[0,1]); info.axis("off")

    for (x0,y0),(x1,y1) in edges:
        ax.plot([x0,x1],[y0,y1], linewidth=0.6, alpha=0.2, color="gray", zorder=1)

    if len(path_pts) > 1:
        xs, ys = zip(*path_pts)
        ax.plot(xs, ys, linewidth=2.2, color="red", zorder=3)

    cset = plt.get_cmap("tab10")
    comb = df["combined"].values
    s = 80 + 420*(comb.max() - comb)/(comb.max()-comb.min()+1e-9)
    for isl in sorted(df["island"].unique()):
        sub = df[df["island"]==isl]
        ax.scatter(sub["compression_tokens"], sub["quality_loss_viz"], s=s[sub.index],
                   color=cset(isl%10), label=f"Island {isl}", zorder=2, alpha=0.95,
                   edgecolor="white", linewidth=0.5)
        if args.label_islands and not sub.empty:
            cx, cy = sub["compression_tokens"].mean(), sub["quality_loss_viz"].mean()
            ax.text(cx, cy, f"Island {isl}", ha="center", va="center", fontsize=10, alpha=0.6)

    p = df.sort_values("compression_tokens")
    frontier = p[p["quality_loss_viz"] <= p["quality_loss_viz"].cummin()]
    ax.plot(frontier["compression_tokens"], frontier["quality_loss_viz"], marker="o",
            linestyle="-", color="black", alpha=0.6)

    ax.set_title("Evolution View", fontsize=14)
    ax.set_xlabel("Compression")
    ax.set_ylabel("Quality loss")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", frameon=False)

    def bar(ax, y, label, val, vmax=1.0):
        ax.text(0.0, y, f"{label}", fontsize=10, va="center")
        ax.add_patch(plt.Rectangle((0.35, y-0.04), 0.6, 0.08, color="#eee", transform=ax.transAxes, clip_on=False))
        w = max(0.0, min(val/vmax, 1.0))*0.6
        ax.add_patch(plt.Rectangle((0.35, y-0.04), w, 0.08, color="#6aa9ff", transform=ax.transAxes, clip_on=False))
        ax.text(0.97, y, f"{val:.3f}", fontsize=10, ha="right", va="center")

    info.text(0.0, 0.97, "Selected Candidate", fontsize=12, weight="bold")
    info.text(0.0, 0.93, f"Iteration: {int(best['iteration'])}", fontsize=10)
    info.text(0.0, 0.89, f"Combined score (w={args.w_comp}/{args.w_qloss}): {best['combined']:.3f}", fontsize=10)

    bar(info, 0.80, "Quality loss", float(best["quality_loss_viz"]), vmax=float(df["quality_loss_viz"].max()))
    bar(info, 0.72, "Compression", float(best["compression_tokens"]), vmax=float(df["compression_tokens"].max()))
    bar(info, 0.64, "Combined", float(best["combined"]), vmax=float(df["combined"].max()))

    wrap = lambda s: textwrap.fill(s, width=48, max_lines=7, placeholder="…")
    info.text(0.0, 0.54, "Prompt:", fontsize=10, weight="bold")
    info.text(0.0, 0.49, wrap(str(df.loc[best_idx, 'prompt'])), fontsize=9)
    info.text(0.0, 0.36, "Output:", fontsize=10, weight="bold")
    info.text(0.0, 0.31, wrap(str(df.loc[best_idx, 'output'])), fontsize=9)

    info.text(0.0, 0.18, "Metrics (best):", fontsize=10, weight="bold")
    if "sbert_cosine" in df.columns:
        info.text(0.0, 0.14, f"SBERT cosine = {float(df.loc[best_idx, 'sbert_cosine']):.3f}", fontsize=9)
    if "bert_f1" in df.columns:
        info.text(0.0, 0.10, f"BERTScore F1  = {float(df.loc[best_idx, 'bert_f1']):.3f}", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf(); fig.tight_layout()
    fig.savefig(out, dpi=180)
    print("Wrote:", out)

if __name__ == "__main__":
    main()