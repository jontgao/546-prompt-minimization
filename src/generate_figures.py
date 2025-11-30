import argparse
import math
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from table_output import compute_milestones, load_run, load_run_karim, discover_runs, discover_runs_karim


def set_publication_style():
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.family": "serif",  # Matches LaTeX paper font
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


NICE_NAME = {
    "marius": "In-Context Learning",
    "karim": "Zero-Shot Learning",
    "li": "LoRA RL"
}

COLOR_MAP = {
    "In-Context Learning": "#1f77b4",  # Blue
    "Zero-Shot Learning": "#d62728",  # Red
    "EvolLoRA RL": "#2ca02c"  # Green
}


def safe_float(val):
    try:
        return float(val)
    except:
        return math.inf


def densify_history(milestones: List[Dict], max_iter: int = 15) -> np.array:
    """
    Converts a sparse list of milestone events into a dense array of length max_iter + 1.
    Performs forward-fill: the best score at step N persists until a better score is found.
    """
    dense = np.full(max_iter + 1, np.nan)
    dense[0] = 0.5

    sorted_evs = sorted(milestones, key=lambda x: int(x.get('iteration', 0)))

    if not sorted_evs:
        return dense

    current_best = 0.5

    milestone_idx = 0

    for step in range(max_iter + 1):
        while (milestone_idx < len(sorted_evs) and
               int(sorted_evs[milestone_idx].get('iteration', 0)) <= step):

            # Update current best if this milestone is valid
            sc = safe_float(sorted_evs[milestone_idx].get('score', math.inf))
            if sc < current_best:
                current_best = sc
            milestone_idx += 1

        if current_best != math.inf:
            dense[step] = current_best

    return dense


def generate_figures(milestones: dict, out_folder: Path):
    set_publication_style()

    data_by_model = defaultdict(lambda: defaultdict(list))

    for method_name, prompts_data in milestones.items():
        for prompt, models_data in prompts_data.items():
            for model_name, events in models_data.items():
                if not events:
                    continue
                dense_scores = densify_history(events, max_iter=15)
                if not np.all(np.isnan(dense_scores)):
                    data_by_model[model_name][method_name].append(dense_scores)

    for model_name, methods_data in data_by_model.items():
        if not methods_data:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for method_name, runs_list in methods_data.items():
            matrix = np.array(runs_list)

            color = COLOR_MAP.get(method_name, 'black')

            x_axis = np.arange(0, 16)
            for row in matrix:
                ax.plot(x_axis, row, color=color, alpha=0.15, linewidth=1)

            mean_line = np.nanmean(matrix, axis=0)

            ax.plot(x_axis, mean_line, color=color, label=f"{method_name} (Mean)", linewidth=3, linestyle='-')

        # Formatting
        ax.set_title(f"Optimization Trajectory: {model_name}", fontweight='bold', pad=15)
        ax.set_xlabel("Iteration Step")
        ax.set_ylabel("Objective Score (Lower is Better)")
        ax.set_xlim(0, 15)
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=16,integer=True))  # Integer ticks

        # Legend
        ax.legend(frameon=True, loc='upper right', fancybox=False, edgecolor='black')

        sanitized_name = model_name.replace("/", "_").replace(" ", "_")
        out_path = out_folder / f"trajectory_{sanitized_name}.png"
        out_pdf = out_folder / f"trajectory_{sanitized_name}.pdf"

        plt.tight_layout()
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        print(f"Generated figure for {model_name} at {out_path}")
        plt.close()


def main(args: Namespace):
    out_folder = Path(args.out_folder)

    out_folder.mkdir(parents=True, exist_ok=True)

    milestones = load_all(args, args.versions)

    generate_figures(milestones, out_folder)


def load_all(args: Namespace, versions: list[str]):
    milestones = {}

    for version in versions:
        milestones[NICE_NAME[version]] = load_single(Path(args.runs_dir + "_" + version), version)

    return milestones


def load_single(runs_dir: Path, version: str):
    if version == 'marius':
        run_folders = discover_runs(runs_dir)
    elif version == 'karim':
        run_folders = discover_runs_karim(runs_dir)
    else:
        raise ValueError(f"Unknown version: {version}")

    print(f"Discovered {len(run_folders)} run folders under {runs_dir}")

    agg = defaultdict(lambda: defaultdict(list))  # prompt -> model -> list[events]
    for rf in run_folders:
        try:
            if version == 'marius':
                initial_prompt, model_name, events = load_run(rf)
            elif version == 'karim':
                initial_prompt, model_name, events = load_run_karim(rf)
        except Exception as e:
            print(f"Skipping {rf} due to error: {e}")
            continue
        agg[initial_prompt][model_name].extend(events)

    per_prompt_milestones = {}
    for prompt, model_map in agg.items():
        per_prompt_milestones[prompt] = {}
        for model, evs in model_map.items():
            milestones = compute_milestones(evs)
            if milestones[0]['iteration'] == 0:
                milestones.pop(0)

            per_prompt_milestones[prompt][model] = milestones

    return per_prompt_milestones


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate running-best milestone LaTeX (two-column) from runs/")
    parser.add_argument("--runs-dir", type=str, default="../runs", help="Top-level runs directory")
    parser.add_argument("--out_folder", type=str, default="../figs", help="Output of the PNG files")
    parser.add_argument("--versions", type=str, default=["marius", 'karim'], nargs='+',
                        choices=['marius', 'karim', 'li'])
    args = parser.parse_args()
    main(args)
