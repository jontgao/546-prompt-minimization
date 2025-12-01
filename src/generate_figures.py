import argparse
import math
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from table_output import compute_milestones, load_run, load_run_karim, load_run_li, discover_runs, discover_runs_karim, discover_runs_li

# Define exact model keys as they appear in your data/logs
MODEL_KEY_QWEN = "Qwen/Qwen2.5-32B-Instruct-AWQ"
MODEL_KEY_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"

DPI = 600


def set_publication_style():
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
    })


NICE_NAME = {
    "marius": "In-Context Learning",
    "karim": "Zero-Shot Learning",
    "li": "LoRA RL"
}

COLOR_MAP = {
    NICE_NAME['marius']: "#1f77b4",
    NICE_NAME['karim']: "#d62728",
    NICE_NAME['li']: "#2ca02c"
}


def safe_float(val):
    try:
        return float(val)
    except:
        return math.inf


def densify_history(milestones: List[Dict], max_iter: int = 15) -> Tuple[np.array, np.array, np.ndarray]:
    """
    Returns dense arrays for (Objective Score, Compression Score).
    Forward-fills the best value found so far.
    """
    dense_obj = np.full(max_iter + 1, np.nan)
    dense_comp = np.full(max_iter + 1, np.nan)
    dense_bert = np.full(max_iter + 1, np.nan)

    dense_obj[0] = 0.5
    dense_comp[0] = 0.5

    # Initialize defaults (assuming start at 1.0 or similar, but using forward fill logic below)
    # If no event at 0, we leave as NaN until first event

    sorted_evs = sorted(milestones, key=lambda x: int(x.get('iteration', 0)))

    if not sorted_evs:
        return dense_obj, dense_comp, dense_bert

    # Do nothing
    current_best_obj = 0.5
    current_best_comp = 0.5
    current_best_bert = np.inf

    # We need to track the 'current state' to fill forward
    milestone_idx = 0

    for step in range(max_iter + 1):
        # Process all milestones that happened at or before this step
        while (milestone_idx < len(sorted_evs) and
               int(sorted_evs[milestone_idx].get('iteration', 0)) <= step):

            ev = sorted_evs[milestone_idx]

            # 1. Update Objective Score
            sc = safe_float(ev.get('score', math.inf))
            if sc < current_best_obj:
                current_best_obj = sc

            # 2. Update Compression Score
            raw_comp = ev.get('scores', {}).get('compression', math.inf)
            sc_comp = safe_float(raw_comp)

            # If strictly better compression found
            if sc_comp < current_best_comp:
                current_best_comp = sc_comp

            # 2. Update Bert Score
            raw_bert = ev.get('scores', {}).get('bert', math.inf)
            sc_bert = safe_float(raw_bert)

            # If strictly better compression found
            if sc_bert < current_best_bert:
                current_best_bert = sc_bert

            milestone_idx += 1

        # Fill the dense array
        if current_best_obj != math.inf:
            dense_obj[step] = current_best_obj

        if current_best_comp != math.inf:
            dense_comp[step] = current_best_comp

        if current_best_bert != math.inf:
            dense_bert[step] = current_best_bert

    dense_bert = dense_bert[1:]

    return dense_obj, dense_comp, dense_bert


def generate_comparative_scatter(milestones: dict, out_folder: Path):
    """
    Generates a Head-to-Head scatter plot for Qwen vs Llama for the SAME prompts.
    """
    # 1. Align Data by Prompt
    # Structure: Dict[Prompt] -> Dict[Model] -> Final Compression Value
    aligned_data = []

    for method_name, prompts_data in milestones.items():
        for prompt, models_data in prompts_data.items():

            # Check if both models exist for this prompt
            if MODEL_KEY_QWEN in models_data and MODEL_KEY_LLAMA in models_data:

                # Get the final iteration values
                qwen_evs = models_data[MODEL_KEY_QWEN]
                llama_evs = models_data[MODEL_KEY_LLAMA]

                if not qwen_evs or not llama_evs:
                    continue

                _, qwen_comp_hist, qwen_bert_hist = densify_history(qwen_evs)
                _, llama_comp_hist, llama_bert_hist = densify_history(llama_evs)

                # Take the value at the final iteration (index 15)
                # If NaN (run failed/empty), skip
                if not np.isnan(qwen_comp_hist[-1]) and not np.isnan(llama_comp_hist[-1]):
                    aligned_data.append({
                        "Method": method_name,
                        "Qwen_Compression": qwen_comp_hist[-1],
                        "Llama_Compression": llama_comp_hist[-1],
                        "Qwen_BERT": 1 - qwen_bert_hist[-1],
                        "Llama_BERT": 1 - llama_bert_hist[-1]
                    })

    if not aligned_data:
        print("No overlapping prompts found between Qwen and Llama for comparison.")
        return

    df = pd.DataFrame(aligned_data)

    for plot_type in ["Compression", "BERT"]:
        # 2. Plotting
        fig, ax = plt.subplots(figsize=(7, 7))

        # Scatter points
        sns.scatterplot(
            data=df,
            x=f"Llama_{plot_type}",
            y=f"Qwen_{plot_type}",
            hue="Method",
            style="Method",
            palette=COLOR_MAP,
            s=100,
            alpha=0.8,
            ax=ax
        )

        # Diagonal Line (x=y)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label="Equal Performance")

        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
        ax.set_xlim(left=max( 0, ax.get_xlim()[0]))

        ax.legend()

        if plot_type == "Compression":
            # Annotations
            ax.text(0.65, 0.4, "Qwen Better\n(More Compression)", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, color="gray", fontweight='bold')
            ax.text(0.4, 0.75, "Llama Better\n(More Compression)", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, color="gray", fontweight='bold')
        else:
            ax.text(0.65, 0.9, "Qwen Better\n(Closer Semantics)", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, color="gray", fontweight='bold')
            ax.text(0.55, 0.2, "Llama Better\n(Closer Semantics)", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, color="gray", fontweight='bold')

        ax.set_aspect('equal')
        ax.set_xlabel(f"Llama 3.1 (8B) {plot_type} Score")
        ax.set_ylabel(f"Qwen 2.5 (32B) {plot_type} Score")
        ax.set_title(f"Head-to-Head: Final {plot_type} Ratio", fontweight='bold')

        # Save
        plt.tight_layout()
        plt.savefig(out_folder / f"comparison_scatter_qwen_vs_llama_{plot_type.lower()}.pdf", bbox_inches='tight')
        plt.savefig(out_folder / f"comparison_scatter_qwen_vs_llama_{plot_type.lower()}.png", dpi=DPI,
                    bbox_inches='tight')
        print(f"Generated comparison scatter plot.")
        plt.close()


def generate_comparative_trajectory(data_by_model: dict, out_folder: Path):
    """
    Plots the average compression trajectory of Qwen vs Llama on the same graph.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models_to_plot = {
        "Qwen 32B": (MODEL_KEY_QWEN, "#1f77b4"),  # Blue
        "Llama 3.1": (MODEL_KEY_LLAMA, "#d62728")  # Red
    }

    for label, (model_key, color) in models_to_plot.items():
        if model_key not in data_by_model:
            continue

        # Aggregate all methods for this model to get a global model average
        # Or you can pick a specific method. Here we aggregate all methods.
        all_runs = []
        for method, runs in data_by_model[model_key].items():
            # runs is list of (obj, comp) tuples. We want comp (index 1)
            comps = [r[1] for r in runs]
            all_runs.extend(comps)

        if not all_runs:
            continue

        matrix = np.array(all_runs)
        mean_line = np.nanmean(matrix, axis=0)
        std_line = np.nanstd(matrix, axis=0)
        x_axis = np.arange(len(mean_line))

        # Plot Mean
        ax.plot(x_axis, mean_line, color=color, label=label, linewidth=3)
        # Plot Shadow (Confidence Interval)
        ax.fill_between(x_axis, mean_line - 0.2 * std_line, mean_line + 0.2 * std_line, color=color, alpha=0.1)

    ax.set_title("Average Compression Trajectory: Qwen vs Llama", fontweight='bold')
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Compression Score (Lower is Better)")
    ax.set_xlim(0, 15)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_folder / "comparison_trajectory_qwen_vs_llama.pdf", bbox_inches='tight')
    plt.savefig(out_folder / "comparison_trajectory_qwen_vs_llama.png", dpi=DPI, bbox_inches='tight')
    plt.close()


def generate_figures(milestones: dict, out_folder: Path):
    set_publication_style()

    # Data structure: Model -> Method -> List of (Obj, Comp) arrays
    data_by_model = defaultdict(lambda: defaultdict(list))

    # 1. Process Data
    for method_name, prompts_data in milestones.items():
        for prompt, models_data in prompts_data.items():
            for model_name, events in models_data.items():
                if not events:
                    continue
                dense_scores, dense_compression, _ = densify_history(events, max_iter=15)

                # Store if valid
                if not np.all(np.isnan(dense_scores)):
                    data_by_model[model_name][method_name].append((dense_scores, dense_compression))

    # 2. Generate Standard Individual Plots (as before)
    for model_name, methods_data in data_by_model.items():
        if not methods_data: continue

        # Plot Type 0: Objective, Type 1: Compression
        for i, metric_name in enumerate(["Objective", "Compression"]):
            fig, ax = plt.subplots(figsize=(8, 6))

            for method_name, runs_list in methods_data.items():
                # extract specific metric (0 or 1)
                matrix = np.array([r[i] for r in runs_list])

                color = COLOR_MAP.get(method_name, 'black')
                x_axis = np.arange(16)

                # Spaghetti lines
                for row in matrix:
                    ax.plot(x_axis, row, color=color, alpha=0.1, linewidth=1)

                # Mean line
                mean_line = np.nanmean(matrix, axis=0)
                ax.plot(x_axis, mean_line, color=color, label=f"{method_name}", linewidth=2.5)

            metric_label = "Objective Score" if i == 0 else "Compression Score"
            ax.set_title(f"{metric_label} Trajectory: {model_name.split('/')[-1]}", fontweight='bold')
            ax.set_ylabel(f"{metric_label} (Lower is Better)")
            ax.set_xlabel("Iteration")

            ax.set_xlim(0, 15)
            if i == 0:
                ax.set_yscale('log')  # Log scale often helps for Objective
                # ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.legend(loc='upper right')

            clean_name = model_name.replace("/", "_").replace(" ", "_")
            suffix = "" if i == 0 else "_compression"
            plt.savefig(out_folder / f"traj_{clean_name}{suffix}.pdf", bbox_inches='tight')
            plt.savefig(out_folder / f"traj_{clean_name}{suffix}.png", dpi=DPI, bbox_inches='tight')
            plt.close()

    # 3. Generate COMPARISON Plots (The new requirement)
    print("Generating Qwen vs Llama comparison plots...")
    generate_comparative_scatter(milestones, out_folder)
    generate_comparative_trajectory(data_by_model, out_folder)


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
    elif version == "li":
        run_folders = discover_runs_li(runs_dir)
    else:
        raise ValueError(f"Unknown version: {version}")

    print(f"Discovered {len(run_folders)} run folders under {runs_dir}")

    agg = defaultdict(lambda: defaultdict(list))
    for rf in run_folders:
        try:
            if version == 'marius':
                initial_prompt, model_name, events = load_run(rf)
            elif version == 'karim':
                initial_prompt, model_name, events = load_run_karim(rf)
            elif version == 'li':
                initial_prompt, model_name, events = load_run_li(rf)
        except Exception as e:
            # print(f"Skipping {rf} due to error: {e}")
            continue
        agg[initial_prompt][model_name].extend(events)

    per_prompt_milestones = {}
    for prompt, model_map in agg.items():
        per_prompt_milestones[prompt] = {}
        for model, evs in model_map.items():
            milestones = compute_milestones(evs)
            # Remove 0-th iteration if it's just initialization noise (optional)
            if milestones and milestones[0]['iteration'] == 0:
                milestones.pop(0)
            per_prompt_milestones[prompt][model] = milestones

    return per_prompt_milestones


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="../runs")
    parser.add_argument("--out_folder", type=str, default="../figs")
    parser.add_argument("--versions", type=str, default=["marius", 'karim', 'li'], nargs='+',
                        choices=['marius', 'karim', 'li'])
    args = parser.parse_args()
    main(args)
