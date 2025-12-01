#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

LATEX_SPECIALS = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\^{}',
    '\\': r'\textbackslash{}',
    '<': r'\textless{}',
    '>': r'\textgreater{}',
}


def latex_escape(s: str) -> str:
    if s is None:
        return ''
    return ''.join(LATEX_SPECIALS.get(ch, ch) for ch in s)


def truncate_text(s: str) -> str:
    return s.strip()


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def discover_runs(runs_dir: Path) -> List[Path]:
    patterns = list(runs_dir.glob("*/*/events.json"))
    return [p.parent for p in patterns]


def discover_runs_karim(runs_dir: Path) -> List[Path]:
    patterns = list(runs_dir.glob("*/run*.json"))
    return patterns


def load_run(run_folder: Path):
    meta_path = run_folder / "meta.json"
    events_path = run_folder / "events.json"
    if not meta_path.exists() or not events_path.exists():
        raise FileNotFoundError(f"Missing meta/events in {run_folder}")
    meta = json.load(meta_path.open("r", encoding="utf-8"))
    events = json.load(events_path.open("r", encoding="utf-8"))
    initial_prompt = meta.get("initial_prompt") or "<unknown prompt>"
    model_name = meta.get("tokenizer_model") or meta.get("tokenizer_model", run_folder.parent.name)
    return initial_prompt, model_name, events


def load_run_karim(run_folder: Path):
    meta_path = run_folder.parent / "meta.json"
    events_path = run_folder
    if not meta_path.exists() or not events_path.exists():
        raise FileNotFoundError(f"Missing meta/events in {run_folder}")
    meta = json.load(meta_path.open("r", encoding="utf-8"))
    events = json.load(events_path.open("r", encoding="utf-8"))
    meta.update(events.get('meta', {}))

    history = events['history']

    for event in history:
        event['prompt'] = event['new_prompt']
        event['scores'] = {
            'bert': 1 - event['bert_score'],
            'compression': event['compression_score']
        }

    initial_prompt = meta.get("original_prompt") or "<unknown prompt>"
    model_name = meta.get("model", run_folder.parent.name)
    return initial_prompt, model_name, history


def compute_milestones(events: List[Dict[str, Any]]):
    """
    From events (unsorted), compute milestones: the first event for each strictly-decreasing
    score (i.e., whenever best_so_far decreases). Return list sorted by iteration ascending.
    Each milestone is the event object.
    """
    # sort by iteration ascending, then by score ascending for deterministic tie-break
    evs = sorted(events, key=lambda e: (int(e.get("iteration", 0)), safe_float(e.get("score", math.inf))))
    best = math.inf
    milestones = []
    seen_iters = set()
    # We want the first time a strictly lower score is observed.
    for ev in evs:
        it = int(ev.get("iteration", 0))
        sc = safe_float(ev.get("score", math.inf))
        # If same iteration has multiple events, we consider the lowest score among them
        # but we still treat discovery at that iteration as a single milestone.
        if sc < best:
            best = sc
            if it in seen_iters:
                # already added milestone for this iteration (rare); still append to be safe
                milestones.append(ev)
            else:
                milestones.append(ev)
                seen_iters.add(it)
    return milestones


def make_float_for_prompt(prompt: str,
                          per_model_milestones: Dict[str, List[Dict[str, Any]]],
                          out_path: Path,
                          version: str
                          ):
    """
    Create a full-width float (figure*) for a single prompt that displays each model
    in a side-by-side minipage and lists only milestones.
    """
    models = sorted(per_model_milestones.keys())
    # Build LaTeX block
    lines = []
    lines.append(r"\begin{figure*}[t]")
    lines.append(r'\centering')
    lines.append(r"""\begin{tcolorbox}[
    colback=mygray, 
    colframe=gray!50!black, 
    title=\textbf{Input: Initial Prompt}, 
    fonttitle=\sffamily\small, 
    sharp corners=south,
    boxrule=0.5pt,
    left=4pt, right=4pt, top=4pt, bottom=4pt
]""")
    # Use a caption area showing the initial prompt (escaped & truncated)
    esc_prompt = latex_escape(prompt)
    lines.append(r"\small\sffamily\textit{ " + esc_prompt)
    lines.append("}")
    lines.append(r'\end{tcolorbox}')
    lines.append(r'\vspace{-2pt}')

    n = max(1, len(models))

    width_each = f"{1 / n - 0.01}\\textwidth"
    # For each model, create a minipage column
    column_blocks = []
    colors = ['myblue', 'myred']

    for i, m in enumerate(models):
        color = colors[i % len(colors)]

        block = []
        block.append(r"\begin{minipage}[t]{%s}" % width_each)
        block.append(r"\centering")
        block.append(fr'\textcolor{{{color}}}{{' + r"\textbf{\small " + latex_escape(m) + r"}}")
        block.append(r"\vspace{2pt}")
        block.append(r'\hrule height 0.8pt')
        block.append(r"\vspace{4pt}")

        milestones = per_model_milestones[m]
        if not milestones:
            block.append(r"\small (no events)")
        else:
            if milestones[0]['iteration'] == 0:
                milestones.pop(0)

            block.append(r"\begin{flushleft}")
            block.append(r"\footnotesize")
            # List each milestone: iteration, score, prompt, output
            for ev in milestones:
                it = int(ev.get("iteration", 0))
                sc = safe_float(ev.get("score", float('nan')))
                p_raw = ev.get("prompt", "")

                scores = ev.get("scores")

                # o_raw = ev.get("output", "")
                p = latex_escape(p_raw)
                # o = latex_escape(truncate_text(o_raw, max_output_chars))
                block.append(
                    r"\textbf{Iter %d} {\scriptsize \color{gray} (Score: %.4f $\mid$ Bert: %.4f $\mid$ Comp: %.4f)} \\ " % (
                        it, sc, 1 - scores['bert'],
                        scores['compression']))
                block.append(r'\vspace{1pt}')
                # Prompt in teletype small
                block.append(r"\texttt{" + p + r"}\\")
                block.append(r"\vspace{4pt}")
                # block.append(r"\textit{\footnotesize " + o + r"}\\[6pt]")
            block.pop(-1)
            block.append(r"\end{flushleft}")
        block.append(r"\end{minipage}")
        column_blocks.append("\n".join(block))
    # Join columns horizontally with small spacing
    # Use \hfill between minipages so they distribute across the width

    temp = 0
    if version == 'marius':
        sec = 'icl'
        temp = 0.9
    elif version == 'karim':
        sec = 'zsl'
        temp = 0
    else:
        sec = 'rl'
        temp = 0

    lines.append(("\n\\hfill\n").join(column_blocks))
    lines.append(r"\vspace{6pt}")
    lines.append(
        fr"\caption{{\textbf{{Milestone Discoveries in Prompt Minimization.}} The top panel shows the verbose initial prompt. The bottom panels compare the minimization trajectory of Qwen-32B (Left) and Llama-3.1-8B (Right) using Algorithm from Section \ref{{sec:{sec}}}. Metrics indicate total score, BERT-score, and compression ratio. Running with Temperature {temp}.}}")
    lines.append(r"\label{fig:milestones_" + str(abs(hash(prompt)) % (10 ** 8)) + r"}")
    lines.append(r"\end{figure*}")
    lines.append("\n")
    with out_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(runs_dir: Path, out_file: Path, version: str):
    if version == 'marius':
        run_folders = discover_runs(runs_dir)
    elif version == 'karim':
        run_folders = discover_runs_karim(runs_dir)
    elif version == 'li':
        run_folders = discover_runs(runs_dir)
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
            elif version == 'li':
                initial_prompt, model_name, events = load_run(rf)
        except Exception as e:
            print(f"Skipping {rf} due to error: {e}")
            continue
        agg[initial_prompt][model_name].extend(events)

    per_prompt_milestones = {}
    for prompt, model_map in agg.items():
        per_prompt_milestones[prompt] = {}
        for model, evs in model_map.items():
            milestones = compute_milestones(evs)
            per_prompt_milestones[prompt][model] = milestones

    # Write LaTeX document with twocolumn class (NeurIPS-like)
    preamble = r"""\documentclass[twocolumn]{article}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=0.7in}
\setlength{\parskip}{0.4ex}

\usepackage{xcolor}
\usepackage{tcolorbox}
\tcbuselibrary{skins, breakable}
\definecolor{myblue}{RGB}{0, 105, 180}
\definecolor{myred}{RGB}{200, 50, 50}
\definecolor{mygray}{RGB}{240, 240, 240}

\begin{document}
\title{Running-best milestone dumps}
\author{Auto-generated}
\maketitle
\thispagestyle{plain}
"""

    with out_file.open("w", encoding="utf-8") as f:
        f.write(preamble)

    for prompt, model_m in per_prompt_milestones.items():
        make_float_for_prompt(prompt, model_m, out_file, version)

    with out_file.open("a", encoding="utf-8") as f:
        f.write(r"\end{document}" + "\n")

    print(f"Wrote LaTeX file to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate running-best milestone LaTeX (two-column) from runs/")
    parser.add_argument("--runs-dir", type=str, default="../runs", help="Top-level runs directory")
    parser.add_argument("--out", type=str, default="running_best_milestones", help="Output .tex file")
    parser.add_argument("--versions", type=str, default=["marius", 'karim', 'li'], nargs='+',
                        choices=['marius', 'karim', 'li'])
    args = parser.parse_args()

    for version in args.versions:
        main(Path(args.runs_dir + "_" + version), Path(args.out + "_" + version + '.tex'), version)
