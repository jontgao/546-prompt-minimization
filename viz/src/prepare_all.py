import argparse, os, glob, subprocess, sys
from pathlib import Path

def is_run_dir(p: Path) -> bool:
    if not (p / "meta.json").exists():
        return False
    if (p / "events.json").exists() or (p / "events.jsonl").exists():
        return True
    if list(p.glob("run-*.json")):
        return True
    return False

def find_runs(root: Path):
    for path in root.rglob("*"):
        if path.is_dir() and is_run_dir(path):
            yield path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--with_quality", action="store_true")
    args = ap.parse_args()

    root = Path(args.runs_root)
    run_dirs = sorted(find_runs(root))
    if not run_dirs:
        print(f"No runs found under {root}")
        return

    for rd in run_dirs:
        out_csv = rd / "prepared.csv"
        subprocess.check_call([sys.executable, "src/loader.py",
                               "--run_dir", str(rd),
                               "--out", str(out_csv)])
        if args.with_quality:
            subprocess.check_call([sys.executable, "src/quality_sbert.py",
                                   "--run_dir", str(rd),
                                   "--in_csv", str(out_csv),
                                   "--out_csv", str(rd / "prepared_quality.csv")])
    print("Done.")

if __name__ == "__main__":
    main()