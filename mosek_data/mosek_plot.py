"""
Generate a single‑panel convergence rate vs. time plot for MOSEK solver logs.

This script is a simplified version of the MOSEK convergence plot generator:
it parses MOSEK MIP logs, extracts the running time and relative gap values
from the branch‑and‑bound progress table, and produces a single plot without
a zoomed inset.  The legend entries are ordered by increasing problem size
N, extracted from the filenames ``mosek_N.txt``, matching the style used
for the CBC and HiGHS plots.

Usage::

    python mosek_convergence_simple.py --files mosek_*.txt --output mosek_time_simple.pdf

If no files are supplied via the command line, a predefined list of
``mosek_2.txt``, ``mosek_4.txt``, ``mosek_6.txt`` etc. is used.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt


# Default log files; can be overridden via CLI
DEFAULT_FILES = [
    "./mosek_2.txt",
    "./mosek_4.txt",
    "./mosek_6.txt",
    "./mosek_8.txt",
    "./mosek_10.txt",
    "./mosek_12.txt",
    "./mosek_14.txt",
]

def parse_mosek_progress(path: str) -> List[Tuple[float, float]]:
    """Extract monotonic (time, gap) pairs from a MOSEK MIP log.

    Parameters
    ----------
    path: str
        Path to a MOSEK solver log file.

    Returns
    -------
    List[Tuple[float, float]]
        A sorted list of (time, relative gap) pairs where the gap is
        non‑increasing.  If the log does not contain progress data, an
        empty list is returned.
    """
    times: List[float] = []
    gaps: List[float] = []
    header_found = False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ':' not in line:
                    continue
                # Split at the last colon to ignore timestamps inside the prefix
                try:
                    _, content = line.rsplit(':', 1)
                except ValueError:
                    continue
                content = content.strip()
                if 'REL_GAP' in content:
                    header_found = True
                    continue
                if not header_found:
                    continue
                parts = content.split()
                if len(parts) < 8:
                    continue
                if not parts[0].isdigit():
                    continue
                time_str = parts[-1]
                gap_str = parts[-2]
                try:
                    t_val = float(time_str)
                except ValueError:
                    continue
                if gap_str.upper() == 'NA':
                    g_val = 1.0
                else:
                    try:
                        g_val = float(gap_str) / 100.0
                    except ValueError:
                        continue
                times.append(t_val)
                gaps.append(g_val)
    except FileNotFoundError:
        return []
    curve: List[Tuple[float, float]] = []
    if times:
        paired = sorted(zip(times, gaps), key=lambda x: x[0])
        current = 1.0
        for t, g in paired:
            if g < current:
                current = g
            curve.append((t, current))
    return curve


def plot_mosek_convergence(files: List[str], output: str) -> None:
    """Generate a single convergence‑rate vs. time plot for MOSEK logs.

    Parameters
    ----------
    files: List[str]
        List of log file paths to process.
    output: str
        Path where the resulting plot will be saved.
    """
    curves: Dict[int, List[Tuple[float, float]]] = {}
    missing: List[str] = []
    for p in files:
        if not os.path.exists(p):
            missing.append(f"{p} (file not found)")
            continue
        stem = Path(p).stem
        m = re.search(r"mosek_(\d+)", stem)
        if not m:
            missing.append(f"{p} (cannot infer N)")
            continue
        try:
            N = int(m.group(1))
        except ValueError:
            missing.append(f"{p} (invalid N)")
            continue
        data = parse_mosek_progress(p)
        curves[N] = data
    if missing:
        print("Notice:")
        for m_item in missing:
            print(" -", m_item)
    if not any(len(v) > 0 for v in curves.values()):
        raise RuntimeError("No valid progress data found in any MOSEK logs.")
    # Plot
    plt.figure(figsize=(10, 6))
    for N in sorted(curves.keys()):
        data = curves[N]
        if data:
            xs, ys = zip(*data)
            plt.step(xs, ys, where="post", label=f"N={N}")
        else:
            plt.step([], [], where="post", label=f"N={N}")
    plt.xlabel("Time (s)")
    plt.ylabel("Convergence rate (Relative gap)")
    plt.title("MOSEK convergence rate vs. time")
    plt.yscale('log')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Problem size")
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    print(f"Saved single-panel MOSEK convergence plot to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MOSEK convergence rate vs. time (single panel)")
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_FILES,
        help="List of MOSEK log files to process",
    )
    parser.add_argument(
        "--output",
        default="mosek_time_simple.pdf",
        help="Filename for the output plot",
    )
    args = parser.parse_args()
    plot_mosek_convergence(args.files, args.output)


if __name__ == "__main__":
    main()
