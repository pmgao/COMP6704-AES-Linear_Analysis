"""
This script parses verbose output from the HiGHS MILP solver and
plots the convergence rate for multiple problem instances.  The input
files are expected to be the raw text output produced by running
HiGHS through CVXPY with the `--solver=HIGHS` flag.  Each file
corresponds to a MILP with a different number of rounds (for example
`highs_2.txt` contains the solver log for a model with 2 rounds).

For every solver log the script extracts the sequence of (time, gap)
pairs from the branch‑and‑bound progress table.  The "gap" column
reports the relative difference between the current best solution
(`BestSol`) and the best bound (`BestBound`) as a percentage.  A gap
of 0.0 means the solver has proven optimality.  HiGHS prints the
progress table with one line per event, ending with the CPU time in
seconds (for example ``0.2s``).  This script uses simple heuristics
to recognise those lines and parse the time and gap values.

The extracted sequences are then plotted using Matplotlib.  For each
problem size the relative gap is plotted against solver time.  The
script uses a step plot so that the value is constant between events
and only drops when the solver finds a better solution or bound.  A
horizontal line at zero would indicate immediate convergence.

Usage::

    python plot_convergence.py highs_*.txt

The script will automatically label each curve based on the numeric
suffix in the filename (e.g. ``highs_10.txt`` → ``ROUNDS=10``) and
save the resulting figure to ``convergence.png`` in the current
directory.

Example::

    $ python plot_convergence.py /home/oai/share/highs_*.txt
    Saved plot to convergence.png

Requirements: ``matplotlib`` must be installed.  If it is not
available, install it via pip (`pip install matplotlib`).
"""

import argparse
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_progress(file_path: str) -> Tuple[List[float], List[float]]:
    """Extract (time, gap) pairs from a HiGHS solver log.

    Only lines from the branch‑and‑bound progress table are parsed.
    These lines are identified by having a trailing time (ending in
    ``'s'``) and containing either a percentage (``%``) or the keyword
    ``Large`` in the gap column.  Lines that describe presolve
    reductions, symmetry detection, or CVXPY compilation are ignored.

    Parameters
    ----------
    file_path: str
        Path to the solver log file.

    Returns
    -------
    times: List[float]
        Cumulative CPU times in seconds when events occurred.
    gaps: List[float]
        Relative gaps (between 0 and 1) corresponding to each time.
    """

    times: List[float] = []
    gaps: List[float] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Only consider lines ending with a time stamp (e.g. "0.2s")
            if not line.rstrip().endswith("s"):
                continue
            # Ignore very short lines (they could be presolve statistics)
            parts = line.split()
            if len(parts) < 10:
                continue
            time_str = parts[-1]
            if not time_str.endswith("s"):
                continue
            try:
                time = float(time_str[:-1])  # strip trailing 's'
            except ValueError:
                continue

            # The gap value sits six columns from the end of the line.
            # According to HiGHS documentation, the column order is:
            # [Cuts, InLp, Confl., LpIters, Time] at the end.  We index
            # backwards to get the gap regardless of whether the "Src"
            # column is present.
            gap_str = parts[-6]
            gap: float | None = None
            # 'Large' denotes an infinite gap (no solution yet)
            if gap_str.lower() == "large":
                gap = 1.0
            # Gaps are reported as percentages (e.g. '96.77%')
            elif gap_str.endswith("%"):
                try:
                    gap_val = float(gap_str.rstrip("%"))
                    gap = gap_val / 100.0
                except ValueError:
                    pass

            if gap is None:
                continue

            times.append(time)
            gaps.append(gap)

    return times, gaps


def make_step_curve(times: List[float], gaps: List[float]) -> Tuple[List[float], List[float]]:
    """Convert raw (time, gap) pairs into a stepwise convergence curve.

    Given lists of times and gaps (not necessarily sorted), this
    function sorts the events by time and creates a step function that
    monotonically decreases.  At each event time the curve jumps down
    to the lowest gap seen so far.  The resulting lists have the same
    length as the number of events.

    Parameters
    ----------
    times: List[float]
        Event times in seconds.
    gaps: List[float]
        Relative gap values.

    Returns
    -------
    step_times: List[float]
        Sorted times.
    step_gaps: List[float]
        Non‑increasing gap values corresponding to ``step_times``.
    """
    # Sort by time ascending
    paired = sorted(zip(times, gaps), key=lambda x: x[0])
    step_times: List[float] = []
    step_gaps: List[float] = []
    current_gap = 1.0
    for t, g in paired:
        if g < current_gap:
            current_gap = g
        step_times.append(t)
        step_gaps.append(current_gap)
    return step_times, step_gaps


def plot_convergence(files: List[str], output: str = "convergence.png") -> None:
    """Plot convergence curves for a list of HiGHS solver logs.

    Each input file is parsed for time/gap pairs and plotted as a
    stepwise curve.  The legend is derived from the numeric suffix of
    the filename (for example ``highs_8.txt`` → ``ROUNDS=8``).  The
    resulting figure is saved to ``output``.

    Parameters
    ----------
    files: List[str]
        List of paths to HiGHS solver log files.
    output: str, optional
        Filename for the saved plot.  Defaults to 'convergence.png'.
    """

    if not files:
        raise ValueError("No input files provided for plotting.")

    plt.figure(figsize=(10, 7))
    # Collect curves keyed by integer N extracted from filename
    curves: dict[int, Tuple[List[float], List[float]]] = {}
    missing: List[str] = []
    for fp in files:
        # Derive a key from the filename (e.g., highs_10.txt → 10)
        basename = os.path.basename(fp)
        m = re.search(r"highs_(\d+)", basename)
        if not m:
            missing.append(fp)
            continue
        try:
            n_val = int(m.group(1))
        except ValueError:
            missing.append(fp)
            continue
        times, gaps = parse_progress(fp)
        # Only include curves with data; record empty lists otherwise
        curves[n_val] = (times, gaps)
    # Plot in ascending order of N to match CBC legend ordering
    for n_val in sorted(curves.keys()):
        times, gaps = curves[n_val]
        if times:
            step_times, step_gaps = make_step_curve(times, gaps)
            plt.step(step_times, step_gaps, where="post", label=f"N={n_val}")
        else:
            # Plot an empty line so that it still appears in the legend
            plt.step([], [], where="post", label=f"N={n_val}")

    plt.xlabel("Time (s)")
    # Label the vertical axis explicitly as the convergence rate (relative gap)
    plt.ylabel("Convergence rate (Relative gap)")
    plt.title("HiGHS MILP solver convergence over time")
    plt.ylim(bottom=0.0, top=1.0)
    plt.xlim(left=0.0)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(title="Problem size")
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot convergence of HiGHS MILP solver from verbose logs")
    parser.add_argument(
        "files",
        nargs="+",
        help="List of HiGHS solver log files to process (e.g. highs_*.txt)",
    )
    parser.add_argument(
        "--output",
        default="convergence.png",
        help="Filename for the saved plot (default: convergence.png)",
    )
    args = parser.parse_args()
    plot_convergence(args.files, args.output)


if __name__ == "__main__":
    main()