"""
Combine convergence plots for CBC, HiGHS, and MOSEK MILP solver logs.

This script reads solver logs from three different MILP solvers—CBC,
HiGHS, and MOSEK—and produces a single figure containing three
subplots.  Each subplot visualizes the relative gap (convergence rate)
versus cumulative runtime for a family of problem sizes (denoted by
``N`` values extracted from the filenames).  The subplots use the
same styling conventions as their standalone scripts: CBC and MOSEK
curves are drawn as continuous lines, while HiGHS curves are drawn as
step functions to reflect the piecewise-constant behaviour between
updates.  Legends are sorted in ascending order of N and plotted on
each subplot.

Usage::

    python combined_convergence_plot.py \
        --cbc cbc_*.txt \
        --highs highs_*.txt \
        --mosek mosek_*.txt \
        --output combined_convergence.png

If no file patterns are supplied, the script falls back to default
lists of filenames for each solver.
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Callable

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# CBC parsing utilities (based on cbc_convergence_time.py)
# -----------------------------------------------------------------------------
NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"

PAT_CBC_A = re.compile(
    rf"After\s+(?P<nodes>\d+)\s+nodes.*?"
    rf"(?P<incumbent>{NUM})\s+best\s+solution,\s*best\s+possible\s+"
    rf"(?P<bound>{NUM}).*?\((?P<time>{NUM})\s+seconds?\)",
    re.IGNORECASE,
)
PAT_CBC_B = re.compile(
    rf"After\s+(?P<nodes>\d+)\s+nodes.*?"
    rf"best\s+possible\s+(?P<bound>{NUM}).*?"
    rf"(?P<incumbent>{NUM})\s+best\s+solution.*?\((?P<time>{NUM})\s+seconds?\)",
    re.IGNORECASE,
)


def _to_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val.replace(",", ""))
    except Exception:
        return None


def parse_cbc_progress(path: str) -> List[Tuple[float, float]]:
    """Parse a CBC log and return a list of (time, relative gap) pairs."""
    pts: List[Tuple[float, float]] = []
    if not os.path.exists(path):
        return pts
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PAT_CBC_A.search(line) or PAT_CBC_B.search(line)
            if not m:
                continue
            time_val = _to_float(m.group("time"))
            inc_val = _to_float(m.group("incumbent"))
            bnd_val = _to_float(m.group("bound"))
            if time_val is None:
                continue
            gap_val: float | None = None
            if inc_val is not None and bnd_val is not None and abs(inc_val) > 1e-12:
                gap_val = abs(inc_val - bnd_val) / abs(inc_val)
            if gap_val is None:
                m_gap = re.search(r"gap(?:\s+of)?\s*(\d+(?:\.\d+)?)\s*%", line, re.IGNORECASE)
                if m_gap:
                    try:
                        gap_val = float(m_gap.group(1)) / 100.0
                    except Exception:
                        gap_val = None
            if gap_val is not None:
                pts.append((time_val, max(gap_val, 0.0)))
    # Deduplicate by time (keep last gap)
    pts.sort(key=lambda x: x[0])
    dedup: Dict[float, float] = {}
    for t, g in pts:
        dedup[t] = g
    return [(t, dedup[t]) for t in sorted(dedup.keys())]


# -----------------------------------------------------------------------------
# HiGHS parsing utilities (based on plot_convergence.py)
# -----------------------------------------------------------------------------

def parse_highs_progress(path: str) -> Tuple[List[float], List[float]]:
    """Extract (time, gap) pairs from a HiGHS solver log.

    Only lines from the branch-and-bound progress table are parsed.  These
    lines are identified by having a trailing time (ending in ``'s'``) and
    containing either a percentage (``%``) or the keyword ``Large`` in the
    gap column.  Lines that describe presolve reductions, symmetry
    detection, or CVXPY compilation are ignored.
    """
    times: List[float] = []
    gaps: List[float] = []
    if not os.path.exists(path):
        return times, gaps
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.rstrip().endswith("s"):
                continue
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
            gap_str = parts[-6]
            gap: float | None = None
            if gap_str.lower() == "large":
                gap = 1.0
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
    """Convert raw (time, gap) pairs into a stepwise convergence curve."""
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


# -----------------------------------------------------------------------------
# MOSEK parsing utilities (based on mosek_convergence_simple.py)
# -----------------------------------------------------------------------------

def parse_mosek_progress(path: str) -> List[Tuple[float, float]]:
    """Parse a MOSEK MIP log and return monotonic (time, gap) pairs."""
    times: List[float] = []
    gaps: List[float] = []
    header_found = False
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ':' not in line:
                continue
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
    curve: List[Tuple[float, float]] = []
    if times:
        paired = sorted(zip(times, gaps), key=lambda x: x[0])
        current = 1.0
        for t, g in paired:
            if g < current:
                current = g
            curve.append((t, current))
    return curve


# -----------------------------------------------------------------------------
# Utility for loading curves
# -----------------------------------------------------------------------------

def load_curves(file_patterns: List[str], parse_fn: Callable[[str], List[Tuple[float, float]]], *, step: bool = False) -> Dict[int, Tuple[List[float], List[float]]]:
    """Load solver logs into dictionaries keyed by problem size N.

    Parameters
    ----------
    file_patterns: List[str]
        Glob patterns or explicit filenames to search for log files.
    parse_fn: Callable[[str], List[Tuple[float, float]]]
        Function that parses a log file and returns a list of (time, gap) pairs.
    step: bool, optional
        Whether to convert the raw points into a step curve via ``make_step_curve``.

    Returns
    -------
    curves: Dict[int, Tuple[List[float], List[float]]]
        A mapping from N to sorted time and gap sequences.
    """
    curves: Dict[int, Tuple[List[float], List[float]]] = {}
    # Expand any glob patterns
    files: List[str] = []
    for patt in file_patterns:
        files.extend(glob.glob(patt))
    for path in files:
        stem = Path(path).stem
        m = re.search(r"_(\d+)$", stem)
        if not m:
            continue
        try:
            N = int(m.group(1))
        except ValueError:
            continue
        data = parse_fn(path)
        if not data:
            curves.setdefault(N, ([], []))
            continue
        if step:
            xs, ys = zip(*data)
            step_times, step_gaps = make_step_curve(list(xs), list(ys))
            curves[N] = (step_times, step_gaps)
        else:
            xs, ys = zip(*data)
            curves[N] = (list(xs), list(ys))
    return curves


# -----------------------------------------------------------------------------
# Plotting function
# -----------------------------------------------------------------------------

def plot_combined(cbc_files: List[str], highs_files: List[str], mosek_files: List[str], output: str) -> None:
    """Create a combined figure with three subplots for CBC, HiGHS, and MOSEK logs."""
    cbc_curves = load_curves(cbc_files, parse_cbc_progress, step=False)
    highs_curves = load_curves(highs_files, lambda fp: list(zip(*parse_highs_progress(fp))), step=True)
    mosek_curves = load_curves(mosek_files, parse_mosek_progress, step=False)

    if not cbc_curves and not highs_curves and not mosek_curves:
        raise RuntimeError("No valid data found in any input logs.")

    # Prepare three subplots horizontally
    # Arrange the subplots in a single row with three columns.  A wide figure
    # better accommodates the horizontal layout.  We do not share the y-axis
    # because the solvers may have different gap scales.
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
    solver_names = ["CBC", "HiGHS", "MOSEK"]
    all_curves = [cbc_curves, highs_curves, mosek_curves]
    # Plot each solver's curves on its corresponding axis
    for ax, solver, curves in zip(axes, solver_names, all_curves):
        # Plot each N in ascending order
        for N in sorted(curves.keys()):
            xs, ys = curves[N]
            if xs:
                if solver == "HiGHS":
                    ax.step(xs, ys, where="post", label=f"N={N}")
                else:
                    ax.plot(xs, ys, label=f"N={N}")
            else:
                # plot empty series to ensure legend entry appears
                if solver == "HiGHS":
                    ax.step([], [], where="post", label=f"N={N}")
                else:
                    ax.plot([], [], label=f"N={N}")
        # Set axis labels and title for this subplot
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative gap")
        ax.set_title(f"{solver} convergence over time")
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(title="Problem size")
    # Adjust layout to avoid overlapping elements
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    print(f"Saved combined convergence plot to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine CBC, HiGHS, and MOSEK convergence plots into one figure")
    parser.add_argument(
        "--cbc",
        nargs="*",
        default=["cbc_*.txt"],
        help="CBC log files or glob patterns (default: cbc_*.txt)",
    )
    parser.add_argument(
        "--highs",
        nargs="*",
        default=["highs_*.txt"],
        help="HiGHS log files or glob patterns (default: highs_*.txt)",
    )
    parser.add_argument(
        "--mosek",
        nargs="*",
        default=["mosek_*.txt"],
        help="MOSEK log files or glob patterns (default: mosek_*.txt)",
    )
    parser.add_argument(
        "--output",
        default="combined_convergence.png",
        help="Filename for the combined output plot (default: combined_convergence.png)",
    )
    args = parser.parse_args()
    plot_combined(args.cbc, args.highs, args.mosek, args.output)


if __name__ == "__main__":
    main()