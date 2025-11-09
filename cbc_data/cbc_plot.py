"""
Plot CBC convergence rate (relative gap) versus time instead of nodes.

This script reads CBC MILP solver logs (as produced by CVXPY + CBC)
and extracts the progress information printed by CBC in lines of the
form::

    Cbc0010I After <nodes> nodes, <...> <inc> best solution, best possible <bound> (<time> seconds)

where ``<inc>`` is the incumbent (best solution value so far),
``<bound>`` is the best bound, and ``<time>`` is the cumulative CPU
time in seconds when the line was printed.  The relative gap is
computed as ``abs(incumbent - bound) / abs(incumbent)`` when both
values are numeric.  If the numeric fields are missing or zero, but
the log records an explicit percentage gap, that is used as a
fallback.

The script produces a two–panel figure: a top subplot showing the full
convergence curves for each problem instance, and a bottom subplot
zoomed into the initial phase of the computation.  The zoom window on
the x‑axis is determined either automatically based on a percentile
(``ZOOM_PERCENTILE``) of the combined time values across all instances
or manually via the ``ZOOM_X_MAX`` constant.  The y‑axis is on a
logarithmic scale in both panels, and limits are computed with some
margin around the data in the zoom window.

Legend entries are sorted in ascending order of the problem size (N)
extracted from the filenames (e.g. ``cbc_4.txt`` → ``N=4``).

Usage::

    python cbc_convergence_time.py --files cbc_*.txt --output figure.pdf

By default the script looks for files listed in the ``FILES`` constant
defined below.  Override this via the ``--files`` CLI argument.  The
resulting figure is saved to the path specified by ``--output``.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Config ----------------
# Default list of CBC log files to process.  These can be overridden
# via the CLI ``--files`` argument.
FILES = [
    "./cbc_2.txt",
    "./cbc_4.txt",
    "./cbc_6.txt",
    "./cbc_8.txt",
    "./cbc_10.txt",
    "./cbc_12.txt",
    "./cbc_14.txt",
]

# Set to a positive number to override automatic zoom window on x-axis (time)
# For example: ZOOM_X_MAX = 10.0  # seconds
ZOOM_X_MAX: float | None = None   # None = auto (based on data percentile)
ZOOM_PERCENTILE: float = 5.0      # If auto: zoom to the 5th percentile of all times

Y_FLOOR: float = 1e-6             # Lower bound for log-scale y-limits
Y_MARGIN: float = 0.2             # 20% margin when auto-scaling y in zoom

# ---------- CBC log parsing (robust) ----------
# Number pattern: supports commas and scientific notation
NUM = r'[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'

# Pattern A: "After X nodes ... <inc> best solution, best possible <bound> ... (<time> seconds)"
PAT_A = re.compile(
    rf"After\s+(?P<nodes>\d+)\s+nodes.*?"
    rf"(?P<incumbent>{NUM})\s+best\s+solution,\s*best\s+possible\s+"
    rf"(?P<bound>{NUM}).*?\((?P<time>{NUM})\s+seconds?\)",
    re.IGNORECASE,
)

# Pattern B: "After X nodes ... best possible <bound> ... <inc> best solution ... (<time> seconds)"
PAT_B = re.compile(
    rf"After\s+(?P<nodes>\d+)\s+nodes.*?"
    rf"best\s+possible\s+(?P<bound>{NUM}).*?"
    rf"(?P<incumbent>{NUM})\s+best\s+solution.*?\((?P<time>{NUM})\s+seconds?\)",
    re.IGNORECASE,
)

def _num(s: str) -> float | None:
    """Convert a number string with optional commas/scientific notation to float.

    Returns ``None`` if conversion fails.
    """
    if s is None:
        return None
    s_clean = s.replace(",", "")
    try:
        return float(s_clean)
    except Exception:
        return None

def parse_convergence(path: str) -> List[Tuple[float, float]]:
    """Extract (time, relative gap) pairs from a CBC log.

    Relative gap = |incumbent - bound| / |incumbent| (for incumbent ≠ 0).
    Falls back to percentage 'gap' on line if numeric fields aren't usable.

    Parameters
    ----------
    path: str
        Path to the CBC log file.

    Returns
    -------
    pts: List[Tuple[float, float]]
        Sorted list of (time, relative gap) pairs.  Duplicate times are
        deduplicated keeping the latest gap for each time.
    """
    pts: List[Tuple[float, float]] = []
    if not os.path.exists(path):
        return pts

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PAT_A.search(line) or PAT_B.search(line)
            if not m:
                continue

            # Extract values
            time_str = m.group("time")
            inc_str  = m.group("incumbent")
            bnd_str  = m.group("bound")

            time_val = _num(time_str) if time_str else None
            incumbent = _num(inc_str) if inc_str else None
            bound     = _num(bnd_str) if bnd_str else None

            if time_val is None:
                continue

            # Compute relative gap
            gap_val: float | None = None
            if incumbent is not None and bound is not None and abs(incumbent) > 1e-12:
                gap_val = abs(incumbent - bound) / abs(incumbent)
            # CBC sometimes reports explicit percentage gap in the message; try to parse it
            # Even if present, our primary method should work, so we attempt only when gap_val is None
            if gap_val is None:
                # Look for 'gap of xx%' or 'gap xx %' in the same line
                m_gap = re.search(r"gap(?:\s+of)?\s*(\d+(?:\.\d+)?)\s*%", line, re.IGNORECASE)
                if m_gap:
                    try:
                        gap_val = float(m_gap.group(1)) / 100.0
                    except Exception:
                        gap_val = None

            if gap_val is not None:
                pts.append((time_val, max(gap_val, 0.0)))

    # Sort by time and deduplicate keeping the latest per time point
    pts.sort(key=lambda x: x[0])
    dedup: dict[float, float] = {}
    for t, g in pts:
        dedup[t] = g
    return sorted(dedup.items(), key=lambda kv: kv[0])


def determine_zoom_range(all_times: List[float]) -> Tuple[float, float]:
    """Determine the x‑axis zoom range based on all time values.

    If ``ZOOM_X_MAX`` is provided and positive, use it as the upper limit.
    Otherwise, take the ``ZOOM_PERCENTILE`` percentile of the time values.

    Returns
    -------
    (x_min, x_max) : Tuple[float, float]
        The minimum and maximum time for the zoom window.
    """
    if not all_times:
        return (0.0, 1.0)
    x_min = min(all_times)
    if ZOOM_X_MAX is None:
        if len(all_times) >= 5:
            x_max = float(np.percentile(all_times, ZOOM_PERCENTILE))
        else:
            x_max = max(all_times) * 0.2
    else:
        x_max = float(ZOOM_X_MAX)
    return (x_min, x_max)


def determine_zoom_ylimits(curves: dict[int, List[Tuple[float, float]]], x_min: float, x_max: float) -> Tuple[float, float]:
    """Compute y‑limits for the zoom subplot based on data within [x_min, x_max]."""
    zoom_ys: List[float] = []
    for data in curves.values():
        if data:
            xs, ys = zip(*data)
            for t, g in zip(xs, ys):
                if x_min <= t <= x_max:
                    zoom_ys.append(max(g, Y_FLOOR))
    if not zoom_ys:
        # Expand window if nothing fell into initial window
        # Try doubling the window size, but cap at the max time
        all_times: List[float] = []
        all_gaps: List[float] = []
        for data in curves.values():
            if data:
                ts, gs = zip(*data)
                all_times.extend(ts)
                all_gaps.extend(gs)
        if all_times:
            new_x_max = min(max(all_times), x_max * 2.0)
            for data in curves.values():
                if data:
                    ts, gs = zip(*data)
                    for t, g in zip(ts, gs):
                        if x_min <= t <= new_x_max:
                            zoom_ys.append(max(g, Y_FLOOR))
            x_max = new_x_max
    # Determine y-limits
    if zoom_ys:
        y_min = max(min(zoom_ys) * (1.0 - Y_MARGIN), Y_FLOOR)
        y_max = max(zoom_ys) * (1.0 + Y_MARGIN)
    else:
        # Fallback to all values if still empty
        all_gaps: List[float] = []
        for data in curves.values():
            if data:
                _, gs = zip(*data)
                all_gaps.extend([max(g, Y_FLOOR) for g in gs])
        if all_gaps:
            y_min = max(min(all_gaps) * (1.0 - Y_MARGIN), Y_FLOOR)
            y_max = max(all_gaps) * (1.0 + Y_MARGIN)
        else:
            y_min, y_max = Y_FLOOR, 1.0
    return (y_min, y_max)


def plot_convergence_by_time(files: List[str], output: str) -> None:
    """Plot convergence curves (relative gap vs. time) for given CBC logs.

    Parameters
    ----------
    files: List[str]
        List of CBC log filenames to process.
    output: str
        Path where the output figure will be saved.
    """
    # Load all curves and collect all time values
    curves: dict[int, List[Tuple[float, float]]] = {}
    missing: List[str] = []
    all_times: List[float] = []
    for p in files:
        if not os.path.exists(p):
            missing.append(f"{p} (file not found)")
            continue
        stem = Path(p).stem  # e.g., 'cbc_12'
        try:
            N = int(stem.split("_")[-1])
        except Exception:
            missing.append(f"{p} (cannot infer N)")
            continue
        data = parse_convergence(p)
        if data:
            curves[N] = data
            all_times.extend([t for t, _ in data])
        else:
            # Keep an empty entry so that it still appears in the legend
            curves.setdefault(N, [])
            missing.append(f"{p} (no valid points parsed)")
    if missing:
        print("Notice: some inputs yielded no curve:")
        for item in missing:
            print(" -", item)
    # Ensure at least one curve has data
    if not any(len(v) > 0 for v in curves.values()):
        raise RuntimeError("No curves parsed. Please check log file paths and formats.")

    # Determine x-range for zoom
    x_min, x_max = determine_zoom_range(all_times)
    # Compute y-limits for zoom subplot
    y_min_zoom, y_max_zoom = determine_zoom_ylimits(curves, x_min, x_max)

    # Create figure with two subplots sharing neither axis
    fig, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(9, 8), sharey=False, constrained_layout=True
    )

    # Sort keys for consistent ordering (small N to large N)
    for N in sorted(curves.keys()):
        data = curves[N]
        if data:
            xs, ys = zip(*data)
            ax_full.plot(xs, ys, label=f"N={N}")
        else:
            # Plot an empty line so that it still appears in legend
            ax_full.plot([], [], label=f"N={N}")
    ax_full.set_xlabel("Time (s)")
    ax_full.set_ylabel("Convergence Rate (Relative Gap)")
    ax_full.set_title("CBC Convergence Rate vs. Time (Full View)")
    ax_full.set_yscale("log")
    ax_full.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_full.legend(title="Problem size", ncols=4, fontsize=9)

    # Zoom subplot
    for N in sorted(curves.keys()):
        data = curves[N]
        if data:
            xs, ys = zip(*data)
            ax_zoom.plot(xs, ys, label=f"N={N}")
        else:
            ax_zoom.plot([], [])
    ax_zoom.set_xlim(x_min, x_max)
    ax_zoom.set_ylim(y_min_zoom, y_max_zoom)
    ax_zoom.set_yscale("log")
    ax_zoom.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_zoom.set_xlabel("Time (s) — Zoomed")
    ax_zoom.set_ylabel("Convergence Rate (Relative Gap)")
    ax_zoom.set_title(f"Zoomed View (Time ≤ {x_max:.2f} s)")

    # Save the figure
    plt.savefig(output, dpi=600)
    print(f"Saved figure with zoom panel to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CBC convergence rate vs. time with zoom panels")
    parser.add_argument(
        "--files",
        nargs="*",
        default=FILES,
        help="List of CBC log files to process (default: predefined list)",
    )
    parser.add_argument(
        "--output",
        default="cbc_convergence_rate_by_time.pdf",
        help="Output filename for the saved plot (default: cbc_convergence_rate_by_time.pdf)",
    )
    args = parser.parse_args()
    plot_convergence_by_time(args.files, args.output)


if __name__ == "__main__":
    main()