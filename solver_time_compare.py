"""
Plot solver solving times versus problem size for selected rounds.

This script visualizes how the solving time changes with the number of
rounds (denoted by ``N``) for three different MIP solvers: CBC,
HiGHS, and MOSEK.  It uses the data from a comparison table (see
documentation) and focuses on the instances with ``N`` equal to
2, 4, 6, 8, 10, 12, and 14.  The resulting plot helps compare
solvers' scalability across increasing problem sizes.

Usage::

    python solver_time_comparison.py --output solver_times.png

By default the script saves the plot as ``solver_times.png``.  Use
the ``--output`` option to specify a different filename.
"""

import argparse
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot solver times vs problem size")
    parser.add_argument(
        "--output",
        default="solver_times.pdf",
        help="Filename for the saved plot (default: solver_times.png)",
    )
    args = parser.parse_args()

    # Data extracted from the provided comparison table for N = 2,4,6,8,10,12,14
    n_vals = [2, 4, 6, 8, 10, 12, 14]
    cbc_times = [0.008, 0.139, 0.688, 4.327, 7.750, 430.515, 1104.072]
    highs_times = [0.010, 0.104, 0.148, 0.216, 0.259, 0.311, 0.496]
    mosek_times = [0.016, 0.073, 0.096, 0.431, 0.378, 0.475, 0.591]

    plt.figure(figsize=(8, 5))
    # Plot each solver's times
    plt.plot(n_vals, cbc_times, marker='o', label='CBC')
    plt.plot(n_vals, highs_times, marker='s', label='HiGHS')
    plt.plot(n_vals, mosek_times, marker='^', label='MOSEK')
    # Use log scale on y-axis to accommodate wide range of times
    plt.yscale('log')
    plt.xlabel('Rounds (N)')
    plt.ylabel('Solving Time (s)')
    plt.title('Solver solving time vs. problem size')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=600)
    print(f"Saved solver time comparison plot to: {args.output}")


if __name__ == '__main__':
    main()