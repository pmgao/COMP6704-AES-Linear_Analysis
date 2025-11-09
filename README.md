# AES Linear Analysis via MILP

## Overview
This repository contains the code used to reproduce and visualize the
minimum-number-of-active-S-boxes study for AES via 0-1 MILP models. The
MILP model is implemented once in CVXPY and can be solved with different
back-end solvers (CBC, HiGHS, MOSEK). Auxiliary scripts parse solver log
files to study convergence behaviour and runtime scaling across rounds.

The original scripts accompany the COMP6704 coursework on AES linear
analysis. This README mirrors the structure of the
[`COMP6704-cfmm-routing`](https://github.com/pmgao/COMP6704-cfmm-routing)
project: it documents the repository layout, environment setup, how to
run the MILP experiments, and how to regenerate all published plots.

## Repository structure

```
.
├── solve_cbc.py             # Build and solve the AES MILP using the CBC solver
├── solve_highs.py           # Build and solve the AES MILP using the HiGHS solver
├── solve_mosek.py           # Build and solve the AES MILP using the MOSEK solver
├── combined_convergence_plot.py  # Combine convergence curves from all solvers
├── solver_time_compare.py        # Static comparison of solver run times per round count
├── cbc_data/
│   ├── cbc_*.txt            # Saved CBC solver logs for N ∈ {2,4,6,8,10,12,14}
│   └── cbc_plot.py          # Plot CBC convergence (full and zoomed views)
├── highs_data/
│   ├── highs_*.txt          # Saved HiGHS solver logs for the same N values
│   └── highs_plot.py        # Generate HiGHS convergence step plots
└── mosek_data/
    ├── mosek_*.txt          # Saved MOSEK solver logs for the same N values
    └── mosek_plot.py        # Generate MOSEK convergence plots (+ PDF export)
```

Each solver script exposes a shared `build_aes_milp_problem` function
that constructs the AES round transition constraints and objective.
Solver-specific entry points call `problem.solve` with the appropriate
solver flag and print summary statistics. Plotting scripts consume the
saved log files to visualize convergence and solver scaling.

## Prerequisites

The code targets Python ≥ 3.12. Install system-level solver binaries if
you plan to run the MILP instances locally:

* **CBC**: install via `apt install coinor-cbc` (Linux) or Homebrew on macOS.
* **HiGHS**: available through `pip install highspy` or packaged with CVXPY ≥ 1.4.
* **MOSEK**: requires a separate MOSEK license and Python bindings.

All Python dependencies are listed below and can be installed inside a
virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install cvxpy numpy matplotlib
# Optional: install extra solver interfaces exposed via CVXPY
pip install "cvxpy[CBC,CLARABEL,ECOS,GLPK,GUROBI,HIGHS,MOSEK,PDLP,SCS]"
```

> **Tip:** If MOSEK is not available, you can still run the CBC and
> HiGHS workflows. The plotting utilities will skip missing data files.

## Running the MILP experiments

The three solver entry points follow the same pattern. Edit the
`N = ...` line inside each script (default values: 4 for CBC, 9 for
HiGHS, 14 for MOSEK) to target a specific number of AES rounds, then run
with Python:

```bash
# CBC backend
python solve_cbc.py

# HiGHS backend
python solve_highs.py

# MOSEK backend
python solve_mosek.py
```

Each run prints the optimal objective (minimum number of active S-boxes)
and the first few binary decision variables. To archive the solver log
for later analysis, redirect stdout to a file inside the corresponding
`*_data/` directory, e.g.

```bash
python solve_cbc.py | tee cbc_data/cbc_8.txt
```

Repeat for the desired round counts `N`. The provided log files already
cover `N = 2, 4, 6, 8, 10, 12, 14` for all three solvers.

## Plotting convergence behaviour

Each solver folder ships with a dedicated plotting script that parses
its log format and produces a two-panel PDF/PNG figure (full trajectory
+ zoomed view). Examples:

```bash
# CBC convergence
cd cbc_data
python cbc_plot.py --files cbc_*.txt --output cbc_convergence.pdf

# HiGHS convergence (step curves)
cd ../highs_data
python highs_plot.py --files highs_*.txt --output highs_convergence.pdf

# MOSEK convergence
cd ../mosek_data
python mosek_plot.py --files mosek_*.txt --output mosek_convergence.pdf
```

Return to the project root to generate a combined comparison across all
solvers:

```bash
cd ..
python combined_convergence_plot.py \
    --cbc cbc_data/cbc_*.txt \
    --highs highs_data/highs_*.txt \
    --mosek mosek_data/mosek_*.txt \
    --output combined_convergence.png
```

The resulting figure aligns the three convergence plots on a single row
with logarithmic y-axes for easier comparison.

## Solver runtime comparison

The `solver_time_compare.py` script visualises the wall-clock time (in
seconds) for each solver at the same round counts. The data originates
from the course report table and is embedded directly inside the script.

```bash
python solver_time_compare.py --output solver_time.png
```

The y-axis is logarithmic because the CBC runtime spans several orders
of magnitude between `N=2` and `N=14`.

## Extending the experiments

1. **Custom round counts** – Import `build_aes_milp_problem` from any of
the solver scripts to create bespoke experiments in a notebook or new
Python file. The function returns the CVXPY problem object and variable
handles.
2. **Alternative solvers** – Replace the `problem.solve(...)` call with
any other MILP-capable solver supported by CVXPY (e.g., Gurobi, SCIP) as
long as the solver is installed.
3. **Different objectives** – Modify the objective in
`build_aes_milp_problem` if you wish to minimise or constrain different
subsets of rounds.

## Troubleshooting

* Ensure the selected solver is installed and licensed. CVXPY will raise
an informative error if a solver backend is unavailable.
* Large round counts (`N ≥ 12`) may take minutes or hours with CBC.
Consider HiGHS or MOSEK for faster convergence.
* If Matplotlib fails to render with `tk` on headless servers, set
`MPLBACKEND=Agg` before running the plotting scripts.
