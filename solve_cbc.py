# AES MILP (minimum number of active S-boxes) using CVXPY
# Objective and constraints match Table 4: only the first N rounds (16*N x-variables) are counted.

import cvxpy as cp
import numpy as np


def build_aes_milp_problem(ROUNDS=4, verbose=False):
    """
    Build the 0-1 MILP for AES rounds as in the paper appendix.
    Only the first 16*ROUNDS x-variables (i.e., N rounds of SubBytes inputs) are counted in the objective
    and in the "at least one S-box active" constraint, so results match Table 4 exactly.
    Returns: (problem, x_vars, d_vars, a_states)
    """
    x_vars = []  # list of scalar boolean variables x_k
    d_vars = []  # list of scalar boolean dummy variables d_t (one per column per round)
    constraints = []

    # a is a 4x4 matrix of indices into x_vars for the current state (bytes)
    a = [[None] * 4 for _ in range(4)]

    # Initialize the first 16 x variables (these correspond to the first round inputs)
    for i in range(4):
        for j in range(4):
            x_vars.append(cp.Variable(boolean=True, name=f"x_{len(x_vars)}"))
            a[i][j] = len(x_vars) - 1

    def shift_rows(a_state):
        """Apply AES ShiftRows: row i is rotated left by i."""
        b = [[None] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                b[i][j] = a_state[i][(j + i) % 4]
        return b

    def mix_columns(a_state):
        """
        For each column j in {0..3}, create 4 new output x's (overwriting a_state[i][j]),
        and add the branch-number-5 constraints:
            sum(inputs + outputs) >= 5 * d
            d >= each input/output
        """
        nonlocal x_vars, d_vars, constraints
        for j in range(4):
            in_idx = [a_state[i][j] for i in range(4)]
            in_vars = [x_vars[idx] for idx in in_idx]

            out_vars = []
            for i in range(4):
                x_vars.append(cp.Variable(boolean=True, name=f"x_{len(x_vars)}"))
                new_idx = len(x_vars) - 1
                a_state[i][j] = new_idx
                out_vars.append(x_vars[new_idx])

            d = cp.Variable(boolean=True, name=f"d_{len(d_vars)}")
            d_vars.append(d)

            constraints.append(cp.sum(in_vars + out_vars) >= 5 * d)
            for v in in_vars:
                constraints.append(d - v >= 0)
            for v in out_vars:
                constraints.append(d - v >= 0)

    # Build ROUNDS rounds: ShiftRows then MixColumns
    a_states = []
    for r in range(ROUNDS):
        a = shift_rows(a)
        mix_columns(a)
        a_states.append([row[:] for row in a])

    # Count only the first N rounds' SubBytes inputs: these are exactly the first 16*ROUNDS x's
    in_round_x = x_vars[:16 * ROUNDS]

    # At least one S-box is active within the first N rounds
    constraints.append(cp.sum(in_round_x) >= 1)

    # Minimize the number of active S-boxes within the first N rounds
    objective = cp.Minimize(cp.sum(in_round_x))

    problem = cp.Problem(objective, constraints)
    return problem, x_vars, d_vars, a_states


def solve_aes_milp(ROUNDS=4, verbose=True):
    problem, x_vars, d_vars, a_states = build_aes_milp_problem(ROUNDS=ROUNDS, verbose=verbose)
    problem.solve(solver=cp.CBC, verbose=True)

    obj = problem.value

    def to01(v, tol=1e-6):
        """Convert cvxpy variable value to {0,1} with numerical tolerance."""
        if v is None:
            return 0
        arr = np.asarray(v)
        val = float(arr) if arr.shape == () else float(arr.flat[0])
        if val < 0 and abs(val) < tol:
            val = 0.0
        if val > 1 and abs(val - 1) < tol:
            val = 1.0
        return int(round(val))

    x_sol = [to01(var.value) for var in x_vars]
    d_sol = [to01(var.value) for var in d_vars]

    if verbose:
        print(f"ROUNDS={ROUNDS}, objective (min active S-boxes over first N rounds) = {obj}")
        print("x (first 32):", x_sol[:32])
        print("d (first 8): ", d_sol[:8])

    return obj, x_sol, d_sol, a_states


if __name__ == "__main__":
    N = 4 # rounds
    obj, x_sol, d_sol, a_states = solve_aes_milp(ROUNDS=N, verbose=True)
    print(f"N={N}, min(k_N) = {int(round(obj))}")