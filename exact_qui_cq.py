"""Numerical reference for U(N) = max_p -log Tr exp(sum_x p_x log rho_x).

All outputs must be full rank. Values are in nats. Float64 bounds with small
padding are numerical checks, not interval-arithmetic proofs.
"""

import numpy as np
from scipy.optimize import minimize


def exact_qui(rho_outputs, tolerance=1e-6, max_iterations=200, max_retries=3):
    """Return midpoint, lower, upper, gap, accepted, probabilities and sigma_b.

    rho_outputs has shape (number of inputs, dimension_B, dimension_B).
    Use midpoint as a reference only when accepted is True (gap <= tolerance).
    The default tolerance is an absolute interval width of 1e-6 nats.
    max_retries permits additional SLSQP runs from the best preceding candidate.
    Return the narrowest interval found, with its own probabilities and sigma_b.
    Optimizer success alone does not certify accuracy.
    """
    states = np.asarray(rho_outputs, dtype=complex)
    if states.ndim != 3 or not states.size or states.shape[1] != states.shape[2]:
        raise ValueError("rho_outputs must be a nonempty array of square matrices.")
    if not np.all(np.isfinite(states)):
        raise ValueError("Output states must be finite.")
    for state in states:
        if np.linalg.norm(state - state.conj().T) > 1e-10:
            raise ValueError("Output states must be Hermitian.")
        if abs(np.trace(state) - 1) > 1e-10:
            raise ValueError("Output states must have trace one.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    for name, value, minimum in (("max_iterations", max_iterations, 1),
                                 ("max_retries", max_retries, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    states = (states + states.conj().transpose(0, 2, 1)) / 2
    states /= np.trace(states, axis1=1, axis2=2).real[:, None, None]
    eigenvalues, eigenvectors = np.linalg.eigh(states)
    if np.any(eigenvalues <= 0):
        raise ValueError("Output states must be full rank.")
    logs = (eigenvectors * np.log(eigenvalues)[:, None, :]) @ eigenvectors.conj().transpose(0, 2, 1)
    count = len(states)
    padding = 1e-10
    best = None

    def objective(p):
        nonlocal best
        p = np.asarray(p, dtype=float)
        if (not np.all(np.isfinite(p)) or p.min() < -1e-10
                or abs(p.sum() - 1) > 1e-10):
            raise ValueError("Probabilities must be nonnegative and sum to one.")
        p = np.maximum(p, 0.0)  # Correct roundoff only; allow zero probabilities.
        p /= p.sum()
        averaged_log = np.einsum("x,xij->ij", p, logs)
        values, vectors = np.linalg.eigh(averaged_log)
        weights = np.exp(values - values[-1])
        value = -float(values[-1] + np.log(weights.sum()))
        sigma = (vectors * (weights / weights.sum())) @ vectors.conj().T
        gradient = -np.einsum("ij,xji->x", sigma, logs).real

        # Concavity: F(p) <= U(N) <= F(p) + max_x g_x - p.g (all labels).
        lower = max(0.0, value - padding)
        upper = value + max(0.0, float(gradient.max() - p @ gradient)) + padding
        if not np.isfinite(lower + upper) or lower > upper:
            raise ValueError("Invalid numerical reference interval.")
        gap = upper - lower
        result = dict(midpoint=(lower + upper) / 2, lower=lower, upper=upper,
                      gap=gap, accepted=gap <= tolerance,
                      probabilities=p.copy(), sigma_b=sigma)
        if best is None or gap < best["gap"]:
            best = result
        return -value, -gradient

    objective(np.full(count, 1.0 / count))
    for _ in range(max_retries + 1):
        if best["accepted"] or count == 1:
            break
        solution = minimize(
            objective, best["probabilities"], method="SLSQP", jac=True,
            bounds=[(0.0, 1.0)] * count,
            constraints={"type": "eq", "fun": lambda p: p.sum() - 1,
                         "jac": lambda p: np.ones(count)},
            options={"maxiter": max_iterations,
                     "ftol": max(np.finfo(float).eps, min(1e-12, tolerance ** 2 / 10))},
        )
        objective(solution.x)
    return best
