"""Numerical reference for max_p min_sigma sum_x p_x D_M(sigma || rho_x).

Use one small witness per CQ output and one common sigma_B. Values are in nats.
The final interval brackets the channel quantity, including optimization over p.
Float64 bounds with small padding are numerical checks, not interval proofs.
"""

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize, root

if __package__:
    from .mqui_bipartite import hermitian, log_derivative, pack, unpack
else:
    from mqui_bipartite import hermitian, log_derivative, pack, unpack


def _exp_gradient(h, state):
    """D exp_H(state) / Tr(state exp(H)), evaluated in the eigenbasis of H."""
    values, vectors = np.linalg.eigh(h)
    values = values - values[-1]
    rotated = vectors.conj().T @ state @ vectors
    normalizer = np.exp(values) @ np.diag(rotated).real
    delta = np.abs(values[:, None] - values[None, :])
    divided = np.exp(np.maximum(values[:, None], values[None, :]))
    unequal = delta > 0
    divided[unequal] *= -np.expm1(-delta[unequal]) / delta[unequal]
    derivative = vectors @ (divided * rotated) @ vectors.conj().T
    return derivative / normalizer


def measured_qui(rho_outputs, tolerance=1e-6, max_retries=3, max_iterations=40):
    """Return midpoint, lower, upper, gap, accepted, sigma_b, witness, probabilities.

    rho_outputs has shape (number of inputs, dimension_B, dimension_B).
    All outputs must be full-rank density matrices. witness[x] satisfies
    Tr(rho_outputs[x] @ witness[x]) = 1, including when probabilities[x] = 0.
    All returned arrays belong to the same candidate; use midpoint as a
    reference only when accepted is True (channel gap <= tolerance).
    The default tolerance is an absolute interval width of 1e-6 nats.

    max_retries permits additional calls per fixed-p solve AND additional
    SLSQP runs, each with at most max_iterations iterations. Both reuse the
    preceding candidates; outer retries also tighten the inner target, down
    to four times the padding. The total root-call budget is not max_retries+1.
    Return the narrowest valid interval even if tolerance is not met. Raise
    RuntimeError only if no valid interval was found. Optimizer success alone
    never certifies accuracy; tolerances below the padding may be unattainable.
    """
    states = np.asarray(rho_outputs, dtype=complex)
    if (states.ndim != 3 or not states.size
            or states.shape[1] != states.shape[2]):
        raise ValueError("rho_outputs must be a nonempty array of square matrices.")
    if not np.all(np.isfinite(states)):
        raise ValueError("Output states must be finite.")
    for state in states:
        if np.linalg.norm(state - state.conj().T) > 1e-10:
            raise ValueError("Output states must be Hermitian.")
        if abs(np.trace(state) - 1) > 1e-10 or np.linalg.eigvalsh(hermitian(state))[0] <= 0:
            raise ValueError("Output states must be full rank and have trace one.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    for name, value, minimum in (("max_retries", max_retries, 0),
                                 ("max_iterations", max_iterations, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    states = np.asarray([hermitian(s) / np.trace(s).real for s in states])
    count, d, _ = states.shape
    k = d * d - 1
    identity = np.eye(d)
    logs = np.asarray([hermitian(logm(s)) for s in states])
    inverse_sqrts = []
    for state in states:
        values, vectors = np.linalg.eigh(state)
        inverse_sqrts.append((vectors / np.sqrt(values)) @ vectors.conj().T)
    padding = 1e-10
    numerical_errors = (ValueError, np.linalg.LinAlgError, FloatingPointError)

    def probabilities(p):
        p = np.asarray(p, dtype=float)
        if (p.shape != (count,) or not np.all(np.isfinite(p))
                or p.min() < -1e-10 or abs(p.sum() - 1) > 1e-10):
            raise ValueError("Probabilities must be nonnegative and sum to one.")
        p = np.maximum(p, 0.0)  # Correct roundoff only; keep small positive weights.
        return p / p.sum()

    def initial(p):
        averaged_log = np.einsum("x,xij->ij", p, logs)
        values, vectors = np.linalg.eigh(averaged_log)
        weights = np.exp(values - values[-1])
        sigma = (vectors * (weights / weights.sum())) @ vectors.conj().T
        log_sigma = averaged_log - (values[-1] + np.log(weights.sum())) * identity
        return np.concatenate([*(pack(log_sigma - log_rho) for log_rho in logs), pack(sigma)])

    def matrices(z):
        if not np.all(np.isfinite(z)):
            raise ValueError("Nonfinite matrix coordinates.")
        hs = [unpack(row, d) for row in z[:count * k].reshape(count, k)]
        sigma = identity / d + unpack(z[count * k:], d)
        return hs, sigma

    # 1. Fixed-p equations: every label has a witness, all share one sigma_B.
    def equations(z, p):
        hs, sigma = matrices(z)
        residuals = [pack(sigma - _exp_gradient(h, state))
                     for h, state in zip(hs, states)]
        residuals.append(pack(np.einsum("x,xij->ij", p, hs)))
        return np.concatenate(residuals)

    # 2. L_p <= channel mQUI <= max_x b_x, with a tangent upper bound b_x.
    def bounds(z, p):
        hs, sigma = matrices(z)
        values, vectors = np.linalg.eigh(sigma)
        if not np.all(np.isfinite(values)) or values[0] < -1e-10:
            raise ValueError("Infeasible sigma_B.")
        sigma = (vectors * np.maximum(values, 0)) @ vectors.conj().T
        sigma /= np.trace(sigma).real
        witnesses, log_witnesses, divergences, uppers = [], [], [], []
        for h, state, inverse_sqrt in zip(hs, states, inverse_sqrts):
            w = hermitian(expm(h - np.linalg.eigvalsh(h)[-1] * identity))
            w /= np.trace(state @ w).real
            if not np.all(np.isfinite(w)) or np.linalg.eigvalsh(w)[0] <= 0:
                raise ValueError("Infeasible witness.")
            log_w = hermitian(logm(w))
            gradient = log_derivative(w, sigma)
            value = float(np.trace(sigma @ log_w).real)
            slack = np.linalg.eigvalsh(hermitian(inverse_sqrt @ gradient @ inverse_sqrt))[-1]
            slack -= np.trace(gradient @ w).real
            if not np.all(np.isfinite([value, slack])):
                raise ValueError("Nonfinite divergence bound.")
            witnesses.append(w)
            log_witnesses.append(log_w)
            divergences.append(value)
            uppers.append(value + max(0.0, slack) + padding)
        divergences, uppers = np.asarray(divergences), np.asarray(uppers)
        lower_eigenvalue = float(np.linalg.eigvalsh(
            hermitian(np.einsum("x,xij->ij", p, log_witnesses)))[0])
        if not np.isfinite(lower_eigenvalue):
            raise ValueError("Nonfinite lower bound.")
        lower = max(0.0, lower_eigenvalue - padding)
        upper = float(uppers.max())  # Includes all labels, even when p_x = 0.
        fixed_gap = float(p @ uppers - lower)
        if (not np.all(np.isfinite(divergences)) or not np.all(np.isfinite(uppers))
                or np.any(uppers < np.maximum(0.0, divergences - padding))
                or lower > upper or fixed_gap < 0 or not np.isfinite(lower + upper)):
            raise ValueError("Invalid numerical interval.")
        result = dict(midpoint=(lower + upper) / 2, lower=lower, upper=upper,
                      gap=upper - lower, accepted=upper - lower <= tolerance,
                      sigma_b=sigma, witness=np.asarray(witnesses), probabilities=p.copy())
        # Zero-weight labels still need accurate divergences to guide p updates.
        inner_error = max(fixed_gap, float(np.max(uppers - divergences)))
        return result, divergences, inner_error

    best = None
    current_p = np.full(count, 1.0 / count)
    starts = [initial(current_p), np.zeros((count + 1) * k)]
    inner_tolerance = max(tolerance / 10, 4 * padding)

    def fixed_input(p):
        nonlocal best
        best_inner = None
        best_candidate = None
        for attempt in range(max_retries + 1):
            index = attempt % 2
            candidate = starts[index]
            for refine in (False, True):
                try:
                    with np.errstate(over="raise", divide="raise", invalid="raise"):
                        if refine and candidate.size:
                            solution = root(equations, candidate, args=(p,), tol=1e-10,
                                            options={"maxfev": max(4000, 10 * (candidate.size + 1)),
                                                     "factor": 0.1})
                            candidate = solution.x
                        row = bounds(candidate, p)
                    result, _, inner_error = row
                    starts[index] = np.r_[candidate[:count * k], pack(result["sigma_b"])]
                    if best is None or result["gap"] < best["gap"]:
                        best = result
                    if best_inner is None or inner_error < best_inner[2]:
                        best_inner = row
                        best_candidate = starts[index].copy()
                    if best["accepted"] or best_inner[2] <= inner_tolerance:
                        starts[0] = best_candidate
                        return best_inner
                except numerical_errors:
                    continue
        if best_candidate is not None:
            starts[0] = best_candidate
        return best_inner

    class StopSearch(Exception):
        """Stop an outer run on a certified result or an inaccurate inner solve."""

    # 3. SLSQP proposes p; only the independent channel interval can accept it.
    for attempt in range(max_retries + 1):
        inner_tolerance = max(tolerance / (10 * 2 ** attempt), 4 * padding)
        last_p, last_value = None, None

        def objective(p):
            nonlocal current_p, last_p, last_value
            p = probabilities(p)
            if last_p is not None and np.array_equal(p, last_p):
                return last_value
            current_p = p.copy()
            row = fixed_input(p)
            if best is not None and best["accepted"]:
                raise StopSearch()
            if row is None or row[2] > inner_tolerance:
                raise StopSearch()  # Refine the inner solve before updating p.
            _, divergences, _ = row
            # At an exact inner solution these divergences form a supergradient
            # of F(p). Numerically they guide the search, not the acceptance test.
            last_p = p.copy()
            last_value = (-float(p @ divergences), -divergences)
            return last_value

        try:
            objective(current_p)
            if count > 1:
                solution = minimize(
                    objective, current_p, method="SLSQP", jac=True,
                    bounds=[(0.0, 1.0)] * count,
                    constraints={"type": "eq", "fun": lambda p: p.sum() - 1,
                                 "jac": lambda p: np.ones(count)},
                    options={"maxiter": max_iterations,
                             "ftol": max(np.finfo(float).eps, min(1e-12, tolerance ** 2 / 10))},
                )
                current_p = probabilities(solution.x)
                objective(current_p)
        except (StopSearch, *numerical_errors):
            pass
        if best is not None and best["accepted"]:
            return best
    if best is None:
        raise RuntimeError("No feasible reference interval was found.")
    return best
