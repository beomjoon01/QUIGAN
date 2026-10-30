"""Numerical bipartite mQUI reference for QUIGAN_bipartite.py.

Only NumPy/SciPy are needed; this file is independent of the other code.
Input: full-rank rho_AB in the A tensor B basis. All values are in nats.
The bounds use float64 with small padding, not interval arithmetic.
"""

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import root


def hermitian(matrix):
    return (matrix + matrix.conj().T) / 2


def pack(matrix):
    """Coordinates in the orthonormal traceless Hermitian basis, without storing it."""
    matrix = hermitian(matrix)
    d = len(matrix)
    i, j = np.triu_indices(d, 1)
    off_diagonal = np.empty(d * (d - 1))
    off_diagonal[0::2] = np.sqrt(2) * matrix[i, j].real
    off_diagonal[1::2] = -np.sqrt(2) * matrix[i, j].imag
    diagonal = np.diag(matrix).real
    k = np.arange(1, d)
    diagonal_coordinates = (np.cumsum(diagonal)[:-1] - k * diagonal[1:]) / np.sqrt(k * (k + 1))
    return np.concatenate([off_diagonal, diagonal_coordinates])


def unpack(coordinates, d):
    """Reconstruct a traceless Hermitian matrix from its real coordinates."""
    i, j = np.triu_indices(d, 1)
    off_count = d * (d - 1)
    matrix = np.zeros((d, d), dtype=complex)
    matrix[i, j] = (coordinates[:off_count:2] - 1j * coordinates[1:off_count:2]) / np.sqrt(2)
    matrix[j, i] = matrix[i, j].conj()
    k = np.arange(1, d)
    scaled = coordinates[off_count:] / np.sqrt(k * (k + 1))
    diagonal = np.r_[np.cumsum(scaled[::-1])[::-1], 0.0]
    diagonal[1:] -= k * scaled
    matrix[np.diag_indices(d)] = diagonal
    return matrix


def log_derivative(w, tau):
    """D log_W(tau), with stable treatment of equal/nearby eigenvalues."""
    values, vectors = np.linalg.eigh(w)
    high = np.maximum(values[:, None], values[None, :])
    low = np.minimum(values[:, None], values[None, :])
    delta = high - low
    equal = delta == 0
    near = (delta <= high / 2) & ~equal
    far = ~equal & ~near
    divided = np.empty_like(delta)
    divided[equal] = 1 / high[equal]
    divided[near] = -np.log1p(-delta[near] / high[near]) / delta[near]
    divided[far] = (np.log(high[far]) - np.log(low[far])) / delta[far]
    return hermitian(vectors @ (divided * (vectors.conj().T @ tau @ vectors))
                     @ vectors.conj().T)


def measured_qui(rho_ab, d_a, tolerance=1e-6, max_retries=3):
    """Return midpoint, lower, upper, gap, accepted, sigma_b and witness.

    d_a is the dimension of A (2**N_A); the dimension of B is inferred from rho_ab.
    accepted is True when gap = upper - lower <= tolerance.
    The default tolerance is an absolute interval width of 1e-6 nats.
    Use the midpoint as a reference only when accepted is True.
    max_retries counts additional solver calls after the first (0 disables retries).
    Each call allows max(4000, 10 * (number of variables + 1)) function evaluations.
    Alternate two initializations, resuming each from its last feasible candidate.
    Return the narrowest interval found, even if its gap exceeds tolerance.
    """
    rho = np.asarray(rho_ab, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or not rho.size:
        raise ValueError("rho_ab must be a nonempty square matrix.")
    if not np.all(np.isfinite(rho)) or np.linalg.norm(rho - rho.conj().T) > 1e-10:
        raise ValueError("rho_ab must be finite and Hermitian.")
    rho = hermitian(rho)
    if abs(np.trace(rho) - 1) > 1e-10 or np.linalg.eigvalsh(rho)[0] <= 0:
        raise ValueError("rho_ab must be full rank and have trace one.")
    if isinstance(d_a, bool) or not isinstance(d_a, (int, np.integer)) or d_a < 1 or len(rho) % d_a:
        raise ValueError("d_a must be a positive integer dividing the matrix dimension.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if isinstance(max_retries, bool) or not isinstance(max_retries, (int, np.integer)) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer.")
    rho = rho / np.trace(rho).real
    d_b = len(rho) // d_a
    rho_a = np.trace(rho.reshape(d_a, d_b, d_a, d_b), axis1=1, axis2=3)
    n_h = len(rho) ** 2 - 1
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    rho_inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.conj().T

    def partial_trace(matrix):
        blocks = matrix.reshape(d_a, d_b, d_a, d_b)
        return hermitian(np.einsum("ac,cbad->bd", rho_a, blocks))

    def matrices(x):
        h = unpack(x[:n_h], len(rho))
        sigma = np.eye(d_b) / d_b + unpack(x[n_h:], d_b)
        return h, sigma

    # 1. Umegaki initialization and an identity initialization for retry.
    x_b = partial_trace(logm(rho))
    sigma_initial = hermitian(expm(x_b - np.linalg.eigvalsh(x_b)[-1] * np.eye(d_b)))
    sigma_initial /= np.trace(sigma_initial).real
    h_initial = logm(np.kron(rho_a, sigma_initial)) - logm(rho)
    initial = np.concatenate([pack(h_initial), pack(sigma_initial)])

    # 2. Stationarity equations using the spectral derivative of exp(H).
    def equations(x):
        h, sigma = matrices(x)
        values, vectors = np.linalg.eigh(h)
        h = h - values[-1] * np.eye(len(h))
        values = values - values[-1]
        rotated_rho = vectors.conj().T @ rho @ vectors
        normalizer = np.exp(values) @ np.diag(rotated_rho).real
        delta = np.abs(values[:, None] - values[None, :])
        divided = np.exp(np.maximum(values[:, None], values[None, :]))
        unequal = delta > 0
        # Stable divided differences, including equal and nearly equal eigenvalues.
        divided[unequal] *= -np.expm1(-delta[unequal]) / delta[unequal]
        derivative = vectors @ (divided * rotated_rho) @ vectors.conj().T
        gradient = np.kron(rho_a, sigma) - derivative / normalizer
        return np.concatenate([pack(gradient), pack(partial_trace(h))])

    # 3. Independent bounds; solver success is not an accuracy certificate.
    def bounds(x):
        h, sigma = matrices(x)
        values, vectors = np.linalg.eigh(sigma)
        if values[0] < -1e-10:
            raise ValueError("Infeasible sigma_B.")
        sigma = (vectors * np.maximum(values, 0)) @ vectors.conj().T
        sigma /= np.trace(sigma).real
        w = hermitian(expm(h - np.linalg.eigvalsh(h)[-1] * np.eye(len(h))))
        w /= np.trace(rho @ w).real
        if np.linalg.eigvalsh(w)[0] <= 0:
            raise ValueError("Infeasible witness.")
        log_w = hermitian(logm(w))
        tau = np.kron(rho_a, sigma)
        gradient = log_derivative(w, tau)
        value = np.trace(tau @ log_w).real
        slack = np.linalg.eigvalsh(hermitian(rho_inverse_sqrt @ gradient @ rho_inverse_sqrt))[-1]
        slack -= np.trace(gradient @ w).real
        lower = max(0.0, float(np.linalg.eigvalsh(partial_trace(log_w))[0]) - 1e-10)
        upper = float(value + max(0.0, slack) + 1e-10)
        if not np.isfinite(lower + upper) or lower > upper:
            raise ValueError("Invalid numerical interval.")
        return dict(midpoint=(lower + upper) / 2, lower=lower, upper=upper,
                    gap=upper - lower, accepted=upper - lower <= tolerance,
                    sigma_b=sigma, witness=w)

    # Alternate the two starts and continue from their last feasible candidates.
    best = None
    starts = [initial, np.zeros_like(initial)]
    for attempt in range(max_retries + 1):
        candidate = starts[attempt % 2]
        # Check the starting point first: commuting examples may already be solved.
        for refine in (False, True):
            try:
                if refine and candidate.size:
                    solution = root(equations, candidate, tol=1e-10,
                                    options={"maxfev": max(4000, 10 * (candidate.size + 1)),
                                             "factor": 0.1})
                    candidate = solution.x
                result = bounds(candidate)
                starts[attempt % 2] = candidate
                if best is None or result["gap"] < best["gap"]:
                    best = result
                if best["accepted"]:
                    return best
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
    if best is None:
        raise RuntimeError("No feasible reference interval was found.")
    return best
