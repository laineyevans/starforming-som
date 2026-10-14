"""Shared prototype initialisation helpers."""

import numpy as np
from sklearn.decomposition import PCA


_ALIASES = {
    "kmeans": "kmeans++",
    "k-means++": "kmeans++",
    "kmeanspp": "kmeans++",
}
_METHODS = ("pca", "kmeans++", "random")


def resolve_init(config, which):
    """Return a per-method initialisation override or the shared default."""
    override = getattr(config, f"{which}_init", None)
    return override if override is not None else getattr(config, "init_method", "pca")


def _normalise_method(method):
    method = str(method).lower()
    method = _ALIASES.get(method, method)
    if method not in _METHODS:
        raise ValueError(
            f"Unknown initialisation method {method!r}; valid options are "
            "'pca', 'kmeans++', and 'random'."
        )
    return method


def _kmeans_plus_plus(X, k, rng):
    if k > len(X):
        raise ValueError(f"k={k} cannot exceed the number of samples {len(X)}")
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    indices = np.empty(k, dtype=int)
    indices[0] = rng.integers(len(X))
    centers[0] = X[indices[0]]
    min_distances = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = min_distances.sum()
        if total == 0:
            indices[i] = rng.integers(len(X))
        else:
            indices[i] = rng.choice(len(X), p=min_distances / total)
        centers[i] = X[indices[i]]
        min_distances = np.minimum(
            min_distances,
            np.sum((X - centers[i]) ** 2, axis=1),
        )
    return centers


def make_init(method, X, n_prototypes=None, grid_shape=None, seed=None,
              pca_span=2.0):
    """Create float64 prototype weights using a shared initialisation method."""
    method = _normalise_method(method)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if grid_shape is not None:
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must be a (rows, cols) pair")
        rows, cols = (int(grid_shape[0]), int(grid_shape[1]))
        n_prototypes = rows * cols
    elif n_prototypes is None:
        raise ValueError("n_prototypes is required when grid_shape is absent")

    if n_prototypes < 1:
        raise ValueError("n_prototypes must be positive")
    rng = np.random.default_rng(seed)

    if method == "pca":
        if grid_shape is None:
            raise ValueError("PCA initialisation requires grid_shape=(rows, cols)")
        # Use one canonical randomized basis so the sign-normalized result is
        # independent of which caller seed was selected.
        pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
        pca.fit(X)
        components = pca.components_.copy()
        for component in components:
            index = np.argmax(np.abs(component))
            if component[index] < 0:
                component *= -1
        scales = np.sqrt(pca.explained_variance_)
        if rows > cols:
            row_component, col_component = 0, 1
        elif cols > rows:
            row_component, col_component = 1, 0
        else:
            row_component, col_component = 0, 1
        centre = X.mean(axis=0)
        result = np.empty((n_prototypes, X.shape[1]), dtype=np.float64)
        row_denominator = (rows - 1) / 2
        col_denominator = (cols - 1) / 2
        for row in range(rows):
            u = (row - (rows - 1) / 2) / row_denominator if row_denominator else 0
            for col in range(cols):
                v = (col - (cols - 1) / 2) / col_denominator if col_denominator else 0
                result[row * cols + col] = centre + pca_span * (
                    u * scales[row_component] * components[row_component]
                    + v * scales[col_component] * components[col_component]
                )
        return result.copy()

    if method == "kmeans++":
        return _kmeans_plus_plus(X, int(n_prototypes), rng).copy()

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return rng.normal(mean, std, size=(int(n_prototypes), X.shape[1])).astype(np.float64)
