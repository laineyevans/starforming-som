import numpy as np
import math
from init_utils import make_init, resolve_init


class SOM:
    """
    A 2-D rectangular Self-Organizing Map implemented in NumPy.
    """

    def __init__(self, m=3, n=3, dim=3, lr=1.0, sigma=1.0, max_iter=3000,
                 random_state=None, *, init_weights=None):
        self.m = m
        self.n = n
        self.dim = dim
        self.shape = (m, n)

        self.initial_lr = lr
        self.lr = lr
        self.sigma = sigma
        self.max_iter = max_iter

        self.random_state = random_state
        rng = np.random.default_rng(random_state)

        if init_weights is None:
            self.weights = rng.normal(size=(m * n, dim))
        else:
            init_weights = np.asarray(init_weights)
            if init_weights.shape != (m * n, dim):
                raise ValueError(
                    f"init_weights must have shape {(m * n, dim)}, "
                    f"got {init_weights.shape}"
                )
            self.weights = np.array(init_weights, dtype=np.float64, copy=True)
        self._locations = self._get_locations(m, n)

        self._inertia_ = None
        self._n_iter_ = None
        self._trained = False
        self.qe_history = []
        self.phase_start_iter = 0
        self.phase_iters = None
        self.lr_start = lr
        self.lr_end = lr
        self.sigma_start = sigma
        self.sigma_end = sigma


    # Core SOM operations

    def _get_locations(self, m, n):
        return np.argwhere(np.ones((m, n))).astype(np.int64)

    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1)
        return np.argmin(distances)

    def step(self, x):
        bmu_index = self._find_bmu(x)
        bmu_loc = self._locations[bmu_index]

        # Distance from each neuron to BMU location
        diff = self._locations.astype(float) - bmu_loc.astype(float)
        sq_dist = np.sum(diff * diff, axis=1)

        neighborhood = np.exp(-(sq_dist) / (2*self.sigma ** 2))
        local_step = self.lr * neighborhood

        delta = (x - self.weights) * local_step[:, np.newaxis]
        self.weights += delta

    def _compute_point_inertia(self, x):
        bmu_idx = self._find_bmu(x)
        return np.sum((x - self.weights[bmu_idx]) ** 2)

    def winner(self, x):
        idx = self._find_bmu(x)
        loc = self._locations[idx]
        return int(loc[0]), int(loc[1])

    def get_weights(self):
        return self.weights.reshape(self.m, self.n, self.dim)

    def _apply_schedule(self):
        phase_iters = max(1, self.phase_iters or 1)
        frac = np.clip(
            (self._n_iter_ - self.phase_start_iter) / float(phase_iters),
            0.0,
            1.0,
        )
        self.lr = self.lr_start + (self.lr_end - self.lr_start) * frac
        self.sigma = max(
            1e-6,
            self.sigma_start + (self.sigma_end - self.sigma_start) * frac,
        )

    # Quantization
    def quantization_error(self, X):
        if X is None or len(X) == 0:
            return 0.0
        preds = self.predict(X)
        bmus = self.weights[preds]
        dists = np.linalg.norm(X - bmus, axis=1)
        return float(np.mean(dists))


    # Training methods
    def train_random(self, X, num_iterations):
        n = X.shape[0]
        if n == 0:
            return
        if self._n_iter_ is None:
            self._n_iter_ = 0

        for _ in range(int(num_iterations)):
            idx = np.random.randint(n)
            self.step(X[idx])
            self._n_iter_ += 1
            self._apply_schedule()


    def train_batch(self, X, num_iterations):
        """Train sequentially online in dataset order; this is not a batch SOM update."""
        n = X.shape[0]
        if n == 0:
            return
        if self._n_iter_ is None:
            self._n_iter_ = 0

        for _ in range(int(num_iterations)):
            idx = self._n_iter_ % n
            self.step(X[idx])
            self._n_iter_ += 1
            self._apply_schedule()

    def train_in_blocks(self, X, total_iters, block_size, method="random"):
        qe_history = []
        n = len(X)
        if n == 0:
            return qe_history

        blocks = total_iters // block_size
        remainder = total_iters % block_size

        if self._n_iter_ is None:
            self._n_iter_ = 0

        for b in range(blocks):
            if method == "random":
                self.train_random(X, block_size)
            else:
                self.train_batch(X, block_size)

            self._trained = True
            qe = self.quantization_error(X)
            qe_history.append(qe)
            print(f"  Block {b+1}/{blocks}, QE={qe:.6f}")

        if remainder > 0:
            if method == "random":
                self.train_random(X, remainder)
            else:
                self.train_batch(X, remainder)

            self._trained = True
            qe = self.quantization_error(X)
            qe_history.append(qe)
            print(f"  Remainder block, QE={qe:.6f}")

        self.qe_history.extend(qe_history)
        return qe_history

   
    # Fit / Predict
    def fit(self, X, epochs=1, shuffle=True):
        n = len(X)
        if n == 0:
            return

        if self._n_iter_ is None:
            self._n_iter_ = 0
        if self.phase_iters is None:
            self.phase_start_iter = self._n_iter_
            self.phase_iters = epochs * n
            self.lr_start = self.lr
            self.lr_end = 0.0
            self.sigma_start = self.sigma
            self.sigma_end = 1e-6
        phase_end = self.phase_start_iter + self.phase_iters

        for epoch in range(epochs):
            if self._n_iter_ >= phase_end:
                break

            if shuffle:
                rng = np.random.default_rng(self.random_state)
                indices = rng.permutation(n)
            else:
                indices = np.arange(n)

            for idx in indices:
                if self._n_iter_ >= phase_end:
                    break

                self.step(X[idx])
                self._n_iter_ += 1
                self._apply_schedule()

        self._inertia_ = np.sum([self._compute_point_inertia(x) for x in X])
        self._trained = True

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("SOM must be trained before calling predict().")

        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Bad input shape {X.shape}, expected (*, {self.dim})")

        preds = np.array([self._find_bmu(x) for x in X])
        return preds

    def transform(self, X):
        X_stack = np.stack([X] * (self.m * self.n), axis=1)
        cluster_stack = np.stack([self.weights] * len(X), axis=0)
        return np.linalg.norm(X_stack - cluster_stack, axis=2)

    def u_matrix(self):
        """
        Compute U-matrix for a 2D SOM.
        Returns array of shape (m, n) where each entry is the mean distance
        between a neuron's weight vector and its immediate neighbours.
        """
        W = self.get_weights()       # shape (m, n, dim)
        U = np.zeros((self.m, self.n))

        for i in range(self.m):
            for j in range(self.n):
                neighbours = []

                if i > 0:
                    neighbours.append(W[i-1, j])
                if i < self.m - 1:
                    neighbours.append(W[i+1, j])
                if j > 0:
                    neighbours.append(W[i, j-1])
                if j < self.n - 1:
                    neighbours.append(W[i, j+1])

                if neighbours:
                    neigh = np.stack(neighbours)
                    U[i, j] = np.mean(np.linalg.norm(W[i, j] - neigh, axis=1))
                else:
                    U[i, j] = 0.0

        return U
    def hits_map(self, X):
        """
        Return a (m x n) matrix counting how many samples fall into each SOM neuron.
        """
        if not self._trained:
            raise RuntimeError("SOM must be trained before computing hits map.")

        hits = np.zeros((self.m, self.n), dtype=int)

        preds = self.predict(X)  # BMU index for each sample

        for p in preds:
            r, c = self._locations[p]   # convert index → 2D coordinate
            hits[r, c] += 1

        return hits




    # Properties

    @property
    def cluster_centers_(self):
        return self.weights.reshape(self.m, self.n, self.dim)

    @property
    def inertia_(self):
        if self._inertia_ is None:
            raise AttributeError("inertia_ not available until after fit()")
        return self._inertia_

    @property
    def n_iter_(self):
        if self._n_iter_ is None:
            raise AttributeError("n_iter_ not available until after fit()")
        return self._n_iter_


def train_som(encoded_data, labels, config):
    init_method = resolve_init(config, "som")
    init_seed = getattr(config, "init_seed", 42)
    init_weights = make_init(
        init_method,
        encoded_data,
        grid_shape=(config.n_num, config.m_num),
        seed=init_seed,
        pca_span=getattr(config, "init_pca_span", 2.0),
    )
    np.random.seed(init_seed)
    print(f"SOM initialisation: {init_method}")
    som = SOM(
        m=config.n_num,
        n=config.m_num,
        dim=encoded_data.shape[1],
        lr=config.rough_learning_rate,
        sigma=config.rough_sigma,
        max_iter=config.rough_epochs_num + config.fine_epochs_num,
        random_state=init_seed,
        init_weights=init_weights,
    )


    fine_lr_end = getattr(config, "fine_lr_end", 0.005)
    fine_sigma_end = getattr(config, "fine_sigma_end", 0.5)

    # Rough stage
    som.phase_start_iter = 0 if som._n_iter_ is None else som._n_iter_
    som.phase_iters = config.rough_epochs_num
    som.lr_start = config.rough_learning_rate
    som.lr_end = config.fine_learning_rate
    som.sigma_start = config.rough_sigma
    som.sigma_end = config.fine_sigma
    som.lr = som.lr_start
    som.sigma = som.sigma_start

    if config.train_in_blocks_flag:
        qe_rough = som.train_in_blocks(
            encoded_data,
            total_iters=config.rough_epochs_num,
            block_size=getattr(config, "som_block_size", 1000),
            method="random",
        )
    else:
        epochs = math.ceil(config.rough_epochs_num / len(encoded_data))
        som.fit(encoded_data, epochs=epochs)
        qe_rough = [som.quantization_error(encoded_data)]


    # Fine stage
    som.phase_start_iter = som._n_iter_
    som.phase_iters = config.fine_epochs_num
    som.lr_start = config.fine_learning_rate
    som.lr_end = fine_lr_end
    som.sigma_start = config.fine_sigma
    som.sigma_end = fine_sigma_end
    som.lr = som.lr_start
    som.sigma = som.sigma_start

    if config.train_in_blocks_flag:
        qe_fine = som.train_in_blocks(
            encoded_data,
            total_iters=config.fine_epochs_num,
            block_size=getattr(config, "som_block_size", 1000),
            method=getattr(config, "som_fine_method", "random"),
        )
    else:
        epochs = math.ceil(config.fine_epochs_num / len(encoded_data))
        som.fit(encoded_data, epochs=epochs)
        qe_fine = [som.quantization_error(encoded_data)]

    preds = som.predict(encoded_data)
    hits = som.hits_map(encoded_data)

    return som, preds, qe_rough, qe_fine, hits
