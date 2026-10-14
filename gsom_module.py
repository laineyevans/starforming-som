"""
Growing Self-Organizing Map (GSOM) module.

Based on the algorithm described by Alahakoon, Halgamuge & Srinivasan (2000),
adapted and modernised for use with NumPy arrays and the ALMAGAL pipeline.

Original reference implementation by Philipp Ludwig (MIT licence).
"""

import numpy as np
from collections import defaultdict
import json
import os
from init_utils import make_init, resolve_init


class GSONode:
    """Single node in the growing SOM grid."""

    __slots__ = (
        "weights", "error", "x", "y",
        "right", "left", "up", "down",
        "hit_count",
    )

    def __init__(self, dim, x, y, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.weights = rng.random(dim).astype(np.float64)
        self.error = 0.0
        self.x = x
        self.y = y
        self.right = None
        self.left = None
        self.up = None
        self.down = None
        self.hit_count = 0

    def adjust_weights(self, target, lr):
        self.weights += lr * (target - self.weights)

    def is_boundary(self):
        return (self.right is None or self.left is None
                or self.up is None or self.down is None)


class GSOM:
    """Growing Self-Organizing Map."""

    def __init__(self, dim, spread_factor=0.5, init_lr=0.3,
                 max_iterations=200000, seed=None,
                 max_nodes=None, growth_phase_ratio=0.5,
                 init_vectors=None,
                 min_hits_to_grow=2,
                 min_active_neighbours_to_grow=2,
                 init_grid_size=2,
                 record_growth_snapshots=False,
                 *,
                 lr_end=None,
                 nb_lr_fraction=0.3):
        self.dim = dim
        self.spread_factor = spread_factor
        self.init_lr = init_lr
        self.max_iterations = max_iterations
        self.seed = seed
        self.max_nodes = max_nodes
        self.growth_phase_ratio = growth_phase_ratio
        if lr_end is not None and not 0 < lr_end <= init_lr:
            raise ValueError("lr_end must satisfy 0 < lr_end <= init_lr")
        if not 0 <= nb_lr_fraction <= 1:
            raise ValueError("nb_lr_fraction must be between 0 and 1")
        self.lr_end = lr_end
        self.nb_lr_fraction = nb_lr_fraction
        self.min_hits_to_grow = int(min_hits_to_grow)
        self.min_active_neighbours_to_grow = int(min_active_neighbours_to_grow)
        self._rng = np.random.default_rng(seed)
        self._record_snapshots = record_growth_snapshots
        self.growth_snapshots = []
        self.growth_events = 0
        self.blocked_hits = 0
        self.blocked_neighbours = 0
        self.blocked_no_boundary = 0
        self.blocked_cap = 0
        self.blocked_error = 0
        self.growth_log = []
        self.nodes_at_growth_end = None

        # Growing threshold
        self._GT = -dim * np.log2(spread_factor)

        # Parse init_grid_size: accepts an int (NxN) or a (rows, cols) tuple
        if isinstance(init_grid_size, (list, tuple)) and len(init_grid_size) == 2:
            grid_rows, grid_cols = int(init_grid_size[0]), int(init_grid_size[1])
        else:
            grid_rows = grid_cols = max(2, int(init_grid_size))
        self.init_grid_size = (grid_rows, grid_cols)

        # Create seed grid of shape grid_rows × grid_cols
        self.nodes = []
        grid = {}
        for gy in range(grid_rows):
            for gx in range(grid_cols):
                node = GSONode(dim, gx, gy, self._rng)
                self.nodes.append(node)
                grid[(gx, gy)] = node

        # Wire neighbours
        for (gx, gy), node in grid.items():
            if (gx - 1, gy) in grid:
                node.left = grid[(gx - 1, gy)]
            if (gx + 1, gy) in grid:
                node.right = grid[(gx + 1, gy)]
            if (gx, gy - 1) in grid:
                node.down = grid[(gx, gy - 1)]
            if (gx, gy + 1) in grid:
                node.up = grid[(gx, gy + 1)]

        # If initial vectors supplied (e.g. from data), use them to set the
        # seed node weights. Expect one vector per seed node (row-major order).
        if init_vectors is not None:
            iv = np.asarray(init_vectors)
            expected_shape = (len(self.nodes), self.dim)
            if iv.shape != expected_shape:
                raise ValueError(
                    f"init_vectors must have shape {expected_shape}, got {iv.shape}"
                )
            for i, node in enumerate(self.nodes):
                node.weights = np.array(iv[i], dtype=np.float64, copy=True)

        self._coord_map = {(n.x, n.y): n for n in self.nodes}
        self._trained = False

        if self._record_snapshots:
            self._record_snapshot()

    # ------------------------------------------------------------------
    # Snapshot recording
    # ------------------------------------------------------------------

    def _record_snapshot(self):
        """Save a lightweight snapshot of the current topology and weights."""
        positions = [(n.x, n.y) for n in self.nodes]
        weights = np.array([n.weights for n in self.nodes], dtype=np.float32)
        pos_to_idx = {pos: i for i, pos in enumerate(positions)}
        edges = []
        seen = set()
        for n in self.nodes:
            for nb in (n.right, n.up):  # only two directions avoids duplicates
                if nb is not None:
                    i = pos_to_idx[(n.x, n.y)]
                    j = pos_to_idx[(nb.x, nb.y)]
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        edges.append(key)
        self.growth_snapshots.append({
            "n_nodes": len(self.nodes),
            "positions": positions,
            "weights": weights,
            "edges": edges,
        })

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(v1, v2):
        return float(np.sum((v1 - v2) ** 2))

    def _find_bmu(self, vec):
        best_dist = float("inf")
        best_node = None
        for node in self.nodes:
            d = self._distance(vec, node.weights)
            if d < best_dist:
                best_dist = d
                best_node = node
        return best_node

    def _find_similar_boundary(self, node):
        best_dist = float("inf")
        best_node = None
        for other in self.nodes:
            if not other.is_boundary() or other is node:
                continue
            d = self._distance(node.weights, other.weights)
            if d < best_dist:
                best_dist = d
                best_node = other
        return best_node

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------

    def _insert(self, x, y, init_node):
        new = GSONode(self.dim, x, y, self._rng)
        self.nodes.append(new)
        self._coord_map[(x, y)] = new

        # Wire neighbours
        for dx, dy, attr_self, attr_other in [
            (-1, 0, "left", "right"),
            (+1, 0, "right", "left"),
            (0, +1, "up", "down"),
            (0, -1, "down", "up"),
        ]:
            nb = self._coord_map.get((x + dx, y + dy))
            if nb is not None:
                setattr(new, attr_self, nb)
                setattr(nb, attr_other, new)

        # Weight initialisation: 2 * parent − neighbour
        neigh = new.left or new.right or new.up or new.down
        if neigh is not None:
            new.weights = 2.0 * init_node.weights - neigh.weights

        if self._record_snapshots:
            self._record_snapshot()
        return new

    def _grow(self, node, max_new=None):
        grown = []
        def can_add():
            return (max_new is None) or (len(grown) < max_new)

        #if node.left is None:
        #   if can_add():
        #        grown.append(self._insert(node.x - 1, node.y, node))
        #if node.right is None:
        #    if can_add():
        #        grown.append(self._insert(node.x + 1, node.y, node))
        #if node.up is None:
        #    if can_add():
        #        grown.append(self._insert(node.x, node.y + 1, node))
        #if node.down is None:
        #    if can_add():
        #        grown.append(self._insert(node.x, node.y - 1, node))

        directions = [
                (-1, 0),  # left
                (1, 0),   # right
                (0, 1),   # up
                (0, -1)   # down
        ]

        self._rng.shuffle(directions)

        for dx, dy in directions:
            if not can_add():
                break
            if (node.x + dx, node.y + dy) not in self._coord_map:
                grown.append(self._insert(node.x + dx, node.y + dy, node))
        
        return grown

    @staticmethod
    def _active_neighbour_count(node):
        cnt = 0
        for nb in (node.left, node.right, node.up, node.down):
            if nb is not None and nb.hit_count > 0:
                cnt += 1
        return cnt

    def _maybe_grow(self, node, iteration):
        if node.error <= self._GT:
            self.blocked_error += 1
            return

        # Anti-tail guardrails: require repeat hits and support from more than
        # one active neighbour before allowing expansion.
        if node.hit_count < self.min_hits_to_grow:
            self.blocked_hits += 1
            return
        if self._active_neighbour_count(node) < self.min_active_neighbours_to_grow:
            self.blocked_neighbours += 1
            return

        if not node.is_boundary():
            node = self._find_similar_boundary(node)
            if node is None:
                self.blocked_no_boundary += 1
                return

        if self.max_nodes is not None:
            remaining = int(self.max_nodes - len(self.nodes))
            if remaining <= 0:
                self.blocked_cap += 1
                return
        else:
            remaining = None

        grown = self._grow(node, max_new=remaining)
        if grown:
            self.growth_events += 1
            self.growth_log.append((iteration, len(self.nodes)))
            node.error = 0.5 * self._GT
            # Require fresh support before this boundary can grow again.
            node.hit_count = 0

    # ------------------------------------------------------------------
    # Post-training pruning
    # ------------------------------------------------------------------

    @staticmethod
    def _neighbour_count(node):
        cnt = 0
        for nb in (node.left, node.right, node.up, node.down):
            if nb is not None:
                cnt += 1
        return cnt

    def _detach_node(self, node):
        """Detach one node from the topology and internal containers."""
        if node.left is not None:
            node.left.right = None
        if node.right is not None:
            node.right.left = None
        if node.up is not None:
            node.up.down = None
        if node.down is not None:
            node.down.up = None

        self._coord_map.pop((node.x, node.y), None)
        try:
            self.nodes.remove(node)
        except ValueError:
            pass

    def prune_dead_nodes(self, X, labels=None, min_hits=1, max_passes=100):
        """Iteratively prune empty tail nodes after training.

        Removes nodes with hit_count < min_hits that are leaf-like (<=1 neighbour).
        This mainly removes long dangling chains of empty nodes.
        Returns number of removed nodes.
        """
        X = np.asarray(X, dtype=np.float64)
        hits = self.hits_map(X, labels=labels)

        # Stamp observed hits onto nodes for pruning decisions.
        for node in self.nodes:
            node.hit_count = int(hits.get((node.x, node.y), 0))

        removed_total = 0
        passes = 0
        changed = True

        while changed and passes < int(max_passes):
            changed = False
            passes += 1

            to_remove = []
            for node in list(self.nodes):
                if node.hit_count >= int(min_hits):
                    continue
                if self._neighbour_count(node) <= 1:
                    to_remove.append(node)

            if not to_remove:
                break

            for node in to_remove:
                self._detach_node(node)
                removed_total += 1
                changed = True

        if self._record_snapshots:
            self._record_snapshot()

        return removed_total

    def _lr_at(self, progress):
        if self.lr_end is None:
            return self.init_lr * np.exp(-3.0 * progress)
        k = np.log(self.init_lr / self.lr_end)
        return self.init_lr * np.exp(-k * progress)

    def _adapt(self, bmu, inp, lr):
        bmu.adjust_weights(inp, lr)
        for nb in [bmu.left, bmu.right, bmu.up, bmu.down]:
            if nb is not None:
                nb.adjust_weights(inp, lr * self.nb_lr_fraction)

    def growth_summary(self):
        return {
            "seed_nodes": self.init_grid_size[0] * self.init_grid_size[1],
            "final_nodes": len(self.nodes),
            "max_nodes": self.max_nodes,
            "nodes_at_growth_end": self.nodes_at_growth_end,
            "growth_events": self.growth_events,
            "blocked_hits": self.blocked_hits,
            "blocked_neighbours": self.blocked_neighbours,
            "blocked_no_boundary": self.blocked_no_boundary,
            "blocked_cap": self.blocked_cap,
            "blocked_error": self.blocked_error,
            "growth_cutoff": int(self._growth_cutoff),
            "iterations": int(self._iterations),
            "GT": self._GT,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, iterations=None, verbose=True):
        """Train the GSOM on data array X of shape (n_samples, dim)."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.dim:
            raise ValueError(f"Data dim {X.shape[1]} != GSOM dim {self.dim}")

        n = X.shape[0]
        iters = iterations if iterations is not None else self.max_iterations
        growth_cutoff = int(iters * self.growth_phase_ratio)
        self._growth_cutoff = growth_cutoff
        self._iterations = iters
        if growth_cutoff == 0:
            self.nodes_at_growth_end = len(self.nodes)

        # Periodic snapshot cadence: aim for ~100 evenly-spaced frames.
        # Node-insertion snapshots are recorded separately inside _insert.
        _snap_interval = max(1, iters // 100) if self._record_snapshots else None
        _last_snap_nodes = len(self.nodes)  # track topology changes

        for it in range(iters):
            if it >= growth_cutoff and self.nodes_at_growth_end is None:
                self.nodes_at_growth_end = len(self.nodes)
            idx = self._rng.integers(n)
            inp = X[idx]

            # Exponential learning rate decay
            progress = it / max(iters - 1, 1)
            lr = self._lr_at(progress)

            bmu = self._find_bmu(inp)
            bmu.hit_count += 1

            self._adapt(bmu, inp, lr)

            # Error accumulation & growth (only during growth phase)
            err = self._distance(bmu.weights, inp)
            bmu.error += err
            if it < growth_cutoff:
                self._maybe_grow(bmu, it)

            # Periodic snapshot (captures weight evolution even without growth)
            if _snap_interval is not None and (it + 1) % _snap_interval == 0:
                # If node count changed since last periodic snap, _insert already
                # recorded that event; update our reference count.
                _last_snap_nodes = len(self.nodes)
                self._record_snapshot()

            if verbose and (it + 1) % 10000 == 0:
                print(f"  GSOM iter {it+1}/{iters}, nodes={len(self.nodes)},"
                      f" lr={lr:.5f},"
                      f" phase={'growing' if it < growth_cutoff else 'smoothing'}")

        if self.nodes_at_growth_end is None:
            self.nodes_at_growth_end = len(self.nodes)
        self._trained = True
        if verbose:
            print(f"GSOM training complete. Final node count: {len(self.nodes)}")

    # ------------------------------------------------------------------
    # Predict / transform
    # ------------------------------------------------------------------

    def predict(self, X):
        """Return an array of (x, y) tuples — one per sample."""
        X = np.asarray(X, dtype=np.float64)
        coords = []
        for row in X:
            bmu = self._find_bmu(row)
            coords.append((bmu.x, bmu.y))
        return coords

    def transform(self, X):
        """
        Transform the data X into cluster distance space.
        """
        X = np.asarray(X, dtype=np.float64)
        nodes = sorted(self.nodes, key=lambda n: (n.x, n.y))
        weights = np.array([node.weights for node in nodes])
        transformed = np.linalg.norm(
            X[:, np.newaxis, :] - weights[np.newaxis, :, :],
            axis=2
        )
        return transformed

    def predict_indices(self, X):
        """Return integer node indices (stable ordering of self.nodes)."""
        node_to_idx = {id(n): i for i, n in enumerate(self.nodes)}
        X = np.asarray(X, dtype=np.float64)
        indices = np.empty(len(X), dtype=int)
        for i, row in enumerate(X):
            bmu = self._find_bmu(row)
            indices[i] = node_to_idx[id(bmu)]
        return indices

    def hits_map(self, X, labels=None):
        """Return dict  (x, y) → hit_count.

        If labels are supplied, augmented samples are excluded.
        """
        augmentation_tags = ["[hflip]", "[vflip]", "[hvflip]", "[rot"]
        X = np.asarray(X, dtype=np.float64)
        hits = defaultdict(int)
        for i, row in enumerate(X):
            if labels is not None:
                lbl = labels[i] if i < len(labels) else ""
                if lbl and any(tag in lbl for tag in augmentation_tags):
                    continue
            bmu = self._find_bmu(row)
            hits[(bmu.x, bmu.y)] += 1
        return dict(hits)

    def quantization_error(self, X):
        X = np.asarray(X, dtype=np.float64)
        total = 0.0
        for row in X:
            bmu = self._find_bmu(row)
            total += np.sqrt(self._distance(bmu.weights, row))
        return total / len(X)

    # ------------------------------------------------------------------
    # Grid extents (for plotting)
    # ------------------------------------------------------------------

    def grid_bounds(self):
        """Return (x_min, x_max, y_min, y_max) of all node positions."""
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return min(xs), max(xs), min(ys), max(ys)


# ------------------------------------------------------------------
# Pipeline entry point
# ------------------------------------------------------------------

def train_gsom(encoded_data, labels, config):
    """Train a GSOM and return (gsom, preds_coords, qe)."""
    spread = getattr(config, "gsom_spread_factor", 0.5)
    lr = getattr(config, "gsom_learning_rate", 0.3)
    lr_end = getattr(config, "gsom_lr_end", None)
    nb_lr_fraction = getattr(config, "gsom_nb_lr_fraction", 0.3)
    iters = getattr(config, "gsom_iterations", 200000)
    max_nodes = getattr(config, "gsom_max_nodes", None)
    growth_ratio = getattr(config, "gsom_growth_phase_ratio", 0.5)
    min_hits_to_grow = getattr(config, "gsom_min_hits_to_grow", 2)
    min_active_neighbours_to_grow = getattr(config, "gsom_min_active_neighbours_to_grow", 2)
    init_grid_size = getattr(config, "gsom_init_grid_size", 2)
    init_method = resolve_init(config, "gsom")
    init_seed = getattr(config, "init_seed", 42)

    rough_epochs = getattr(config, "rough_epochs_num", None)
    fine_epochs = getattr(config, "fine_epochs_num", None)
    if (rough_epochs is not None and fine_epochs is not None
            and iters != rough_epochs + fine_epochs):
        print("Info: GSOM and SOM training budgets differ.")
    print(f"GSOM seed: {init_seed}")

    # Determine the seed grid shape for shared initialisation.
    if isinstance(init_grid_size, (list, tuple)) and len(init_grid_size) == 2:
        grid_shape = (int(init_grid_size[0]), int(init_grid_size[1]))
    else:
        side = max(2, int(init_grid_size))
        grid_shape = (side, side)
    init_vecs = make_init(
        init_method,
        encoded_data,
        grid_shape=grid_shape,
        seed=init_seed,
        pca_span=getattr(config, "init_pca_span", 2.0),
    )
    print(f"GSOM initialisation: {init_method}")

    record_snaps = getattr(config, "plot_gsom_growth_animation", False)

    gsom = GSOM(
        dim=encoded_data.shape[1],
        spread_factor=spread,
        init_lr=lr,
        max_iterations=iters,
        max_nodes=max_nodes,
        growth_phase_ratio=growth_ratio,
        init_vectors=init_vecs,
        min_hits_to_grow=min_hits_to_grow,
        min_active_neighbours_to_grow=min_active_neighbours_to_grow,
        init_grid_size=init_grid_size,
        record_growth_snapshots=record_snaps,
        seed=init_seed,
        lr_end=lr_end,
        nb_lr_fraction=nb_lr_fraction,
    )

    gsom.fit(encoded_data, iterations=iters)

    if getattr(config, "gsom_prune_after_training", False):
        removed = gsom.prune_dead_nodes(
            encoded_data,
            labels=labels,
            min_hits=getattr(config, "gsom_prune_min_hits", 1),
            max_passes=getattr(config, "gsom_prune_max_passes", 100),
        )
        if removed > 0:
            print(f"GSOM pruning removed {removed} empty tail node(s). Final nodes={len(gsom.nodes)}")

    summary = gsom.growth_summary()
    summary_text = "\n".join(
        ["GSOM growth summary:"]
        + [f"  {key}: {value}" for key, value in summary.items()]
    )
    print(summary_text)
    if max_nodes is not None and len(gsom.nodes) < max_nodes:
        blocked = {
            key: value for key, value in summary.items()
            if key.startswith("blocked_") and key != "blocked_error"
        }
        most_common_guard = max(blocked, key=blocked.get)
        print(
            f"WARNING: GSOM stopped at {len(gsom.nodes)} nodes, below max_nodes={max_nodes}; "
            "comparison with a 36-prototype SOM/SOUMAP is not at equal M. "
            f"Most frequent growth guard: {most_common_guard}={blocked[most_common_guard]}; "
            f"blocked_error={summary['blocked_error']}."
        )
    if getattr(config, "gsom_require_full_nodes", False) and (
            max_nodes is not None and len(gsom.nodes) < max_nodes):
        raise RuntimeError(summary_text)

    run_dir = getattr(config, "run_dir", None)
    if run_dir:
        output_summary = dict(summary)
        output_summary.update({
            "lr_end": gsom.lr_end,
            "nb_lr_fraction": gsom.nb_lr_fraction,
            "seed": init_seed,
            "init_method": init_method,
        })
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "gsom_growth_summary.json"), "w") as f:
            json.dump(output_summary, f, indent=2, default=str)

    preds = gsom.predict(encoded_data)
    qe = gsom.quantization_error(encoded_data)

    return gsom, preds, qe
