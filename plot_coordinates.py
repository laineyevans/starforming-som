"""Shared coordinate helpers for neuron-map plotting."""

import numpy as np


def fixed_som_display_position(x, y, m_num, n_num):
    """Return the display-array position for fixed-SOM coordinate ``(x, y)``."""
    if not (0 <= x < m_num and 0 <= y < n_num):
        raise ValueError(f"Fixed-SOM coordinate {(x, y)} is outside {(m_num, n_num)}")
    return y, x


def fixed_som_subplot_index(x, y, m_num, n_num):
    """Return the Matplotlib subplot row/column for fixed-SOM ``(x, y)``."""
    row, column = fixed_som_display_position(x, y, m_num, n_num)
    return n_num - 1 - row, column


def configure_fixed_som_axes(ax, m_num, n_num):
    """Configure fixed-SOM axes with (0, 0) at bottom-left."""
    ax.set_xlim(-0.5, m_num - 0.5)
    ax.set_ylim(-0.5, n_num - 0.5)
    ax.set_xticks(np.arange(m_num))
    ax.set_yticks(np.arange(n_num))
    ax.set_aspect("equal")
