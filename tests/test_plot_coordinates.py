import numpy as np

from plot_coordinates import (
    configure_fixed_som_axes,
    fixed_som_display_position,
    fixed_som_subplot_index,
)


def test_fixed_som_non_square_orientation():
    m_num, n_num = 3, 4
    values = np.arange(m_num * n_num).reshape(n_num, m_num)

    assert fixed_som_display_position(0, 0, m_num, n_num) == (0, 0)
    assert fixed_som_display_position(m_num - 1, n_num - 1, m_num, n_num) == (n_num - 1, m_num - 1)
    assert values[0, 0] == 0
    assert values[n_num - 1, m_num - 1] == m_num * n_num - 1

    top_right_row, top_right_column = fixed_som_subplot_index(
        m_num - 1, n_num - 1, m_num, n_num
    )
    bottom_left_row, bottom_left_column = fixed_som_subplot_index(
        0, 0, m_num, n_num
    )
    assert (top_right_row, top_right_column) == (0, m_num - 1)
    assert (bottom_left_row, bottom_left_column) == (n_num - 1, 0)


def test_fixed_som_hits_and_u_matrix_share_axis_orientation():
    import matplotlib.pyplot as plt

    m_num, n_num = 3, 4
    fig, ax = plt.subplots()
    configure_fixed_som_axes(ax, m_num, n_num)
    assert ax.get_xlim() == (-0.5, m_num - 0.5)
    assert ax.get_ylim() == (-0.5, n_num - 0.5)
    plt.close(fig)


def test_gsom_negative_coordinates_keep_raw_upward_orientation():
    import matplotlib.pyplot as plt

    coordinates = [(-1, -2), (0, -2), (-1, -1), (0, 0)]
    fig, ax = plt.subplots()
    xs = np.asarray([x for x, _ in coordinates])
    ys = np.asarray([y for _, y in coordinates])
    ax.scatter(xs, ys)
    ax.set_xlim(xs.min() - 0.8, xs.max() + 0.8)
    ax.set_ylim(ys.min() - 0.8, ys.max() + 0.8)
    assert ax.get_xlim()[0] < -1 < ax.get_xlim()[1]
    assert ax.get_ylim()[0] < -2 < ax.get_ylim()[1]
    assert not ax.yaxis_inverted()
    plt.close(fig)
