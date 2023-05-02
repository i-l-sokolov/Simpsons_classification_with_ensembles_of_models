import numpy as np


def min_columns_cover(matrix):
    rows, cols = matrix.shape
    covered = np.zeros(rows, dtype=bool)
    selected_columns = []

    while not np.all(covered):
        uncovered_rows = np.logical_not(covered)
        gains = (matrix[uncovered_rows, :]).sum(axis=0)
        max_col = np.argmax(gains)
        selected_columns.append(max_col)
        covered = np.logical_or(covered, matrix[:, max_col])

    return selected_columns