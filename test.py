import numpy as np


def matrix_hash(mat):
    return np.sum(mat)


def find_submatrix(target, search_space):
    target_hash = matrix_hash(target)
    print(target_hash)
    target_rows, target_cols = target.shape
    search_rows, search_cols = search_space.shape

    match_matrix = np.zeros(
        (search_rows - target_rows + 1, search_cols - target_cols + 1)
    )

    for i in range(search_rows - target_rows + 1):
        for j in range(search_cols - target_cols + 1):
            sub_matrix = search_space[i : i + target_rows, j : j + target_cols]
            match_matrix[i, j] = matrix_hash(sub_matrix)

    match_indices = np.where(match_matrix == target_hash)

    for i, j in zip(match_indices[0], match_indices[1]):
        if np.array_equal(
            search_space[i : i + target_rows, j : j + target_cols], target
        ):
            return (i, j)

    return None


target = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
search_space = np.array(
    [[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0]]
)

result = find_submatrix(target, search_space)
print(result)
