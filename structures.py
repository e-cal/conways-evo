import numpy as np

"""                                             structure id                """
GLIDERS = [
    # up left
    np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]]),  # 0
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]]),  # 1
    np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]]),  # 2
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1]]),  # 3
    # up right
    np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]]),  # 4
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]]),  # 5
    np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]]),  # 6
    np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]]),  # 7
    # down left
    np.array([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),  # 8
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]]),  # 9
    np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]]),  # 10
    np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]]),  # 11
    # down right
    np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]),  # 12
    np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]]),  # 13
    np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]),  # 14
    np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]]),  # 15
]

R_PENTIMINOS = [
    # upright
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),  # 16
    # upright flipped
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]]),  # 17
    # upside down
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]]),  # 18
    # upside down flipped
    np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0]]),  # 19
    # right
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]]),  # 20
    # right flipped
    np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]),  # 21
    # left
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]]),  # 22
    # left flipped
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]]),  # 23
]

EXPLODERS = [
    # up
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]),  # 24
    # right
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]]),  # 25
    # down
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]]),  # 26
    # left
    np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]]),  # 27
]

STRUCTURES = GLIDERS + R_PENTIMINOS + EXPLODERS
STRUCTURES = [np.pad(s, 1, "constant") for s in STRUCTURES]

STRUCTURE_NAMES = (
    ["glider"] * len(GLIDERS)
    + ["rpentimino"] * len(R_PENTIMINOS)
    + ["exploder"] * len(EXPLODERS)
)
