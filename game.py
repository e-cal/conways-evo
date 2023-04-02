from enum import Enum

import numpy as np
import pygame
from evaluate import *

NCOLS = 60
NROWS = 60
CELLSIZE = 16

EVAL_WINDOW = (40, 54)

# TODO: no pygame if no render
RENDER = True
UPDATE_RATE_MS = 100


class Color(Enum):
    ALIVE = (255, 255, 215)
    DYING = (200, 200, 225)
    BG = (10, 10, 40)
    GRID = (30, 30, 60)


def update(surface, cur):
    nxt = np.zeros((cur.shape[0], cur.shape[1]))  # all cells dead by default

    for r, c in np.ndindex(cur.shape):
        alive_neighbors = np.sum(cur[r - 1 : r + 2, c - 1 : c + 2]) - cur[r, c]

        color = Color.BG.value

        if cur[r, c] == 1 and alive_neighbors < 2 or alive_neighbors > 3:
            color = Color.DYING.value

        elif (
            (cur[r, c] == 1 and 2 <= alive_neighbors <= 3) or 
            (cur[r, c] == 0 and alive_neighbors == 3)  # fmt: skip
        ):
            nxt[r, c] = 1
            color = Color.ALIVE.value

        color = color if cur[r, c] == 1 else Color.BG.value
        pygame.draw.rect(
            surface,
            color,
            (c * CELLSIZE, r * CELLSIZE, CELLSIZE - 1, CELLSIZE - 1),
        )

    return nxt


def test_init():
    # try a manual pattern

    # fmt: off
    pattern = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                        [0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]);
    # fmt: on

    cells = np.zeros((NROWS, NCOLS))

    pos = (3, 3)
    cells[
        pos[0] : pos[0] + pattern.shape[0], pos[1] : pos[1] + pattern.shape[1]
    ] = pattern
    return cells


def init():
    cells = np.zeros(NROWS * NCOLS)

    # genetic alg sets cells

    cells = test_init()

    return cells.reshape((NROWS, NCOLS))


def main():
    pygame.init()
    GAME_TICK = pygame.USEREVENT + 1
    pygame.time.set_timer(GAME_TICK, millis=UPDATE_RATE_MS)
    surface = pygame.display.set_mode((NCOLS * CELLSIZE, NROWS * CELLSIZE))

    cells = init()
    fitness = 0

    step = 0

    while step <= 100:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.unicode == "q":
                    pygame.quit()
                    return

            if event.type == GAME_TICK:
                surface.fill(Color.GRID.value)
                cells = update(surface, cells)

                if EVAL_WINDOW[0] <= step <= EVAL_WINDOW[1]:
                    fitness += evaluate(cells)

                pygame.display.update()
                step += 1

    print(fitness)


if __name__ == "__main__":
    main()
