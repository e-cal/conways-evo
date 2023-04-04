import time
from enum import Enum
from pprint import pprint

import numpy as np
import pygame

from evaluate import evaluate
from mutation import Mutation

# Game
NCOLS = 60
NROWS = 60
CELLSIZE = 16

# Genetic Algorithm
POP_SIZE = 64
EVAL_WINDOW = (40, 54)
MAX_GENS = 400
N_STEPS = EVAL_WINDOW[1]  # end when done evaluating

# Rendering
RENDER = False
UPDATE_RATE_MS = 100


class Color(Enum):
    ALIVE = (255, 255, 215)
    DYING = (200, 200, 225)
    BG = (10, 10, 40)
    GRID = (30, 30, 60)


def update(cur, surface=None):
    nxt = np.zeros((cur.shape[0], cur.shape[1]))  # all cells dead by default

    for row, col in np.ndindex(cur.shape):
        alive_neighbors = (
            np.sum(cur[row - 1 : row + 2, col - 1 : col + 2]) - cur[row, col]
        )

        color = Color.BG.value

        if cur[row, col] == 1 and alive_neighbors < 2 or alive_neighbors > 3:
            color = Color.DYING.value

        elif (
            (cur[row, col] == 1 and 2 <= alive_neighbors <= 3) or 
            (cur[row, col] == 0 and alive_neighbors == 3)  # fmt: skip
        ):
            nxt[row, col] = 1
            color = Color.ALIVE.value

        color = color if cur[row, col] == 1 else Color.BG.value

        if surface is not None:
            pygame.draw.rect(
                surface,
                color,
                (col * CELLSIZE, row * CELLSIZE, CELLSIZE - 1, CELLSIZE - 1),
            )

    return nxt


def init_pattern():
    """Input a manual pattern to initialize the game with."""

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
    cells = np.random.randint(low=0, high=2, size=(POP_SIZE, NROWS, NCOLS))
    return cells


def run_pygame(cells, surface):
    pygame.init()
    GAME_TICK = pygame.USEREVENT + 1
    pygame.time.set_timer(GAME_TICK, millis=UPDATE_RATE_MS)
    surface = pygame.display.set_mode((NCOLS * CELLSIZE, NROWS * CELLSIZE))

    fitness = 0
    for step in range(N_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.unicode == "q":
                    pygame.quit()
                    return

            if event.type == GAME_TICK:
                surface.fill(Color.GRID.value)
                cells = update(cells, surface)

                if EVAL_WINDOW[0] <= step <= EVAL_WINDOW[1]:
                    fitness += evaluate(cells)

                pygame.display.update()

    return fitness


def run(cells):
    fitness = 0
    for step in range(N_STEPS):
        cells = update(cells)
        if EVAL_WINDOW[0] <= step <= EVAL_WINDOW[1]:
            fitness += evaluate(cells)

    return fitness


def main():
    # init
    population = init()
    fitnesses = [0] * POP_SIZE

    for gen in range(MAX_GENS):
        for i, individual in enumerate(population):
            start = time.time()
            fitnesses[i] = run(individual)
            print(f"Gen {gen} individual {i} fitness {fitnesses[i]}")
            print(f"Took: {time.time() - start:.2f}s")

        print(fitnesses)
        exit()
        chrom = Selection.tournament(fitnesses, POP_SIZE, chrom)
        print(chrom)
        print(f"Generation {gen} done")

    # 1 gen


if __name__ == "__main__":
    if not RENDER:
        main()
