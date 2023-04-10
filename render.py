import os
from enum import Enum

import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from evaluate import evaluate
from main import update

NCOLS = 60
NROWS = 60
CELLSIZE = 16

UPDATE_RATE_MS = 100

EVAL_WINDOW = (40, 54)
N_STEPS = EVAL_WINDOW[1]  # end when done evaluating


class Color(Enum):
    ALIVE = (255, 255, 215)
    DYING = (200, 200, 225)
    BG = (10, 10, 40)
    GRID = (30, 30, 60)


def draw_cells(cur, surface):
    surface.fill(Color.GRID.value)
    for row, col in np.ndindex(cur.shape):
        color = Color.BG.value

        if cur[row, col] == 1:
            color = Color.ALIVE.value
        elif cur[row, col] == 0:
            color = Color.BG.value

        pygame.draw.rect(
            surface,
            color,
            (col * CELLSIZE, row * CELLSIZE, CELLSIZE - 1, CELLSIZE - 1),
        )

    pygame.display.update()


def load(path):
    cells = np.load(path)
    return cells


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


def run(cells, debug=False, interactive=False):
    pygame.init()
    GAME_TICK = pygame.USEREVENT + 1
    pygame.time.set_timer(GAME_TICK, millis=UPDATE_RATE_MS)
    surface = pygame.display.set_mode((NCOLS * CELLSIZE, NROWS * CELLSIZE))

    fitness = 0
    step = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.unicode == "q":
                    pygame.quit()
                    return

            if event.type == GAME_TICK:
                draw_cells(cells, surface)

                if EVAL_WINDOW[0] <= step <= EVAL_WINDOW[1]:
                    # fmt: off
                    if debug: print(f"Evaluating step {step}")
                    fitness += evaluate(cells, debug=debug)
                    if interactive: input()
                    # fmt: on

                cells = update(cells)
                step += 1
                if step > N_STEPS:
                    return fitness


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true")
    args = vars(parser.parse_args())
    fp = args["file"]
    debug = args["debug"]
    interactive = args["interactive"]

    if not debug:
        print("Debug mode off, use -d to see debug output.")
    if not interactive:
        print("Interactive mode off, use -i to pause after each evaluation step.")

    if fp is None:
        print("No file specified, initializing pattern.")
        cells = init_pattern()
    else:
        print(f"Loading {fp}")
        cells = load(fp)

    fitness = run(cells, debug, interactive)
    print(f"Fitness: {fitness}")
