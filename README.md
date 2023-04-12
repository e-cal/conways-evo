# Evolving Interesting States in Conway's Game of Life

## Running the Algorithms

#### Baseline:

`python main.py`

```
usage: main.py [-h] [-f FILEPATH] [-d]

options:
  -h, --help            show this help message and exit
  -f FILEPATH, --filepath FILEPATH
                        Path to save logs and individuals (disables input prompt)
  -d, --delete          Delete the save directory if it already exists (instead of erroring out)
```

#### Multi-population island model:

`python island.py`

```
usage: island.py [-h] [-f FILEPATH] [-d]

options:
  -h, --help            show this help message and exit
  -f FILEPATH, --filepath FILEPATH
                        Path to save logs and individuals (disables input prompt)
  -d, --delete          Delete the save directory if it already exists (instead of erroring out)
```

## Render and evaluate an individual

`python render.py -f FILE`

Note: you may want to tweak the settings at the top of the file (`N_STEPS` and `UPDATE_RATE_MS`).

```
usage: render.py [-h] [--file FILE] [--debug] [--interactive]

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Path to the individual to load. Omitting will run the individual specified
                        in this file (init_pattern).
  --debug, -d           Print debug (evaluation) output
  --interactive, -i     Interactive mode. Step through evaluation steps.
```

## Plot evolution history

`python plot_history.py FILE [-s]`

Without `-s` it will display the plot, with `-s` it will save to `FILE.png`.
