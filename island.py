from main import (DEFUALT_PATH, argparse, async_run, init, mutate, np, os,
                  recombine, select_parents, select_survivors, tqdm, trange)

POP_SIZE = 64
MAX_GENS = 400
N_SUB_POPULATIONS = 4
MIGRATION_FREQUENCY = 20  # migrate every n generations
MIGRATION_RATE = 0.2  # proportion of individuals to migrate

SUB_POP_SIZE = POP_SIZE // N_SUB_POPULATIONS
N_PARENTS = round(0.125 * SUB_POP_SIZE)  # same ratio as 8:64


def generate_offspring(parent_idxs, population):
    offspring = []
    for _ in range(N_PARENTS // 2):
        # randomly pair parents, popping them from the pool
        parent1 = population[parent_idxs.pop(np.random.randint(len(parent_idxs)))]
        parent2 = population[parent_idxs.pop(np.random.randint(len(parent_idxs)))]

        offspring.extend(recombine(parent1, parent2))

    return offspring


def ga(fp):
    print(
        f"Starting genetic algorithm with {POP_SIZE} random individuals spilt into {N_SUB_POPULATIONS} sub populations\n"
    )
    population = init()

    bar_format = "{l_bar}{bar}| eta: {remaining}"

    with tqdm(
        total=POP_SIZE,
        desc="Evaluating initial population",
        bar_format=bar_format,
        position=1,
        leave=False,
    ) as progbar:
        fitnesses = async_run(population, progbar)

    populations = np.array_split(population, N_SUB_POPULATIONS)
    fitnesses = np.array_split(fitnesses, N_SUB_POPULATIONS)
    fitnesses = [f.tolist() for f in fitnesses]

    # Create a progress bar
    gen_progress = trange(
        1,
        MAX_GENS + 1,
        desc=f"Generation 0/{MAX_GENS}",
        bar_format=bar_format,
        position=0,
        leave=False,
    )
    metrics = trange(1, 1, bar_format="metrics{postfix}", position=1, leave=False)
    best = [max(fitnesses[i]) for i in range(N_SUB_POPULATIONS)]
    avg = [f"{np.mean(fitnesses[i]):.2f}" for i in range(N_SUB_POPULATIONS)]
    metrics.set_postfix(best=best, avg=avg)

    for gen in gen_progress:
        for i, population in enumerate(populations):
            # select parents
            parent_idxs = select_parents(fitnesses[i], N_PARENTS)

            # generate offspring
            offspring = generate_offspring(parent_idxs, population)

            # mutate offspring
            offspring = list(map(mutate, offspring))

            # evaluate offspring
            offspring_fitnesses = async_run(offspring)

            # select survivors
            if isinstance(population, np.ndarray):
                population = population.tolist()

            populations[i], fitnesses[i] = select_survivors(
                population, fitnesses[i], offspring, offspring_fitnesses
            )

        # Update the logging output underneath the progress bar
        gen_progress.set_description(f"Generation {gen}/{MAX_GENS}")
        best = [max(fitnesses[i]) for i in range(N_SUB_POPULATIONS)]
        avg = [f"{np.mean(fitnesses[i]):.2f}" for i in range(N_SUB_POPULATIONS)]
        metrics.set_postfix(best=best, avg=avg)

        log_and_save(populations, fitnesses, gen, fp)

        if gen % MIGRATION_FREQUENCY == 0:
            metrics.set_postfix_str(f"best={best}, avg={avg}, migrating...")
            for i in range(N_SUB_POPULATIONS):
                idxs = np.random.choice(
                    SUB_POP_SIZE,
                    round(MIGRATION_RATE * SUB_POP_SIZE),
                    replace=False,
                )

                for idx in sorted(idxs, reverse=True):
                    populations[(i + 1) % N_SUB_POPULATIONS].append(populations[i].pop(idx))  # fmt: skip
                    fitnesses[(i + 1) % N_SUB_POPULATIONS].append(fitnesses[i].pop(idx))  # fmt: skip

    # hack to clear and close metrics "progress bar"
    for _ in metrics:
        pass

    best = [max(fitnesses[i]) for i in range(N_SUB_POPULATIONS)]
    print(f"\nCompleted {MAX_GENS} generations.\nBest fitnesses: {best}")


def log_and_save(population, fitnesses, gen, fp):
    if not os.path.exists(f"{fp}/log_all.csv"):
        with open(f"{fp}/log_all.csv", "w") as f:
            f.write("generation")
            for i in range(N_SUB_POPULATIONS):
                f.write(f",max{i},avg{i},min{i}")

    max_fitnesses = [max(f) for f in fitnesses]
    avg_fitnesses = [np.mean(f) for f in fitnesses]
    min_fitnesses = [min(f) for f in fitnesses]
    with open(f"{fp}/log_all.csv", "a") as f:
        f.write(f"\n{gen}")
        for i in range(N_SUB_POPULATIONS):
            f.write(f",{max_fitnesses[i]},{avg_fitnesses[i]},{min_fitnesses[i]}")

    # flatten fitnesses and population
    fitnesses = [f for sub in fitnesses for f in sub]
    population = [p for sub in population for p in sub]

    if not os.path.exists(f"{fp}/log.csv"):
        with open(f"{fp}/log.csv", "w") as f:
            f.write("generation,max,avg,min")

    with open(f"{fp}/log.csv", "a") as f:
        f.write(f"\n{gen},{max(fitnesses)},{np.mean(fitnesses)},{min(fitnesses)}")

    max_fitness = max(fitnesses)
    n = 1
    for individual, fitness in zip(population, fitnesses):
        if fitness == max_fitness:
            np.save(f"{fp}/gen{gen}_{n}.npy", individual)
            n += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        help="Path to save logs and individuals (disables input prompt)",
    )
    parser.add_argument(
        "-d",
        "--delete",
        help="Delete the save directory if it already exists (instead of erroring out)",
        action="store_true",
    )
    args = parser.parse_args()

    path = args.filepath
    if path is None:
        path = input(
            f"Enter path to save logs and individuals (default={DEFUALT_PATH}): "
        )
        if path == "":
            path = DEFUALT_PATH

        if path[-1] == "/":
            path = path[:-1]

    if args.delete:
        if os.path.exists(path):
            print(f"Deleting {path}")
            os.system(f"rm -r {path}")
    try:
        os.makedirs(path, exist_ok=False)
    except FileExistsError:
        print(
            f'Directory "{path}" already exists. Please move/delete it or choose a different path.'
        )
        exit()

    ga(path)


if __name__ == "__main__":
    main()
