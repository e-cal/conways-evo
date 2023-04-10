import numpy as np

###############################
# Parent Selection Algorithms
###############################


def nbest(population, fitnesses, n):
    """Best n parent selection"""
    idx = np.array(fitnesses).argsort()[::-1][:n]
    return np.array(population)[idx].tolist(), np.array(fitnesses)[idx].tolist()


###############################
# Survivor Selection Algorithms
###############################


def mu_plus_lambda(current_pop, current_fitness, offspring, offspring_fitness):
    """mu+lambda selection"""
    population = []
    fitness = []

    # select the mu fittest individuals from both pools
    mu = len(current_pop)
    pool = np.array(current_pop + offspring)
    fp = np.array(current_fitness + offspring_fitness)

    # sort the fitness and pick the top mu choices
    idx = fp.argsort()[::-1][:mu]

    # add the top mu choices to the new population
    for i in range(mu):
        ix = idx[i]
        population.append(pool[ix].tolist())
        fitness.append(fp[ix].tolist())

    return population, fitness


def replacement(current_pop, current_fitness, offspring, offspring_fitness):
    """replacement selection"""

    population = []
    fitness = []

    mu = len(current_fitness)

    # sort the fitness and replace the worst mu individuals
    idx = np.array(current_fitness).argsort()[::-1][:mu]

    for i in range(mu - len(offspring)):
        ix = idx[i]
        population.append(np.array(current_pop)[ix].tolist())
        fitness.append(np.array(current_fitness)[ix].tolist())

    population += offspring
    fitness += offspring_fitness

    return population, fitness
