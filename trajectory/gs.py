from typing import List, Tuple
from ils import ThisType, get_makespan

import random
#######
# Genetic Strategy
# Encoding: There are n machines and m jobs, where each job has n steps.
# Candidate solutions are represented by an ordered list of integers from
# 0 to m exclusive, where the i:th occurance of an integer j means that at that point,
# the i:th task of job j is performed.


def job_based_crossover(parent1: List[int], parent2: List[int], n_jobs: int):
    mask = random.choices([0, 1], k=n_jobs)
    print(mask)
    mask_lookup = {i for i, val in enumerate(mask) if val == 1}
    print(mask_lookup)
    offspring = [-1 for _ in range(len(parent1))]
    for i, val in enumerate(parent1):
        if val in mask_lookup:
            offspring[i] = val
    remaining_positions = [i for i, val in enumerate(offspring) if val == -1]
    current_pos = 0
    for val in parent2:
        if val not in mask_lookup:
            offspring[remaining_positions[current_pos]] = val
            current_pos += 1
    return offspring


def generate_random_genome(n_machines: int, n_jobs: int):
    res = []
    for _ in range(n_jobs):
        res.extend(random.sample(range(n_jobs), n_jobs))
    return tuple(res)


# Generate a random population of size size.
# Stores generated genomes in a set to ensure uniqueness of candidate solutions
def generate_random_population(size: int, n_machines: int, n_jobs: int):
    genomes = set()

    current_size = 0
    while True:
        if current_size == size:
            break
        genome = generate_random_genome(n_machines, n_jobs)
        if genome not in genomes:
            current_size += 1
            genomes.add(genome)
    return list(genomes)
# TODO:
# - Write generate random genome function
# - Score them using Deivids method
# - mu parents and lambda offspring
# - Implement the 1/5th rule to modify crossover rate
# - skip mutation, f that shit
# - Run until convergence, i.e certain number of generations without improvement


class Population():
    def __init__(self, problem_instance: ThisType,
                 population_size: int = 200,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.5):
        self.problem_instance = problem_instance
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = generate_random_population(
            population_size, problem_instance.n_jobs, problem_instance.n_machines)

    def simulate_generation(self):
        
        pass
