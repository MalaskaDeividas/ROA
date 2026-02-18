from typing import List, Tuple
from trajectory.ils import ThisType, get_makespan
import time

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
    for j in range(n_jobs):
        res.extend([j] * n_machines)
    random.shuffle(res)
    for j in range(n_jobs):
        assert res.count(j) == n_machines, f"job {j} count {res.count(j)} != {n_machines}"
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


def mutate(genome: Tuple[int]):
    i, j = random.sample(range(len(genome)), 2)
    genome = list(genome)
    genome[i], genome[j] = genome[j], genome[i]
    return genome

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
                 crossover_rate: float = 0.5,
                 max_iterations_without_improvement: int = 100,
                 new_genomes_per_gen: int = 50
                 ):
        self.problem_instance = problem_instance
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = generate_random_population(
            population_size, problem_instance.n_machines, problem_instance.n_jobs)
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.new_genomes_per_gen = new_genomes_per_gen
        self.history_best = []

    def simulate_generation(self):
        #random.seed()
        self.population.sort(key=lambda x: get_makespan(
            self.problem_instance, x))
        current_idx = 0
        new_genomes = []
        while True:
            if len(new_genomes) == self.new_genomes_per_gen:
                break
            current_idx %= self.population_size
            mutation_check = random.random() < self.mutation_rate
            if mutation_check:
                new_genome = mutate(self.population[current_idx])
                new_genomes.append(new_genome)

            current_idx += 1

        self.population.extend(new_genomes)
        self.population.sort(key=lambda x: get_makespan(
            self.problem_instance, x))
        self.population = self.population[:self.population_size]

    def simulation(self, verbose_every: int = 0):
        gens_without_improvement = 0
        best_score = float("inf")
        gen = 0

        self.history_best = []  # reset every run

        while gens_without_improvement < self.max_iterations_without_improvement:
            self.simulate_generation()
            gen += 1

            current_best = get_makespan(self.problem_instance, self.population[0])

            if current_best < best_score:
                best_score = current_best
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            # best-so-far curve
            self.history_best.append(best_score)

            if verbose_every and (gen % verbose_every == 0):
                print(
                    f"[GA] gen={gen} current={current_best} best={best_score} "
                    f"no_improve={gens_without_improvement}"
                )

        return best_score
