from typing import List, Tuple
import random 
#######
# Genetic Strategy
# Encoding: There are n machines and m jobs, where each job has n stepshttps://scheduleopt.github.io/benchmarks/jsplib/#overview-of-the-jobshop-benchmark.
# Candidate solutions are represented by an ordered list of integers from
# 0 to m exclusive, where the i:th occurance of an integer j means that at that point,
# the i:th task of job j is performed.



def get_makespan(instance: ThisType, seq: list[int]):

    next_op       = [0]* instance.n_jobs
    job_ready     = [0]*instance.n_jobs
    machine_ready = [0]*instance.n_machines

    for s in seq:
        n = next_op[s]
        m, p = instance.jobs[s][n]

        start = machine_ready[m] if machine_ready[m] > job_ready[s] else job_ready[s]
        finish = start + p

        machine_ready[m] = finish
        job_ready[s] = finish
        next_op[s] = n + 1
    
    #print(f"job ready; {job_ready}, max job ready; {max(job_ready)}")
    return max(machine_ready)


def job_based_crossover(parent1: List[int], parent2: List[int], n_jobs: int):
    mask = random.choices([0, 1], k=n_jobs)
    mask_lookup = {i for i, val in enumerate(mask) if val == 1}
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


def generate_random_genome(n_jobs: int, n_machines: int):
    res = []
    for _ in range(n_machines):
        res.extend(random.sample(range(n_jobs), n_jobs))
    for i in range(n_jobs):
        assert res.count(i) == n_machines, f"Expected {n_machines} occurances of i, got {res.count(i)}"
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
# - Run until convergence, i.e certain number of generations without improvement


class Population():
    def __init__(self, problem_instance: ThisType,
                  lambda_param: int = 100, 
                  mu_param: int = 750, 
                  max_iterations_without_improvement: int = 100,
                  new_genomes_per_gen: int = 50
                 ):
        self.problem_instance = problem_instance
        self.lambda_param = lambda_param 
        self.mu_param = mu_param
        self.population = generate_random_population(
            lambda_param, problem_instance.n_jobs, problem_instance.n_machines)
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.new_genomes_per_gen = new_genomes_per_gen
        self.history_best = []
    def simulate_generation(self):
        random.seed()
        current_idx = 0
        new_genomes = [] 
        no_better_children = 0 
        for i in range(self.mu_param):
            parent = random.choice(self.population)
            new_genome = mutate(parent)            
            new_genomes.append(new_genome)
            if get_makespan(self.problem_instance, new_genome) < get_makespan(self.problem_instance, parent):
                no_better_children += 1
        old_population = self.population.copy()
        self.population.extend(new_genomes)
        self.population.sort(key=lambda x: get_makespan(
            self.problem_instance, x))
        self.population = self.population[:self.lambda_param]
        old_population.sort(key=lambda x: get_makespan(self.problem_instance, x))

        # apply Rechenberg's 1/5 success rule
        if no_better_children > self.mu_param // 5:   
            self.mu_param += 1 
        elif no_better_children < self.mu_param // 5: 
            self.mu_param -= 1
    def simulation(self):
        current_gen = 0
        gens_without_improvement = 0
        best_score = float("inf")
        print("Starting run with instance", self.problem_instance.name)
        while True:
            self.simulate_generation()
            current_best_score = get_makespan(self.problem_instance, self.population[0])
            if current_best_score >= best_score:
                gens_without_improvement += 1
            else:
                best_score = current_best_score
                gens_without_improvement = 0
            if gens_without_improvement == self.max_iterations_without_improvement:
                break
            # best-so-far curve
            self.history_best.append(best_score)

        return best_score
