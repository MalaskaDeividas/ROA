---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: ML_course
    language: python
    name: python3
---

# Random Optimization Algorithms 

For a population based algorithm, we chose Evolutionary Strategies with a (μ+λ) approach,
and for a trajectory based algorithm we chose iterated local search.
# instances 

```python
# get all instances fr m jobshop file

import parser.abz_parser as parser 
import random

with open("jobshop.txt", "r", encoding="utf-8") as f: 
  contents = f.read() 

instances = parser.parse_all_abz(contents)

```

# Algorithms 

## Trajectory 
```python

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Optional, Protocol, Sequence, Tuple


class ThisType(Protocol):
    name: str
    n_jobs: int
    n_machines: int
    jobs: List[List[Tuple[int, int]]]


@dataclass
class ILSParams:
    iterations: int
    local_steps: int              #for ze hill climb steps per iteration
    swaps: int                    #how many random swaps
    seed: int
    accept_worse_prob: float      # probability to accept worse solutions
    verbose_every: int            # how often to print results 0 never
    init: str = "order"           # how to startinitial solution, order = job0,job1 and so on, while random is random lol


@dataclass
class ILSResults:
    best_sequence: List[int]      #list of the best jobs order
    best_makespan: int                #best finish time, total schedule length
    best_historically: List[int]  #best so far
    

# get total time from jobs in a set
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

# how many operations job do
def job_op_counts(instance: ThisType):
    return [len(instance.jobs[n]) for n in range(instance.n_jobs)]

# two initial states, we either take the jobshop list as is in order aka "order" or random aka "random"
def initial_state(instance: ThisType, rng: random.Random, method="order"):
    counts = job_op_counts(instance)
    total_ops = sum(counts)

    if method == "random":
        seq: List[int] = []

        for i, c in enumerate(counts):
            # instead of another for loop aka for _ in range(c) do seq.append(i). This way is faster.
            seq.extend([i]*c)
        rng.shuffle(seq)
        return seq

    # if order
    seq = []
    rem = counts[:]

    while len(seq) < total_ops:
        for i in range(instance.n_jobs):

            if rem[i] > 0:
                seq.append(i)
                rem[i] -= 1

    print(f"Sequence from initial_state: {seq}")
    return seq

# swao two neighbors
def random_swap(seq, rng):
    i, j = rng.sample(range(len(seq)), 2)
    return i, j


def apply_swap(seq, i, j):
    seq[i], seq[j] = seq[j], seq[i]


def local_search(instance: ThisType, seq, rng, steps):
    s = seq[:]  # copy current sequence
    s_cost = get_makespan(instance, s)

    for _ in range(steps):
        i, j = random_swap(s, rng)
        apply_swap(s, i, j)
        s_candidate_cost = get_makespan(instance, s)

        if s_candidate_cost <= s_cost:
            s_cost = s_candidate_cost
        else:
            apply_swap(s, i, j)

    return s


# helps escape local maximum depending on value of swaps
def escape_local(seq, rng, swaps: int):
    s = seq[:]
    for _ in range(swaps):
        i, j = random_swap(s, rng)
        apply_swap(s, i, j)
    return s

def iterated_local_search(instance: ThisType, params: ILSParams, initial=None):
    rng         = random.Random(params.seed)

    #initiate seaquence and search
    s0          = initial[:] if initial is not None else initial_state(instance,rng,method=params.init)
    s           = local_search(instance, s0, rng, steps=params.local_steps)

    best        = s[:]
    best_cost   = get_makespan(instance,best)
    history     = [best_cost]

    for i in range(1, params.iterations + 1):
        S_ESC   = escape_local(s,rng,params.swaps)
        S_NEW   = local_search(instance,S_ESC,rng, steps=params.local_steps)

        cost_s  = get_makespan(instance,s)
        cost_new= get_makespan(instance,S_NEW)

        #we acccept
        if cost_new < cost_s:
            s = S_NEW
        else:
            if params.accept_worse_prob > 0 and rng.random() < params.accept_worse_prob:
                s = S_NEW
        
        cost_s = get_makespan(instance,s)
        if cost_s < best_cost:
            best = s[:]
            best_cost = cost_s
        
        history.append(best_cost)

        if i % 10 == 0:
            print("current:", cost_s, "best:", best_cost)

        if params.verbose_every and (i % params.verbose_every == 0):
            print(f"[ILS] iter={it} current={cost_s} best={best_cost}")
        
    return ILSResults(best_sequence=best, best_makespan=best_cost, best_historically=history)

```

```python
# test run of ILS algorithm 
inst = instances[5]
rand = 42 + random.randint(0, 1_000_000_000)
params = ILSParams(iterations=10, local_steps=10, swaps=7, seed=rand, accept_worse_prob=0.1, verbose_every=200, init="order")
res = iterated_local_search(inst, params)
job_ready = get_makespan(inst, res.best_sequence)
print("instance:", inst.name, inst.n_jobs, inst.n_machines)
print("best makespan:", res.best_makespan)
print("best job completion times:", job_ready)
    

```

## Population 

For the population based algorithm, we chose evolutionary strategies. 

The algorithm is designed such as a population of size lambda is randomly generated. 
Then, mu new genomes are created by random uniform selection of a parent from the previous population and then the mutation operation is applied. 
Lambda is a fixed value, set at 100 for this experiment, whereas mu will be adjusted according to Rechenberg's 1/5 success rule, such as mu will be incremented or decremented if more or less than 1/5 of new genomes outperform their parents. 
Then, the lambda best out of mu + lambda genomes "survive" and carry on to the next generation. 

For the final experiment, lambda is set to 100 and mu is set to 750, following the recommendation in [Introduction to Evolutionary Computation](https://link.springer.com/book/10.1007/978-3-662-44874-8), to set a ratio of lambda/mu greater than 7.

The mutation method used swaps places of two items in the array

```python
from typing import List, Tuple
#######
# Genetic Strategy
# Encoding: There are n machines and m jobs, where each job has n stepshttps://scheduleopt.github.io/benchmarks/jsplib/#overview-of-the-jobshop-benchmark.
# Candidate solutions are represented by an ordered list of integers from
# 0 to m exclusive, where the i:th occurance of an integer j means that at that point,
# the i:th task of job j is performed.


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
        print("ended simulation with mu =", self.mu_param)
        return best_score
```

```python
# test run for evolutionary strategies 
population = Population(inst) 
best_result = population.simulation() 
print(best_result)


```

<!-- #region -->


## Evolutionary strategies 

We decided on using these 10 problem for our benchmarking: yn1, la37, la02, orb01, la30, ft06, la22, swv02, orb10, abz8. They are a good mix of "easy" and "medium" problems, as defined by [JSPLib](https://scheduleopt.github.io/benchmarks/jsplib/#overview-of-the-jobshop-benchmark) 

<!-- #endregion -->

```python
test_instances_names = ["yn1", "la37", "la02", "orb01", "la30", "ft06", "la22", "swv02", "orb10", "abz8"] 
test_instances = [] 
for instance in instances: 
    name = instance.name.split()[1] 
    if name in test_instances_names: 
        test_instances.append(instance) 
```

# Results 

For our final experiment, we ran each of our 10 problem instances 5 times each for both algorithms and compared the results. 



```python
import matplotlib.pyplot as plt
global_results_ils = []
global_results_es = []
for instance in test_instances:
  results_ils = [] 
  results_es = [] 
  for i in range(5):
    rand = 42 + random.randint(0, 1_000_000_000)
    params = ILSParams(iterations=87, local_steps=200, swaps=2, seed=rand, accept_worse_prob=0.15, verbose_every=200, init="order")
    res = iterated_local_search(instance, params)
    job_ready = get_makespan(instance, res.best_sequence)
    results_ils.append(job_ready) 
    population = Population(problem_instance=inst, lambda_param=161, mu_param=444)
    population = Population(instance) 
    results_es.append(population.simulation())

  print("Instance:", instance.name)
  print(f"Best solution ils: {min(results_ils)}, best solution es: {min(results_es)}")
  print(f"Average ils: {sum(results_ils) / 5}, average es: {sum(results_es) / 5}")
  global_results_ils.append(min(results_ils))
  global_results_es.append(min(results_es))
plot_labels = [instance.name.split()[1] for instance in test_instances] 

```



```python
print(plot_labels)
plt.plot(plot_labels, global_results_ils, label="ils")
plt.plot(plot_labels, global_results_es, label="es")
optimal_solutions = [667, 55, 655, 927, 1355,1397,1059,944,1475,884]
plt.plot(plot_labels, optimal_solutions, linestyle="dotted",label="optimal",)
plt.legend()
plt.show()

```

As you can see by the plot, the evolution strategies algorithm outperforms ILS on every instance except ft06 (where both found the optimum of 55), and very closely follows the optimal solution line. 

However, it's worth noting that the runtime of the ES algorithm is several orders of magnitude higher than ILS, so it's probable that ILS could perform a lot better given more iterations. 
