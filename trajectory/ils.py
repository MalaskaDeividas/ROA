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
    local_steps: int  # for ze hill climb steps per iteration
    pert_swaps: int  # how many random swaps
    seed: int
    accept_worse_prob: float  # probability to accept worse solutions
    verbose_every: int  # how often to print results 0 never
    # how to startinitial solution, order = job0,job1 and so on, while random is random lol
    init: str = "order"


@dataclass
class ILSResults:
    best_sequence: List[int]  # list of the best jobs order
    best_time: int  # best finish time, total schedule length
    best_historically: List[int]  # best so far


# get total time from jobs in a set
def get_makespan(instance: ThisType, seq: list[int]):

    nJ = instance.n_jobs
    next_op = [0]*nJ
    job_ready = [0]*nJ
    machine_ready = [0]*nJ

    for s in seq:
        n = next_op[s]
        m, p = instance.jobs[s][n]

        start = machine_ready[m] if machine_ready[m] > job_ready[s] else job_ready[s]
        finish = start + p

        machine_ready[m] = finish
        job_ready[s] = finish
        next_op[s] = n + 1

    print(f"job ready; {job_ready}, max job ready; {max(job_ready)}")
    return max(job_ready) if job_ready else 0

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

# fam


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
