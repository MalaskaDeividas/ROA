from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Optional, Protocol, Sequence, Tuple


class Interface(Protocol):
    name: str
    n_jobs: int
    n_machines: int
    jobs: List[List[Tuple[int, int]]]


@dataclass
class ILSParams:
    iterations: int
    local_steps: int              #for ze hill climb steps per iteration
    pert_swaps: int               #how many random swaps
    seed: int
    accept_worse_prob: float      #probability to accept worse solutions
    verbose_every: int            #how often to print results 0 never
    init: str = "round_robin"     # how to startinitial solution, round_robin = job0,job1 and so on, while random is random lol

@dataclass
class ILSResults:
    best_sequence: List[int]      #list of the best jobs order
    best_time: int                #best finish time, total schedule length
    best_historically: List[int]  #best so far
