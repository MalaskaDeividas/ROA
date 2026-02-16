from __future__ import annotations
from typing import List, Tuple

class Schedule_Instance:
    
    def __init__(
        self,
        name: str,
        n_jobs: int,
        n_machines: int,
        jobs: List[List[Tuple[int, int]]],
    ) -> None:
        self.name = name
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.jobs = jobs