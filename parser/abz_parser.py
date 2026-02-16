

from __future__ import annotations
from typing import List, Tuple
from .schedule_instance import Schedule_Instance

def parse_all_abz(text) -> List[Schedule_Instance]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    instances: List[Schedule_Instance] = []

    i = 0
    while i < len(lines):
        pairs = lines[i].split()
        if len(pairs) == 2 and all(p.lstrip("-").isdigit() for p in pairs):

            n_jobs, n_machines = map(int, pairs)
            job_lines          = lines[i+1:i+1+n_jobs]

            if len(job_lines) != n_jobs:
                break

            jobs = []
            good = True

            for job in job_lines:
                n = job.split()

                if len(n) != 2*n_machines:
                    good   = False
                    break

                ints       = list(map(int,n))
                ops        = [(ints[k], ints[k+1]) for k in range(0, len(ints), 2)]

                jobs.append(ops)

            if good:
                instances.append(
                    Schedule_Instance(
                        name=f"instance={len(instances)}",
                        n_jobs=n_jobs,
                        n_machines=n_machines,
                        jobs=jobs,
                    )
                )
                i = i+1+n_jobs
        i += 1
    return instances



def print_all():
    with open("jobshop.txt", "r", encoding="utf-8") as file:
        jobshop = file.read()

    instances = parse_all_abz(jobshop)

    print("count:", len(instances))

    for x, instance in enumerate(instances):
        print(f"\n=== {x} {instance.name} ({instance.n_jobs} jobs, {instance.n_machines} machines) ===")
        for job_id, ops in enumerate(instance.jobs):
            print(f"job {job_id}: {ops}")
    return
