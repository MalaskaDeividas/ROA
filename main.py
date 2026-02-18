from parser.abz_parser import parse_all_abz, print_all
from trajectory.ils import ILSParams, iterated_local_search, get_makespan
from population.gs import Population
import random

def run_specific_ils():

    with open("jobshop.txt", "r", encoding="utf-8") as f:
        jobshop = f.read()

    instances = parse_all_abz(jobshop)
    inst = instances[5]
    rand = 42 + random.randint(0, 1_000_000_000)

    params = ILSParams(iterations=10, local_steps=10, swaps=7, seed=rand, accept_worse_prob=0.1, verbose_every=200, init="order")
    res = iterated_local_search(inst, params)
    job_ready = get_makespan(inst, res.best_sequence)
    print("instance:", inst.name, inst.n_jobs, inst.n_machines)
    print("best makespan:", res.best_makespan)
    print("best job completion times:", job_ready)
    
def run_gs(): 
    with open("jobshop.txt", "r", encoding="utf-8") as f: 
        jobshop = f.read() 
    instances = parse_all_abz(jobshop)
    for instance in instances: 
        inst = instance 
        if inst.name == "instance la22": 
            break 
    print(inst.name)
    print(inst.jobs[0])
    print(inst.n_machines)
    print(inst.n_jobs)
    population = Population(
            inst
            )
    solution = population.simulation() 
    print("Solution found:", solution)




def main():
    return
    

if __name__ == "__main__":
    run_specific_ils()
