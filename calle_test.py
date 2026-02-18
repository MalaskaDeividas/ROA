import population.gs
import trajectory.ils 
import parser.abz_parser

with open("jobshop.txt") as f: 
    contents = f.read()

instances = parser.abz_parser.parse_all_abz(contents)

instance = instances[0]
print(instance.n_machines)

gs_pop = population.gs.Population(instance)
for pop in gs_pop.population: 
    print("result:", trajectory.ils.get_makespan(instance, pop))


print(gs_pop)
