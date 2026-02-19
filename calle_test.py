import population.gs
import trajectory.ils 
import parser.abz_parser
from tuner.tune import run_tuner
with open("jobshop.txt") as f: 
    contents = f.read()

instances = parser.abz_parser.parse_all_abz(contents)

instance = instances[0]
print(instance.n_machines)

run_tuner()
