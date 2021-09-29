#!/env/python

from typing import List
from definitions import Mem, MemList, Intrinsic, IntrinsicsList
from instructions import instructions

memory_addresses = [
    Mem("p", 10),
    Mem("p", 9),
    Mem("p", 2),
    Mem("p", 0),
]


class Candidate:
    def get_new_register(self, reg_type: str = "__m128") -> str:
        new_reg_name = f"__tmpreg{reg_type}_{self.registers[reg_type]}"
        self.registers[reg_type] += 1
        return new_reg_name

    def __init__(self, instructions: List[Intrinsic]):
        self.instructions = instructions
        self.reg_no = 0
        self.registers = {"__m128": 0, "__m256": 0}


def choose_best_candidates(mem: MemList, candidates: IntrinsicsList):
    return candidates


candidates = []
for mem in memory_addresses:
    candidates = choose_best_candidates(mem, candidates)

print(f"Best options for memory addresses = {memory_addresses}")
for candidate in candidates:
    print(f"\t{candidate}")
