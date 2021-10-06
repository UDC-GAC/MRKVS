#!/env/python

from typing import List
from definitions import Mem, MemList, Intrinsic, Instruction, Reg
from instructions import instructions


def gen_empty_reg(size: int = 4):
    return [Reg([Mem("p", "-1")] * size)]


class Candidate:
    def get_new_register(self, reg_type: str = "__m128") -> str:
        new_reg_name = f"__tmpreg{reg_type}_{self.registers[reg_type]}"
        self.registers[reg_type] += 1
        return new_reg_name

    def add_instruction(self, instruction: dict) -> int:
        new_loads = 0
        for idx in range(self.objective.get_size()):
            r = instruction.res
            if idx < r.get_size() and r[idx] in self.objective:
                new_loads += 1
                self.already_loaded[self.objective.slots.index(r[idx])] = 1
        if new_loads > 0:
            self.instructions.append(instruction)
        return new_loads

    def is_complete(self) -> bool:
        return sum(self.already_loaded) == self.objective.get_size()

    def is_sorted(self) -> bool:
        for idx in range(self.objective.get_size()):
            if self.objective[idx] != self.output[idx]:
                return False
        return True

    def get_missing_addresses(self) -> List[int]:
        return [
            idx
            for idx in range(len(self.already_loaded))
            if self.already_loaded[idx] == 0
        ]

    def __str__(self):
        s = ""
        for i in range(len(self.instructions) - 1, -1, -1):
            s += f"{self.instructions[i]}\n"
        return f"{s}"

    def __repr__(self) -> str:
        return f"{self.__str__()}"

    def __init__(self, objective: Reg):
        self.objective = objective
        self.already_loaded = [0] * objective.get_size()
        self.instructions = []
        self.output = []
        self.reg_no = 0
        self.registers = {"__m128": 0, "__m256": 0}


def generate_seed_candidates(objective: MemList):
    candidates = []
    mem = objective[-1]
    for ins in instructions["load"]:
        if ins.output.get_size() > objective.get_size():
            continue
        new_candidate = Candidate(objective)
        new_inst = Instruction(ins, ins.evaluate_output(mem.idx, gen_empty_reg()))
        new_loads = new_candidate.add_instruction(new_inst)
        print(f"new loads = {new_loads}")
        candidates.append(new_candidate)
    return candidates


def most_promising_new_instructions(
    value: int, candidate: Candidate, max_children: int = 2
):

    pass


def generate_new_branches(candidate: Candidate, max_children: int = 2):
    new_candidates = []
    pos = candidate.get_missing_addresses()[0]
    value = candidate.objective[pos].idx
    new_inst = most_promising_new_instructions(value, candidate, max_children)
    for idx in range(len(new_inst[1:])):
        copy_candidate = candidate
        copy_candidate.add_instruction(new_inst[idx])
    candidate.add_instruction(new_inst[0])
    return new_candidates


debug = True


def explore_space(candidates: List[Candidate], max_children: int = 2):
    all_candidates_completed = False
    # complete
    while not all_candidates_completed:
        completed = 0
        for candidate in candidates:
            if candidate.is_complete():
                completed += 1
                continue
            candidates.append(generate_new_branches(candidate, max_children))
        if not debug:
            all_candidates_completed = completed == len(candidates)
        else:
            all_candidates_completed = True
    # then sort
    return candidates


memory_addresses = [
    Mem("p", 10),
    Mem("p", 9),
    Mem("p", 1),
    Mem("p", 0),
]

objective = Reg(memory_addresses)
candidates = generate_seed_candidates(objective)
# candidates = explore_space(candidates)

print(f"Best options for memory addresses = {objective}")
i = 0
for candidate in candidates:
    print(f"\tCandidate #{i}")
    print(f"\t{candidate}")
    i += 1
