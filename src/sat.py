from typing import List
from z3 import sat
from sat.x86_sat.evaluate import check, Var, Call
from instructions import (
    load_instructions,
    _mv_insert_mem_ps,
    insert_blend_instructions,
    # move_instructions,
    set_4_float_elements,
    set_8_float_elements,
)
from tqdm import tqdm

MIN_CANDIDATES = 2

DATA_TYPE = {
    "float": 4,
    "double": 8,
}


class Solution:
    def __len__(self):
        return len(self.instructions)

    def __init__(self, instructions=None, model=None):
        self.instructions = [] if instructions is None else instructions
        self.model = model


def max_width(dtype: str = "float", nelems: int = 4):
    return int(DATA_TYPE[dtype] * 8 * nelems)


def print_instruction(signature, output, *args):
    instruction = f"{signature}("
    if not signature.startswith("_mv"):
        instruction = f"{output} = {instruction}"
    else:
        instruction += f"{output},"
    for arg in args:
        instruction += f"{arg},"
    instruction = f"{instruction[:-1]});"
    print(f"[BACKEND]\t{instruction}")
    return instruction


def get_register(args, arg, tmp_reg, solution):
    for tmp in tmp_reg:
        res, m = check(arg == tmp_reg[tmp])
        # print(arg == tmp_reg[tmp], res)
        model_ok = True
        for i in m:
            model_ok = model_ok and m[i] == solution.model[i]
        if sat == res and model_ok:
            args += [tmp]
            break
    return args


def get_arguments(ins, tmp_reg, solution):
    args = []
    for arg in ins.args:
        if type(arg) == Var:
            args += [solution.model[arg.name]]
            continue
        if type(arg) != Call:
            args += [f"p[{arg}]"]
            continue
        n_args = len(args)
        args = get_register(args, arg, tmp_reg, solution)
        assert n_args + 1 == len(args)
    return args


def generate_code(solution: Solution):
    reg_no = 0
    tmp_reg = {}
    liveness = {}
    instructions = solution.instructions
    c_instructions = []
    for n in range(len(instructions)):
        ins = instructions[n]
        args = get_arguments(ins, tmp_reg, solution)
        output = f"r{reg_no}"
        tmp_reg[output] = ins
        reg_no += 1
        liveness[reg_no] = [n, n]
        c_instructions.append(print_instruction(ins.fn.name, output, *args))
    print(liveness)


def get_all_combinations(
    n_ins: int, packing, objective, dtype="float"
) -> List[Solution]:
    solutions = []

    def new_solutions(case: int, instructions: List[Call], packing: List) -> Solution:
        result, model = check(objective == instructions[-1])
        if result == sat:
            print("ALRIGHT: ", objective == instructions[-1], model)
            for i in instructions:
                print("\t", i)
            return Solution(instructions, model)
        else:
            if case == n_ins:
                return Solution()
            # keep exploring
            new_insert = _mv_insert_mem_ps(
                instructions[-1], packing[0], Var("i" * case, "int")
            )
            instructions.append(new_insert)
            return new_solutions(case + 1, instructions, packing[1:])

    for load in load_instructions:
        new_inst = load(packing[0])
        if new_inst.width > max_width(dtype, len(packing)):
            continue
        new_sol = new_solutions(1, [new_inst], packing[1:])
        if len(new_sol) == 0:
            print(f"\tNo solutions using {n_ins} instructions with seed {new_inst}")
        else:
            solutions.append(new_sol)
    return solutions


def main(packing: List):
    vpacking = globals()[f"set_{len(packing)}_float_elements"](*packing)
    packing.reverse()
    MAX_INS = 2 * len(packing)
    solutions_found = 0
    print(f"Searching packing combinations for: {packing}")
    S = []
    for n_ins in range(1, MAX_INS):
        print(f"- using {n_ins:3} instruction(s)")
        R = get_all_combinations(n_ins, packing, vpacking)
        solutions_found += len(R)
        if len(R) > 0:
            S += [R]
        if len(R) == 0:
            print(f"\tNo solutions using {n_ins} instruction(s)")
        if solutions_found >= MIN_CANDIDATES:
            print(f"** SEARCH FINISHED WITH {solutions_found} CANDIDATES")
            break

    for s in S:
        for c in s:
            generate_code(c)
    return S


S = main([10, 9, 1, 0])
