from re import I
from typing import List
from z3 import sat
from sat.x86_sat.evaluate import check, Var, Call
from instructions import (
    load_instructions,
    _mv_insert_mem_ps,
    insert_blend_instructions,
    full_instruction_list,
    move_instruction_list,
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


def print_instruction(ins, output, *args):
    instruction = f"{ins.fn.name}("
    if "load" in instruction:
        instruction += "&"
    if not instruction.startswith("_mv"):
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
        c_instructions.append(print_instruction(ins, output, *args))
    print(liveness)


def get_all_combinations(
    n_ins: int,
    packing: List[int],
    objective: Call,
    dtype: str = "float",
    min_combinations: int = 5,
) -> List[Solution]:
    solutions = []

    def _check_new_solution(
        new_ins: Call, instructions: List[Call], packing: List[int]
    ):
        result, model = check(objective == instructions[-1])
        if result == sat:
            print("ALRIGHT: ", objective == instructions[-1], model)
            for i in instructions:
                print("\t", i)
            return Solution(instructions, model)
        return None

    def _generate_new_solutions_graph(
        case: int, instructions: List[Call], packing: List
    ) -> List[Solution]:
        # Heuristic: if we have already consumed ALL instructions using memory
        # operands, then we can limit the search space now
        _candidates = (
            full_instruction_list if len(packing) > 0 else move_instruction_list
        )
        _new_solutions = []
        _pending = []
        for ins in _candidates:
            if (sol := _check_new_solution(ins, instructions, packing)) is not None:
                _new_solutions.append(sol)
                if len(_new_solutions) >= min_combinations:
                    return _new_solutions
            else:
                _pending.append(ins)

        # I guess this is **not** tail-recursion by definition,
        for p in _pending:
            _sub_new_solutions = _generate_new_solutions_graph(
                case + 1, instructions + [p], packing[1:]
            )
            if len(_sub_new_solutions) != 0:
                _new_solutions.append(_sub_new_solutions)

        return _new_solutions

    # Conceptually, this generates |load_instructions| roots. From them, we are
    # going to create non-binary trees, and we would want to traverse them in
    # level or natural order. This way we can easily decide whether to
    # visit/generate next level, as it is a stop condition in our approach.
    for load in load_instructions:
        new_inst = load(packing[0])
        if new_inst.width > max_width(dtype, len(packing)):
            continue
        if (sol := _check_new_solution(1, [new_inst])) is not None:
            solutions.append([sol])
            continue
        new_solutions = _generate_new_solutions_graph(1, [new_inst], packing[1:])
        if len(new_solutions) == 0:
            print(f"\tNo solutions using {n_ins} instructions with seed {new_inst}")
        else:
            solutions.append(new_solutions)
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
