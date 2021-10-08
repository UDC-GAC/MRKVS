# Copyright 2021 Marcos Horro
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Any, Tuple
import itertools as it
from z3 import sat, unsat
from sat.x86_sat.evaluate import check, Call, Var
from code_generator import generate_code, Solution, max_width
from instructions import (
    InsType,
    ArgsType,
    ArgsTypeException,
    load_instructions,
    full_instruction_list,
    move_instruction_list,
    set_4_float_elements,
    set_8_float_elements,
)
from utils import dprint
from tqdm import tqdm


MIN_CANDIDATES = 2


def get_all_combinations(
    n_ins: int,
    packing: List[int],
    objective: Call,
    dtype: str = "float",
    min_combinations: int = 5,
) -> List[Solution]:
    solutions = []
    c_max_width = max_width(dtype, len(packing))

    def _check_solution(new_product: Call, instructions: List[Call]) -> Any:
        result, model = check(objective == new_product)
        dprint(objective == new_product)
        if result == sat:
            print("ALRIGHT: ", objective == new_product, model)
            for i in instructions:
                print("\t", i)
            return Solution(instructions + [new_product], model)
        return unsat

    def _generate_new_solution(
        case: int, _new_ins: Call, instructions: List[Call], packing: List[int]
    ) -> Tuple[Any, Any]:
        __instype = _new_ins.instype
        __argstype = _new_ins.argstype
        __hasimm = _new_ins.hasimm
        if __instype == InsType.LOAD:
            if (
                sol := _check_solution(_new_ins(packing[0]), instructions)
            ) is not unsat:
                return sat, sol
            return unsat, instructions + [_new_ins(packing[0])]

        _i = Var("i" * case, "int")
        imm = []
        _unsat_candidates = []
        if __instype in [InsType.BLEND, InsType.INSERT] and __hasimm:
            imm = [_i]
        if __argstype == ArgsType.ALLREG:
            if len(instructions) < 2:
                return unsat, _unsat_candidates
            for c in it.combinations(instructions, 2):
                args = list(c) + imm
                if (sol := _check_solution(_new_ins(*args), instructions)) is not unsat:
                    return sat, sol
                else:
                    _unsat_candidates.append(_new_ins(*args))
        elif __argstype == ArgsType.REG_MEM:
            for output in instructions:
                args = [output, packing[0]] + imm
                if (sol := _check_solution(_new_ins(*args), instructions)) is not unsat:
                    return sat, sol
                else:
                    _unsat_candidates.append(_new_ins(*args))
        else:
            raise ArgsTypeException
        return unsat, _unsat_candidates

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
            if ins.width > c_max_width:
                continue
            res, sol = _generate_new_solution(case, ins, instructions, packing)
            if res == sat:
                _new_solutions.append(sol)
                if len(_new_solutions) >= min_combinations:
                    return _new_solutions
            else:
                if len(sol) > 0:
                    _pending += sol

        # I guess this is **not** tail-recursion by definition,
        for p in _pending:
            if len(instructions + [p]) > n_ins:
                continue
            _sub_new_solutions = _generate_new_solutions_graph(
                case + 1, instructions + [p], packing[1:]
            )
            if len(_sub_new_solutions) != 0:
                _new_solutions += _sub_new_solutions

        if len(_new_solutions) > 0:
            for i in range(len(_new_solutions)):
                assert type(_new_solutions[i]) == Solution
        return _new_solutions

    # Conceptually, this generates |load_instructions| roots. From them, we are
    # going to create non-binary trees, and we would want to traverse them in
    # level or natural order. This way we can easily decide whether to
    # visit/generate next level, as it is a stop condition in our approach.
    for load in tqdm(load_instructions, desc="Load ins"):
        new_inst = load(packing[0])
        if new_inst.width > c_max_width:
            continue
        res, sol = _generate_new_solution(0, load, [], packing)
        if res == sat:
            solutions.append(sol)
            continue
        new_solutions = _generate_new_solutions_graph(1, [new_inst], packing[1:])
        if len(new_solutions) >= MIN_CANDIDATES:
            solutions += new_solutions
            break
        if len(new_solutions) == 0:
            print(f"\tNo solutions using {n_ins} instructions with seed {new_inst}")
        else:
            solutions += new_solutions
    return solutions


def main(packing: List):
    MAX_INS = len(packing) + int(len(packing) / 4)
    print(
        f"Searching packing combinations for: {packing} (max. instructions {MAX_INS}, minimum candidates {MIN_CANDIDATES})"
    )
    if len(packing) == 8:
        vpacking = set_8_float_elements(*packing)
    else:
        vpacking = set_4_float_elements(*packing)
    packing.reverse()
    n_candidates = 0
    S = []
    import time

    t0 = time.time_ns()
    for n_ins in range(1, MAX_INS):
        print(f"- using {n_ins:3} instruction(s)")
        C = get_all_combinations(n_ins, packing, vpacking)
        n_candidates += len(C)
        if len(C) > 0:
            S += [*C]
        if len(C) == 0:
            print(f"\tNo solutions using {n_ins} instruction(s)")
        if n_candidates >= MIN_CANDIDATES:
            print(f"** SEARCH FINISHED WITH {n_candidates} CANDIDATES")
            break
    t_elapsed = (time.time_ns() - t0) / 1e9
    print(f"Time elapsed: {t_elapsed} sec.")

    print(S)

    for solution in S:
        generate_code(solution)
    return S


# TODO:
S = main([10, 9, 8, 7, 3, 2, 1, 0])
