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
import time
import itertools as it
from z3 import sat, unsat
from sat.x86_sat.evaluate import check, Call, Var
from code_generator import generate_code, generate_micro_benchmark, Candidate, PackingT
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

"""
Terminology:
* case: abstraction for describing the contiguity of elements to pack onto a
  vector register
* candidate: set of instructions combined which produce a vector register

"""

MIN_CANDIDATES = 5
FOUND_SOLUTIONS = 0
N_CHECKS = 0


def _check_candidate(
    new_product: Call, instructions: List[Call], objective: Call
) -> Any:
    global N_CHECKS
    if N_CHECKS % 100 == 0 and N_CHECKS > 0:
        print(f"[DEBUG] N_CHECKS = {N_CHECKS}")
    N_CHECKS += 1
    result, model = check(objective == new_product)
    dprint(objective == new_product)
    if result == sat:
        print("[DEBUG] Candidate: ", objective == new_product, model)
        return Candidate(instructions + [new_product], model)
    return unsat


def _generate_new_candidate(
    case: int, _new_ins: Call, instructions: List[Call], packing: PackingT, objective
) -> Tuple[Any, Any]:
    """Core function for generating the candidates."""
    __len_inst = len(instructions) + 1
    __instype = _new_ins.instype
    __argstype = _new_ins.argstype
    __hasimm = _new_ins.hasimm
    __hasmaskvec = _new_ins.maskvec
    _unsat_candidates = []
    if __instype == InsType.LOAD:
        args = [packing[case]]
        if __hasmaskvec:
            args += [Var("mask_" + ("i" * case), "__m128i")]
        if _new_ins.needsregister:
            for c in instructions:
                new_args = [c] + args
                if (__len_inst >= packing.min_instructions) and (
                    (
                        sol := _check_candidate(
                            _new_ins(*new_args), instructions, objective
                        )
                    )
                    is not unsat
                ):
                    return sat, sol
                else:
                    _unsat_candidates.append(_new_ins(*new_args))
            return unsat, _unsat_candidates

        if __len_inst >= packing.min_instructions and (
            (sol := _check_candidate(_new_ins(*args), instructions, objective))
            is not unsat
        ):
            return sat, sol
        return unsat, instructions + [_new_ins(*args)]

    _i = Var("i" * case, "int")
    imm = []

    if __instype in [InsType.BLEND, InsType.INSERT] and __hasimm:
        imm = [_i]
    if __argstype == ArgsType.ALLREG:
        if len(instructions) < 2:
            return unsat, _unsat_candidates
        for c in it.combinations(instructions, 2):
            args = list(c) + imm
            if (__len_inst >= packing.min_instructions) and (
                (sol := _check_candidate(_new_ins(*args), instructions, objective))
                is not unsat
            ):
                return sat, sol
            else:
                _unsat_candidates.append(_new_ins(*args))
    elif __argstype == ArgsType.REG_MEM:
        for output in instructions:
            # Heuristic: attempt blends with 3 different offsets.
            if __instype == InsType.BLEND:
                for offset in [0, -1, 1]:
                    args = [output, packing[case] + offset] + imm
                    if (__len_inst >= packing.min_instructions) and (
                        (
                            sol := _check_candidate(
                                _new_ins(*args), instructions, objective
                            )
                        )
                        is not unsat
                    ):
                        return sat, sol
                args = [output, packing[case] + offset] + imm
                _unsat_candidates.append(_new_ins(*args))
            else:
                args = [output, packing[case]] + imm
                if (__len_inst >= packing.min_instructions) and (
                    (sol := _check_candidate(_new_ins(*args), instructions, objective))
                    is not unsat
                ):
                    return sat, sol
                else:
                    _unsat_candidates.append(_new_ins(*args))
    else:
        raise ArgsTypeException
    return unsat, _unsat_candidates


def get_instructions_list(case: int, instructions: List[Call], packing: PackingT):
    l = []
    if case >= len(packing):
        return move_instruction_list
    n_inserts = 0
    n_blends = 0
    for ins in instructions:
        if "insert" in ins.fn.name:
            n_inserts += 1
        if "blend" in ins.fn.name:
            n_blends += 1
    for ins in full_instruction_list:
        if ins.instype == InsType.BLEND and (
            n_blends > 1 or sum(packing.contiguity) == 0
        ):
            continue
        if ins.instype == InsType.INSERT and n_inserts >= packing.nnz - 1:
            continue
        l.append(ins)
    return l


def _generate_new_candidates_forest(
    case: int, instructions: List[Call], packing: PackingT, objective: Call, n_ins: int,
) -> List[Candidate]:
    # Heuristic: if we have already consumed ALL instructions using memory
    # operands, then we can limit the search space now
    _candidates = get_instructions_list(case, instructions, packing)
    _new_candidates = []
    _pending = []
    len_new_list = len(instructions) + 1
    for ins in _candidates:
        if ins.width > packing.c_max_width:
            continue
        res, sol = _generate_new_candidate(case, ins, instructions, packing, objective)
        if res == sat:
            _new_candidates.append(sol)
            if len(_new_candidates) + FOUND_SOLUTIONS >= MIN_CANDIDATES:
                return _new_candidates
        else:
            if len(sol) > 0:
                _pending += sol

    # I guess this is **not** tail-recursion by definition,
    for p in _pending:
        if len_new_list > n_ins:
            continue
        _sub_new_candidates = _generate_new_candidates_forest(
            case + 1, instructions + [p], packing, objective, n_ins
        )
        if len(_sub_new_candidates) != 0:
            _new_candidates += _sub_new_candidates
            if len(_new_candidates) + FOUND_SOLUTIONS >= MIN_CANDIDATES:
                return _new_candidates

    if len(_new_candidates) > 0:
        for i in range(len(_new_candidates)):
            assert isinstance(_new_candidates[i], Candidate)
    return _new_candidates


def find_all_candidates(
    n_ins: int, packing: PackingT, objective: Call,
) -> List[Candidate]:
    global FOUND_SOLUTIONS
    global MIN_CANDIDATES
    __candidates = []
    # Conceptually, this generates |load_instructions| roots. From them, we are
    # going to create non-binary trees, and we would want to traverse them in
    # level or natural order. This way we can easily decide whether to
    # visit/generate next level, as it is a stop condition in our approach.
    # for load in tqdm(load_instructions, desc="Load ins"):
    for load in load_instructions:
        args = [packing[-1]]
        __width = load.width
        # It does not make sense using 256 bits instructions when packing only
        # 4 elements
        if __width > packing.c_max_width:
            continue
        if load.maskvec:
            args += [Var("mask_i", f"__m{__width}i")]
        if load.needsregister:
            args = [Var("aux", f"__m{__width}")] + args
        _new_inst = load(*args)
        # Check if root satisfies condition
        _val = _check_candidate(_new_inst, [], objective)
        if _val != unsat:
            _new_candidates = [_val]
        else:
            if n_ins == 1:
                continue
            _new_candidates = _generate_new_candidates_forest(
                1, [_new_inst], packing, objective, n_ins
            )
        if len(_new_candidates) + FOUND_SOLUTIONS >= MIN_CANDIDATES:
            __candidates += _new_candidates
            break
        __candidates += _new_candidates
    return __candidates


def exploration_space(packing: PackingT):
    MAX_INS = len(packing) + int(len(packing) / 4)
    print("*" * 80)
    print(
        f"* Searching packing combinations for: {packing} (max. instructions {MAX_INS}, minimum candidates {MIN_CANDIDATES})"
    )
    print("*" * 80)

    objective = globals()[f"set_{len(packing)}_float_elements"](*packing)
    n_candidates = 0
    C = []
    t0 = time.time_ns()
    for n_ins in range(1, MAX_INS):
        if n_ins < packing.min_instructions - 1:
            continue
        global FOUND_SOLUTIONS
        FOUND_SOLUTIONS = len(C)
        print(f"** Using max. {n_ins:3} instruction(s)")
        _C = find_all_candidates(n_ins, packing, objective)
        n_candidates += len(_C)
        if len(_C) > 0:
            C += [*_C]
        else:
            print(f"\tNo candidates using {n_ins} instruction(s)")
        if n_candidates >= MIN_CANDIDATES or n_ins + 1 > packing.max_instructions:
            print(f"*** SEARCH FINISHED WITH {n_candidates} CANDIDATES FOUND")
            break
    t_elapsed = (time.time_ns() - t0) / 1e9
    print(f"- Time elapsed: {t_elapsed} sec.")

    for i in range(len(C)):
        candidate = C[i]
        candidate.number = i
        candidate.packing = packing
        generate_code(candidate)
        generate_micro_benchmark(candidate)
    return C


def generate_all_cases(max_size: int = 8) -> List[PackingT]:
    list_cases = []
    for i in range(1, max_size + 1):
        print(f"Generating cases with {i} elements")
        vector_size = 4 if i < 5 else 8
        for c in range(2 ** (i - 1)):
            values = [0] * vector_size
            values[0] = 10
            contiguity = (
                list(map(lambda x: int(x), list(f"{c:0{i-1}b}"))) if i > 1 else []
            )
            for v in range(1, len(contiguity) + 1):
                offset = 1 if contiguity[v - 1] == 1 else 10
                values[v] = values[v - 1] + offset
            values.reverse()
            packing = PackingT(values, contiguity)
            list_cases.append(packing)
    return list_cases


debug = False

if not debug:
    all_cases = generate_all_cases(2)
    for case in all_cases:
        exploration_space(case)
        FOUND_SOLUTIONS = 0
else:
    case = PackingT([7, 5, 3, 0], [0, 0, 0])
    c = exploration_space(case)
