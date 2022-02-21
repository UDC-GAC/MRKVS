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
from instructions_double import (
    InsType,
    ArgsType,
    ArgsTypeException,
    load_instructions,
    full_instruction_list,
    move_instruction_list,
#    set_4_float_elements,
    set_2_double_elements,
#    set_8_float_elements,
    set_4_double_elements,
#    set_hi_lo,
    set_hi_lo_double,
)
from utils import dprint

"""
Terminology:
* case: abstraction for describing the contiguity of elements to pack onto a
  vector register
* candidate: set of instructions combined which produce a vector register

"""

MIN_CANDIDATES = 3
FOUND_SOLUTIONS = 0
N_CHECKS = 1


class MinSolFound(Exception):
    pass


AUX_CONDITION = Var(f"aux", f"__m128")


def _check_candidate(
    new_product: Call, instructions: List[Call], objective: Call
) -> Any:
    global N_CHECKS
    if N_CHECKS % 100 == 0:
        print(f"[DEBUG] N_CHECKS = {N_CHECKS}")
    N_CHECKS += 1
    result, model = check(objective == new_product)
    dprint(objective == new_product)
    if result == sat:
        # if "aux" in model and int(model["aux"], base=16) != 0x0:
        #     print(model["aux"])
        #     return unsat
        print("[DEBUG] Candidate: ", objective == new_product, model)
        return Candidate(instructions + [new_product], model)
    return unsat


def _gen_new_candidate(
    case: int,
    _new_ins: Call,
    instructions: List[Call],
    packing: PackingT,
    objective,
    var_name: str = "i",
) -> Tuple[Any, Any]:
    """Core function for generating the candidates."""
    __len_inst = len(instructions) + 1
    __instype = _new_ins.instype
    __argstype = _new_ins.argstype
    __hasimm = _new_ins.hasimm
    __hasmaskvec = _new_ins.maskvec
    _unsat_candidates = []
    if __instype == InsType.LOAD:
        args = [packing[-1 - case]]
        if __hasmaskvec:
            args += [Var("mask_" + (var_name * (case + 1)), "__m128i")]
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

    _i = Var(var_name * (case + 1), "int")
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
        for n_case in range(case, len(packing)):
            for output in instructions:
                # Heuristic: attempt blends with 3 different offsets.
                if __instype == InsType.BLEND:
                    for offset in [0, -1, 1]:
                        args = [output, packing[-1 - n_case] + offset] + imm
                        if (__len_inst >= packing.min_instructions) and (
                            (
                                sol := _check_candidate(
                                    _new_ins(*args), instructions, objective
                                )
                            )
                            is not unsat
                        ):
                            return sat, sol
                    args = [output, packing[-1 - n_case] + offset] + imm
                    _unsat_candidates.append(_new_ins(*args))
                else:
                    args = [output, packing[-1 - n_case]] + imm
                    if (__len_inst >= packing.min_instructions) and (
                        (
                            sol := _check_candidate(
                                _new_ins(*args), instructions, objective
                            )
                        )
                        is not unsat
                    ):
                        return sat, sol
                    else:
                        _unsat_candidates.append(_new_ins(*args))
    else:
        raise ArgsTypeException
    return unsat, _unsat_candidates


def _prune_ld_ins(packing: PackingT, case: int = 1) -> list:
    new_list = []
    for load in load_instructions:
        __width = load.width
        # It does not make sense using 256 bits instructions when packing only
        # 4 elements
        if __width > packing.c_max_width:
            continue
        if sum(packing.contiguity) == 0:
            if not load.name.endswith("sd"):
                continue
        else:
            if (len(packing.contiguity) >= 1 and packing.contiguity[-1] == 0) or (
                len(packing.contiguity) >= 5 and packing.contiguity[-5] == 0
            ):
                if "loadl" in load.name:
                    continue
            if (len(packing.contiguity) >= 3 and packing.contiguity[-3] == 0) or (
                len(packing.contiguity) == 7 and packing.contiguity[-7] == 0
            ):
                if "loadh" in load.name:
                    continue
        new_list.append(load)
    if case == 0:
        for ins in full_instruction_list:
            __width = ins.width
            if __width > packing.c_max_width or ins.instype != InsType.BLEND:
                continue
            if ins.instype == InsType.BLEND and ins.argstype != ArgsType.REG_MEM:
                continue
            new_list.append(ins)
    assert len(new_list) > 0
    return new_list


def _prune_ins_list(case: int, instructions: List[Call], packing: PackingT):
    l = []
    if case >= len(packing):
        return move_instruction_list
    n_inserts = 0
    n_blends = 0
    n_loads = 0
    for ins in instructions:
        if "load" in ins.fn.name:
            n_loads += 1
        if "insert" in ins.fn.name:
            n_inserts += 1
        if "blend" in ins.fn.name:
            n_blends += 1
    for ins in full_instruction_list:
        if ins.instype == InsType.LOAD:
            continue
        if sum(packing.contiguity) == 0:
            if ins.instype == InsType.BLEND and (n_blends >= 1):
                continue
            if (
                ins.instype == InsType.LOAD
                and "_ps" in ins.name
                and not "mask" in ins.name
            ):
                continue
        if ins.instype == InsType.INSERT and n_inserts >= packing.nnz - 1:
            continue
        if ins.instype == InsType.BLEND and (
            (n_blends > 1) or len(instructions) >= packing.nnz - 1
        ):
            continue
        l.append(ins)
    l.extend(_prune_ld_ins(packing))
    return l


def _gen_forest_deep_first(
    case: int,
    instructions: List[Call],
    packing: PackingT,
    objective: Call,
    n_ins: int,
    var_name: str = "i",
) -> List[Candidate]:
    # Heuristic: if we have already consumed ALL instructions using memory
    # operands, then we can limit the search space now
    _candidates = _prune_ins_list(case, instructions, packing)
    # _new_candidates = []
    _pending = []
    len_new_list = len(instructions) + 1
    for ins in _candidates:
        if ins.width > packing.c_max_width:
            continue
        res, sol = _gen_new_candidate(
            case, ins, instructions, packing, objective, var_name
        )
        if res == sat:
            # _new_candidates.append(sol)
            return [sol]
        else:
            if len(sol) > 0:
                _pending += sol

    # I guess this is **not** tail-recursion by definition,
    for p in _pending:
        if len_new_list >= n_ins:
            continue
        _sub_new_candidates = _gen_forest_deep_first(
            case + 1, instructions + [p], packing, objective, n_ins, var_name
        )
        if _sub_new_candidates == None:
            continue
        if len(_sub_new_candidates) != 0:
            # _new_candidates += _sub_new_candidates
            # if len(_new_candidates) + FOUND_SOLUTIONS >= MIN_CANDIDATES:
            return _sub_new_candidates
    return None
    # if len(_new_candidates) > 0:
    #     for i in range(len(_new_candidates)):
    #         assert isinstance(_new_candidates[i], Candidate)
    # return _new_candidates


def search_deep_first(
    n_ins: int, packing: PackingT, objective: Call, var_name: str = "i"
) -> List[Candidate]:
    __candidates = []
    # Conceptually, this generates |load_instructions| roots. From them, we are
    # going to create non-binary trees, and we would want to traverse them in
    # level or natural order. This way we can easily decide whether to
    # visit/generate next level, as it is a stop condition in our approach.
    # for load in tqdm(load_instructions, desc="Load ins"):
    pruned_load_instructions = _prune_ld_ins(packing, 0)
    for load in pruned_load_instructions:
        args = [packing[-1]]
        __width = load.width
        if load.maskvec:
            args += [Var(f"{var_name}_mask_i", f"__m{__width}i")]
        if load.needsregister or load.instype == InsType.BLEND:
#            args = [Var(f"aux", f"__m{__width}")] + args
            args = [Var(f"aux", load.params[0].type)] + args # Is it always param 0?
        if load.hasimm:
            args += [Var(var_name, "int")]
        _new_inst = load(*args)
        # Check if root satisfies condition
        _val = _check_candidate(_new_inst, [], objective)
        if _val != unsat:
            _new_candidates = [_val]
        else:
            if n_ins == 1:
                continue
            _new_candidates = _gen_forest_deep_first(
                1, [_new_inst], packing, objective, n_ins, var_name
            )
        if _new_candidates == None:
            continue
        __candidates += [i for i in _new_candidates if i not in __candidates]
        if len(__candidates) >= MIN_CANDIDATES:
            break
    return __candidates


def _gen_forest_breadth_first(
    case: int,
    breadth_first_list: List,
    # instructions: List[Call],
    packing: PackingT,
    objective: Call,
    max_ins: int,
    var_name: str = "i",
) -> List[Candidate]:
    # Heuristic: if we have already consumed ALL instructions using memory
    # operands, then we can limit the search space now

    _pending = []
    _solutions = []
    for instructions in breadth_first_list:
        _candidates = _prune_ins_list(case, instructions, packing)
        for ins in _candidates:
            if ins.width > packing.c_max_width:
                continue
            res, sol = _gen_new_candidate(case, ins, instructions, packing, objective)
            if res == sat:
                _solutions.append(sol)
            else:
                if len(sol) > 0:
                    tmp = instructions + sol
                    _pending.append(tmp)

    if len(_solutions) != 0:
        return _solutions

    if len(breadth_first_list[0]) == max_ins:
        return None

    _sub_new_candidates = _gen_forest_breadth_first(
        case + 1, _pending, packing, objective, max_ins
    )

    if _sub_new_candidates != None and len(_sub_new_candidates) != 0:
        return _sub_new_candidates
    return None


def search_breadth_first(
    max_ins: int, packing: PackingT, objective: Call, var_name: str = "i"
) -> List[Candidate]:
    __candidates = []
    # Conceptually, this generates |load_instructions| roots. From them, we are
    # going to create non-binary trees, and we would want to traverse them in
    # level or natural order. This way we can easily decide whether to
    # visit/generate next level, as it is a stop condition in our approach.
    # for load in tqdm(load_instructions, desc="Load ins"):
    pruned_load_instructions = _prune_ld_ins(packing, 0)
    breadth_first_list = []
    for load in pruned_load_instructions:
        args = [packing[-1]]
        __width = load.width
        if load.maskvec:
            args += [Var(f"{var_name}_mask_i", f"__m{__width}i")]
        if load.needsregister or load.instype == InsType.BLEND:
            args = [Var("aux", f"__m{__width}")] + args
        if load.hasimm:
            args += [Var(var_name, "int")]
        _new_inst = load(*args)
        # Check if root satisfies condition
        _val = _check_candidate(_new_inst, [], objective)
        if _val != unsat:
            __candidates += [_val]
        else:
            breadth_first_list.append([_new_inst])

    if len(__candidates) >= MIN_CANDIDATES:
        return __candidates

    if (
        __new_candidates := _gen_forest_breadth_first(
            1, breadth_first_list, packing, objective, max_ins
        )
    ) != None:
        __candidates.extend(__new_candidates)

    return __candidates


def exploration_space(packing: PackingT, var_name: str = "i") -> List:
    MAX_INS = len(packing) + int(len(packing) / 4)

    objective = globals()[f"set_{len(packing)}_{packing.dtype}_elements"](*packing)
    n_candidates = 0
    full_candidates_list = []
    t0 = time.time_ns()
    for n_ins in range(1, MAX_INS + 1):
        if n_ins < packing.min_instructions:
            print(
                f"** Skipping {n_ins:3d} instruction(s), min {packing.min_instructions}"
            )
            continue
        global FOUND_SOLUTIONS
        FOUND_SOLUTIONS = len(full_candidates_list)
        print(f"** Using max. {n_ins:3d} instruction(s)")
        _candidates = search_deep_first(n_ins, packing, objective, var_name)
        # _candidates = search_breadth_first(n_ins, packing, objective, var_name)
        n_candidates += len(_candidates)
        if len(_candidates) > 0:
            full_candidates_list += [*_candidates]
        else:
            print(f"\tNo candidates using {n_ins} instruction(s)")
        if n_candidates >= MIN_CANDIDATES or n_ins + 1 >= packing.max_instructions:
            print(f"*** SEARCH FINISHED WITH {n_candidates} CANDIDATES FOUND")
            break
    t_elapsed = (time.time_ns() - t0) / 1e9
    print(f"- Time elapsed: {t_elapsed} sec.")
    return full_candidates_list


def generate_packing(c: int, i: int, val: int = 16, dtype: str = "float") -> PackingT:
#    vector_size = 4 if i < 5 else 8
    vector_size = 2 if i < 3 else 4
    values = [0] * vector_size
    values[0] = val
    contiguity = list(map(lambda x: int(x), list(f"{c:0{i-1}b}"))) if i > 1 else []
    for v in range(1, len(contiguity) + 1):
        offset = 1 if contiguity[v - 1] == 1 else 10
        values[v] = values[v - 1] + offset
    values.reverse()
    contiguity.reverse()
    return PackingT(values, contiguity, dtype)


def generate_all_packing(min_size: int = 1, max_size: int = 8, dtype: str = "float") -> List[PackingT]:
    list_cases = []
    for i in range(min_size, max_size + 1):
        print(f"Generating cases with {i} elements")
        new_packing = []
        for c in range(2 ** (i - 1)):
            new_packing.append(generate_packing(c, i, dtype=dtype))
        new_packing.reverse()
        list_cases.extend(new_packing)
    return list_cases


def get_sub_packing_from_values(values):
    contiguity = []
    copy_values = list(reversed(values))
    nnz = len(copy_values) - copy_values.count(0)
    for i in range(1, nnz):
        contiguity.insert(0, int(copy_values[i] - 1 == copy_values[i - 1]))

    return PackingT(values, contiguity)


def synthesize_code(packing, full_candidates_list):
    for i in range(len(full_candidates_list)):
        candidate = full_candidates_list[i]
        candidate.number = i
        candidate.packing = packing
        generate_code(candidate)
        generate_micro_benchmark(candidate, dtype=candidate.packing.dtype)


debug = False

if __name__ == "__main__":
    import sys

    start = 1
    end = 4
    dtype = "double"
    if len(sys.argv) == 3:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    if not debug:
        all_packings = generate_all_packing(start, end, dtype=dtype)
        for packing in all_packings:
            print("*" * 80)
            print(
                f"* Searching packing combinations for: {packing} (find {MIN_CANDIDATES} candidates at least)"
            )
            print("*" * 80)
            if packing.nnz > 4:
                high_half = get_sub_packing_from_values(packing.packing[:4])
                lower_half = get_sub_packing_from_values(packing.packing[4:])
                print(f"* Dividing into two halves: {high_half.nnz} + {lower_half.nnz}")
                hi_candidates_list = exploration_space(high_half, "hi", dtype)
                lo_candidates_list = exploration_space(lower_half, "lo", dtype)
                candidates_list = []
                for hi, lo in it.product(hi_candidates_list, lo_candidates_list):
                    instructions = lo.instructions + hi.instructions
                    merge_ins = set_hi_lo(hi.instructions[-1], lo.instructions[-1])
                    instructions.append(merge_ins)
                    new_candidate = Candidate(instructions, lo.model | hi.model)
                    candidates_list.append(new_candidate)
            else:
                candidates_list = exploration_space(packing)
            synthesize_code(packing, candidates_list)
            FOUND_SOLUTIONS = 0
            N_CHECKS = 1
    else:
        case = PackingT([17, 16, 15, 10], [0, 1, 1])
        c = exploration_space(case)
