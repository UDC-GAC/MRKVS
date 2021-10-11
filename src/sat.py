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

import multiprocessing
from typing import List, Any, Tuple
from multiprocessing import Pool, TimeoutError
import time
import itertools as it
from z3 import sat, unsat, CheckSatResult
from sat.x86_sat.evaluate import check, Call, Var
from code_generator import generate_code, Candidate, max_width
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
from itertools import repeat
from utils import dprint
from tqdm import tqdm

"""
Terminology:
* case: abstraction for describing the contiguity of elements to pack onto a
  vector register
* candidate: set of instructions combined which produce a vector register

"""

MIN_CANDIDATES = 5
FOUND_SOLUTIONS = 0

manager = multiprocessing.Manager()
_NEW_CANDIDATES_L = manager.list()
_NEW_PENDING_L = manager.list()

_CANDIDATES = None
_INSTRUCTIONS = None
_OBJECTIVE = None
_NEW_CANDIDATE = None
_MLOCK = None


class PackingT:
    def __getitem__(self, idx):
        return self.packing[idx]

    def __len__(self):
        return self.vector_size

    def __init__(
        self,
        packing: List[int],
        contiguity: List[int] = [1, 1, 1],
        dtype: str = "float",
    ):
        self.packing = packing
        self.nnz = len(packing) - packing.count(-1)
        self.vector_size = len(self.packing)
        self.contiguity = contiguity
        self.min_instructions = self.contiguity.count(0)
        self.dtype = dtype
        self.c_max_width = max_width(dtype, len(packing))
        assert len(contiguity) + 1 == self.nnz


def ad_hoc_sat_hash(self):
    if self == sat:
        return 131
    return 12412344323


CheckSatResult.__hash__ = ad_hoc_sat_hash

from typing import Dict, Any
import hashlib
import json


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


class CacheResults:
    def __hash__(self):
        return hash((self.res, dict_hash(self.model)))

    def __eq__(self, other):
        satisfiability = self.res == other.res
        same_model = self.model == other.model
        return satisfiability and same_model

    def __init__(self, res, model):
        self.res = res
        self.model = model


def memoize(f):
    memo = {}

    def helper(new_product: Call, instructions: List[Call], objective: Call):
        x = CacheResults(*check(objective == new_product))
        if x not in memo:
            memo[x] = f(new_product, instructions, objective)
        return memo[x]

    return helper


N_CHECKS = 0

# @memoize
def _check_candidate(
    new_product: Call, instructions: List[Call], objective: Call
) -> Any:
    global N_CHECKS
    if N_CHECKS % 100 == 0:
        print(f"N_CHECKS = {N_CHECKS}")
    N_CHECKS += 1
    result, model = check(objective == new_product)
    dprint(objective == new_product)
    if result == sat:
        print("ALRIGHT: ", objective == new_product, model)
        return Candidate(instructions + [new_product], model)
    return unsat


def _generate_new_candidate(
    case: int, _new_ins: Call, instructions: List[Call], packing: List[int], objective
) -> Tuple[Any, Any]:
    """Core function for generating the candidates."""
    __instype = _new_ins.instype
    __argstype = _new_ins.argstype
    __hasimm = _new_ins.hasimm
    if __instype == InsType.LOAD:
        if (
            sol := _check_candidate(_new_ins(packing[case]), instructions, objective)
        ) is not unsat:
            return sat, sol
        return unsat, instructions + [_new_ins(packing[case])]

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
            if (
                sol := _check_candidate(_new_ins(*args), instructions, objective)
            ) is not unsat:
                return sat, sol
            else:
                _unsat_candidates.append(_new_ins(*args))
    elif __argstype == ArgsType.REG_MEM:
        for output in instructions:
            # Heuristic: attempt blends with 3 different offsets.
            if __instype == InsType.BLEND:
                for offset in [0, -1, 1]:
                    args = [output, packing[case] + offset] + imm
                    if (
                        sol := _check_candidate(
                            _new_ins(*args), instructions, objective
                        )
                    ) is not unsat:
                        return sat, sol
                args = [output, packing[case] + offset] + imm
                _unsat_candidates.append(_new_ins(*args))
            else:
                args = [output, packing[case]] + imm
                if (
                    sol := _check_candidate(_new_ins(*args), instructions, objective)
                ) is not unsat:
                    return sat, sol
                else:
                    _unsat_candidates.append(_new_ins(*args))
    else:
        raise ArgsTypeException
    return unsat, _unsat_candidates


def __multi_process(
    # _candidates: List[Call],
    # instructions: List[Call],
    n_candidate: int,
    packing: PackingT,
    # objective: Call,
    case: int,
) -> Tuple[bool, List]:
    global _OBJECTIVE
    global _INSTRUCTIONS
    global _CANDIDATES
    ins = _CANDIDATES[n_candidate]
    if ins.width > packing.c_max_width:
        return False
    res, sol = _generate_new_candidate(case, ins, _INSTRUCTIONS, packing, _OBJECTIVE)
    if res == sat:
        _NEW_CANDIDATES_L.append(sol)
        return True
    _NEW_PENDING_L.append(sol)
    return False


MULTIPROCESS = True


def __multiprocess_wrapper(
    case, _candidates, instructions, objective, packing, min_combinations
):
    global _CANDIDATES
    global _INSTRUCTIONS
    global _OBJECTIVE
    _CANDIDATES = _candidates
    _INSTRUCTIONS = instructions
    _OBJECTIVE = objective
    with Pool() as pool:
        pool.starmap(
            __multi_process,
            zip(
                range(len(_CANDIDATES)),
                repeat(packing),
                repeat(case),
            ),
        )
    print(_NEW_CANDIDATES_L, _NEW_PENDING_L)
    return _NEW_CANDIDATES_L, _NEW_PENDING_L


def _generate_new_candidates_forest(
    case: int,
    instructions: List[Call],
    packing: PackingT,
    objective: Call,
    min_combinations: int,
    n_ins: int,
) -> List[Candidate]:
    # Heuristic: if we have already consumed ALL instructions using memory
    # operands, then we can limit the search space now
    _candidates = (
        full_instruction_list if case < len(packing) else move_instruction_list
    )
    _new_candidates = []
    _pending = []
    if MULTIPROCESS:
        _new_candidates, _pending = __multiprocess_wrapper(
            case, _candidates, instructions, objective, packing, min_combinations
        )
    else:
        for ins in _candidates:
            if ins.width > packing.c_max_width:
                continue
            res, sol = _generate_new_candidate(
                case, ins, instructions, packing, objective
            )
            if res == sat:
                _new_candidates.append(sol)
                if len(_new_candidates) + FOUND_SOLUTIONS >= min_combinations:
                    return _new_candidates
            else:
                if len(sol) > 0:
                    _pending += sol

    # I guess this is **not** tail-recursion by definition,
    for p in _pending:
        if len(instructions + [p]) > n_ins:
            continue
        _sub_new_candidates = _generate_new_candidates_forest(
            case + 1, instructions + [p], packing, objective, min_combinations, n_ins
        )
        if len(_sub_new_candidates) != 0:
            _new_candidates += _sub_new_candidates
            if len(_new_candidates) + FOUND_SOLUTIONS >= min_combinations:
                return _new_candidates

    if len(_new_candidates) > 0:
        for i in range(len(_new_candidates)):
            assert type(_new_candidates[i]) == Candidate
    return _new_candidates


def find_all_candidates(
    n_ins: int,
    packing: PackingT,
    objective: Call,
    min_combinations: int = 5,
) -> List[Candidate]:
    candidates = []

    # Conceptually, this generates |load_instructions| roots. From them, we are
    # going to create non-binary trees, and we would want to traverse them in
    # level or natural order. This way we can easily decide whether to
    # visit/generate next level, as it is a stop condition in our approach.
    for load in tqdm(load_instructions, desc="Load ins"):
        new_inst = load(packing[-1])
        # It does not make sense using 256 bits instructions when packing only
        # 4 elements
        if new_inst.width > packing.c_max_width:
            continue
        # Check if root satisfies condition
        _val = _check_candidate(new_inst, [], objective)
        if _val != unsat:
            new_candidates = [_val]
        else:
            new_candidates = _generate_new_candidates_forest(
                1, [new_inst], packing, objective, min_combinations, n_ins
            )
        global FOUND_SOLUTIONS
        if len(new_candidates) + FOUND_SOLUTIONS >= MIN_CANDIDATES:
            candidates += new_candidates
            break
        if len(new_candidates) == 0:
            print(f"\tNo candidates using {n_ins} instructions with seed {new_inst}")
        else:
            candidates += new_candidates
    return candidates


def exploration_space(packing: PackingT):
    MAX_INS = len(packing) + int(len(packing) / 4)
    print(
        f"Searching packing combinations for: {packing} (max. instructions {MAX_INS}, minimum candidates {MIN_CANDIDATES})"
    )
    if len(packing) == 8:
        objective = set_8_float_elements(*packing)
    else:
        objective = set_4_float_elements(*packing)

    n_candidates = 0
    C = []
    t0 = time.time_ns()
    for n_ins in range(1, MAX_INS):
        global FOUND_SOLUTIONS
        FOUND_SOLUTIONS = len(C)
        print(f"- using max. {n_ins:3} instruction(s)")
        _C = find_all_candidates(n_ins, packing, objective)
        n_candidates += len(_C)
        if len(_C) > 0:
            C += [*_C]
        else:
            print(f"\tNo candidates using {n_ins} instruction(s)")
        if n_candidates >= MIN_CANDIDATES:
            print(f"*** SEARCH FINISHED WITH {n_candidates} CANDIDATES")
            break
    t_elapsed = (time.time_ns() - t0) / 1e9
    print(f"Time elapsed: {t_elapsed} sec.")

    for candidate in C:
        generate_code(candidate)
    return C


def generate_all_cases(max_size: int = 8) -> List[PackingT]:
    list_cases = []
    for i in range(1, max_size + 1):
        vector_size = 4 if i < 5 else 8
        for c in range(2 ** (i - 1)):
            values = [-1] * vector_size
            values[0] = 0
            contiguity = (
                list(map(lambda x: int(x), list(f"{c:(i-1)b}"))) if i > 1 else []
            )
            for v in range(len(contiguity) - 1, -1, -1):
                offset = 1 if contiguity[v] == 1 else 10
                values[v] = values[v - 1] + offset
            values.reverse()
            packing = PackingT(values, contiguity)
            list_cases.append(packing)
    return list_cases


debug = True

if not debug:
    all_cases = generate_all_cases()
    for case in all_cases:
        exploration_space(case)
else:
    case = PackingT([10, 9, 8, 6], [1, 1, 0])
    exploration_space(case)