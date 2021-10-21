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

import os
from z3 import sat, unsat
from sat.x86_sat.evaluate import check, Var, Call
from typing import List

DATA_TYPE = {
    "float": 4,
    "double": 8,
}


def max_width(dtype: str = "float", nelems: int = 4):
    return int(DATA_TYPE[dtype] * 8 * nelems)


class PackingT:
    def __repr__(self):
        return f"{self.packing} ({self.dtype})"

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
        self.nnz = len(packing) - packing.count(0)
        self.vector_size = len(self.packing)
        self.contiguity = contiguity
        self.min_instructions = self.contiguity.count(0)
        self.max_instructions = self.min_instructions + int(self.vector_size / 4)
        self.dtype = dtype
        self.c_max_width = max_width(dtype, len(packing))
        assert len(contiguity) + 1 == self.nnz


class Candidate:
    def __len__(self):
        return len(self.instructions)

    def __init__(self, instructions=None, model=None):
        self.instructions = [] if instructions is None else instructions
        self.model = model


def print_instruction(ins, output, *args):
    instruction = f"{ins.fn.name}("
    if "load" in instruction:
        instruction += "&"
    if not instruction.startswith("_mv"):
        instruction = f"{output} = {instruction}"
    else:
        instruction += f"{output},"
    instruction += ",".join(args) + ");"
    return instruction


def get_register(args, arg, tmp_reg, candidate):
    for tmp in tmp_reg:
        res, m = check(arg == tmp_reg[tmp])
        if res == unsat:
            continue
        model_ok = True
        for i in m:
            model_ok = model_ok and m[i] == candidate.model[i]
        if sat == res and model_ok:
            args += [tmp]
            break
    return args


def get_arguments(ins, tmp_reg, candidate):
    args = []
    for arg in ins.args:
        if isinstance(arg, Var):
            try:
                args += [candidate.model[arg.name]]
            except KeyError:
                args += ["0x0"]
            continue
        if not isinstance(arg, Call):
            args += [f"p[{arg}]"]
            continue
        n_args = len(args)
        args = get_register(args, arg, tmp_reg, candidate)
        assert n_args + 1 == len(args)
    return args


def generate_code(candidate: Candidate):
    reg_no = 0
    tmp_reg = {}
    liveness = {}
    instructions = candidate.instructions
    c_instructions = []
    registers = {"__m128": [], "__m256": []}
    for n in range(len(instructions)):
        ins = instructions[n]
        args = get_arguments(ins, tmp_reg, candidate)
        output = f"r{reg_no}" if n + 1 < len(instructions) else "output"
        registers[f"__m{ins.width}"].append(output)
        tmp_reg[output] = ins
        liveness[output] = [n, n]
        reg_no += 1
        for i in args:
            if not i.startswith("r"):
                continue
            liveness[i][1] = n
        c_instructions.append(print_instruction(ins, output, *args))

    for reg in liveness:
        out = liveness[reg]
        if out[0] == out[1] and reg != "output":
            del c_instructions[out[0]]
            for w in registers:
                try:
                    del registers[w][registers[w].index(reg)]
                except ValueError:
                    pass

    return registers, c_instructions


def generate_micro_benchmark(candidate: Candidate, dtype: str = "float"):
    try:
        os.mkdir("codes")
    except FileExistsError:
        pass
    registers, c_instructions = generate_code(candidate)
    packing_str = (
        f"{candidate.packing.vector_size}elems_{candidate.packing.nnz}nnz_"
        + "_".join(list(map(lambda x: str(x), candidate.packing.contiguity)))
    )
    body_str = ""
    for k in registers:
        if len(registers[k]) == 0:
            continue
        body_str += f"{k} {', '.join(registers[k])};\n    "
    body_str += "\n    ".join(c_instructions)
    file_name = f"codes/rvp_{dtype}_{packing_str}_{candidate.number}.c"
    kernel = f"""#include "marta_wrapper.h"
#include "macveth_api.h"

void kernel({dtype} *restrict p) {{
    // {candidate.packing}
    {body_str}
    DO_NOT_TOUCH(output);
}}
    """
    with open(file_name, "w+") as f:
        f.write(kernel)
        print(f"Micro-benchmark written in {file_name}")
