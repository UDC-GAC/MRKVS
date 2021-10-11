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

from z3 import sat, unsat
from sat.x86_sat.evaluate import check, Var, Call

DATA_TYPE = {
    "float": 4,
    "double": 8,
}


def max_width(dtype: str = "float", nelems: int = 4):
    return int(DATA_TYPE[dtype] * 8 * nelems)


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
    for arg in args:
        instruction += f"{arg},"
    instruction = f"{instruction[:-1]});"
    print(f"[BACKEND]\t{instruction}")
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
        if type(arg) == Var:
            try:
                args += [candidate.model[arg.name]]
            except Exception:
                pass
            continue
        if type(arg) != Call:
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
    for n in range(len(instructions)):
        ins = instructions[n]
        args = get_arguments(ins, tmp_reg, candidate)
        output = f"r{reg_no}"
        tmp_reg[output] = ins
        reg_no += 1
        liveness[reg_no] = [n, n]
        c_instructions.append(print_instruction(ins, output, *args))
    print(liveness)