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

from types import FunctionType
from typing import List, Any


class Mem:
    pass


class Mem:
    def __str__(self) -> str:
        # try:
        #     if eval(self.idx)
        #         return "0"
        # except NameError:
        #     pass
        return f"MEM[{self.idx}]"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: Mem) -> bool:
        p = self.p == other.p
        idx = self.idx == other.idx
        return p and idx

    def __hash__(self):
        return hash((self.p, self.idx))

    def __lt__(self, other):
        return eval(self.idx) < eval(other.idx)

    def __le__(self, other):
        return eval(self.idx) <= eval(other.idx)

    def eval_idx(self, value):
        try:
            return str(eval(self.idx.replace("N", str(value))))
        except TypeError:
            return self.idx

    def __init__(self, p="p", idx=0):
        self.p = p
        self.idx = idx


def identity(*args):
    return args


class TmpVal:
    def __init__(self, pos: int = 0, idx: int = 0):
        self.pos = pos
        self.idx = idx


class Reg:
    def eval_idx(self, value):
        r = []
        for s in self.slots:
            r.append(s.eval_idx(value))
        # r.reverse()
        return r

    def is_temporal(self):
        return self.varname == "__temp_reg"

    def get_size(self) -> int:
        return len(self.slots)

    def get_slot(self, pos: int) -> Mem:
        return self.slots[pos]

    def __str__(self):
        s = f"{self.varname} ["
        for arg in self.slots[:-1]:
            s += f"{arg},"
        s += f"{self.slots[-1]}"
        s += "]"
        return f"{s}"

    def __repr__(self) -> str:
        return (
            f"{self.__str__()} ({self.output_func.__name__}, eval = {self.needs_eval})"
        )

    def __getitem__(self, key):
        return self.slots[key]

    def __init__(
        self,
        slots: List[Any],
        output_func: FunctionType = identity,
        needs_eval: bool = False,
    ):
        self.varname = "__temp_reg"
        self.slots = slots
        self.output_func = output_func
        self.needs_eval = needs_eval


class Imm:
    def __init__(self, value: int = 0b00000000):
        self.value = value

    def __getitem__(self, key):
        return self.value


class Intrinsic:
    pass


class Intrinsic:
    cpuid_order = [
        "MMX",
        "SSE",
        "SSE2",
        "SSE3",
        "SSSE3",
        "SSE4.1",
        "SSE4.2",
        "AVX",
        "AVX2",
    ]

    def replace_args(self, out: Any, value: int, args: List[Any]) -> List[Any]:
        new_args = []
        for item in out:
            if TmpVal == type(item):
                print(item.pos, item.idx)
                new_args.append(args[item.pos][item.idx])
            elif Mem == type(item):
                new_args.append(Mem(item.p, item.eval_idx(value)))
            elif int == type(item):
                new_args.append(item)
            else:
                raise TypeError
        return new_args

    def evaluate_output(self, value: int, args: List[Any] = []) -> Reg:
        output = []
        if self.output.needs_eval:
            for out in self.output.slots:
                replaced_args = self.replace_args(out, value, args)
                output.append(self.output.output_func(*replaced_args))

        else:
            for mem in self.output.slots:
                output.append(Mem(mem.p, mem.eval_idx(value)))
        print(output)
        return Reg(output)

    def compatible_cpuid(self, cpuid):
        return self.cpuid_order.index(cpuid) <= self.cpuid_order.index(self.cpuid)

    def __str__(self):
        s = f"{self.name}("
        for arg in self.args[:-1]:
            s += f"{arg},"
        s += f"{self.args[-1]}"
        s += ")"
        return f"{s}"

    def __repr__(self) -> str:
        return f"{self.__str__()} -> {self.output}"

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.output == other.output
            and self.args == other.args
        )

    def __lt__(self, other):
        return self.output < other.output

    def __le__(self, other):
        return self.output <= other.output

    def __hash__(self):
        return hash((self.name, self.args, self.output))

    def __getitem__(self, key):
        return self.output[key]

    def render(self):
        return f"{self.output_var} = {self.__str__()};"

    def __init__(
        self,
        name: str,
        args: list,
        ret_type: str,
        c_type: str,
        width: int,
        output: Reg,
        cpuid="SSE4.2",
        aligned=False,
    ):
        self.name = name
        self.args = args
        self.ret_type = ret_type
        self.c_type = c_type
        self.width = width
        self.output = output
        self.output_var = ""
        self.aligned = aligned
        self.cpuid = cpuid


IntrinsicsList = List[Intrinsic]
MemList = List[Mem]


def generate_new_cases(size: int = 8) -> MemList:
    array = []
    for i in range(size):
        positions = list(range(size))
        for j in range(i):
            positions[-i + j] = (j + 1) * size
        array.append(positions)

    target_addresses = []
    offset = 0
    for comb in array:
        new_comb = [Mem("p", f"{offset + elem}") for elem in comb]
        target_addresses.append(new_comb)
    return target_addresses


def generate_debug_case() -> MemList:
    return [
        [
            Mem("p", "19"),
            Mem("p", "18"),
            Mem("p", "17"),
            Mem("p", "16"),
            Mem("p", "3"),
            Mem("p", "2"),
            Mem("p", "1"),
            Mem("p", "0"),
        ]
    ]
