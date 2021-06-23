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


class Mem:
    def __str__(self):
        if eval(self.idx) < 0:
            return "0"
        return f"MEM[{self.idx}]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
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
            return str(eval(self.idx.replace("N", value)))
        except TypeError:
            return self.idx

    def __init__(self, p="p", idx=0):
        self.p = p
        self.idx = idx


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

    @staticmethod
    def evaluate_output(ins: Intrinsic, value: int) -> list:
        output = []
        for mem in ins.output:
            output.append(Mem(mem.p, mem.eval_idx(value)))
        return output

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

    def render(self):
        return f"{self.output_var} = {self.__str__()};"

    def __init__(
        self,
        name: str,
        args: list,
        ret_type: str,
        c_type: str,
        width: int,
        output: list,
        cpuid="SSE4.2",
    ):
        self.name = name
        self.args = args
        self.ret_type = ret_type
        self.c_type = c_type
        self.width = width
        self.output = output
        self.output_var = ""
        self.cpuid = cpuid
