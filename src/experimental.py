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

import copy
import itertools as it
import tqdm
import multiprocessing as mp
import custom_mp

print_combinations = False
n_processes = 16


class VReg:
    def __str__(self):
        return f"{self.v}[{self.idx}]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        v = self.v == other.v
        idx = self.idx == other.idx
        return v and idx

    def __init__(self, v="v", idx=0):
        self.v = v
        self.idx = idx


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

    @staticmethod
    def generate_gather(base_addr: str, vindex: list, scale=9):
        vindex = ""
        gather_ins = f"__m256 tmp = _mm256_i32gather_ps({base_addr},{vindex},{scale});"
        return gather_ins

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


loads = [
    Intrinsic(
        "_mm_loadu_ps",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        [Mem("p", "N+3"), Mem("p", "N+2"), Mem("p", "N+1"), Mem("p", "N+0")],
    ),
    Intrinsic(
        "_mm_load_ss",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        [Mem("p", "-1"), Mem("p", "-1"), Mem("p", "-1"), Mem("p", "N+0")],
    ),
    Intrinsic(
        "_mm256_loadu_ps",
        [Mem("p", "N")],
        "__m256",
        "float",
        256,
        [
            Mem("p", "N+7"),
            Mem("p", "N+6"),
            Mem("p", "N+5"),
            Mem("p", "N+4"),
            Mem("p", "N+3"),
            Mem("p", "N+2"),
            Mem("p", "N+1"),
            Mem("p", "N+0"),
        ],
        "AVX2",
    ),
]


shuffles = [
    Intrinsic(
        "_mm_shuffle_ps",
        [Mem("p", "N"), Mem("p", "N"), "IMM8"],
        "__m128",
        "float",
        128,
        [Mem("p", "N+3"), Mem("p", "N+2"), Mem("p", "N+1"), Mem("p", "N+0")],
    )
]

tmp_reg = 0


def new_tmp_reg():
    global tmp_reg
    n = f"__tmp{tmp_reg}"
    tmp_reg += 1
    return n


def get_ptr(mem):
    return f"&{mem.p}[{mem.idx}]"


#################### MAIN
target_addr = [
    [
        Mem("p", "112"),
        Mem("p", "96"),
        Mem("p", "80"),
        Mem("p", "64"),
        Mem("p", "48"),
        Mem("p", "32"),
        Mem("p", "16"),
        Mem("p", "0"),
    ],
    [
        Mem("p", "7"),
        Mem("p", "6"),
        Mem("p", "5"),
        Mem("p", "4"),
        Mem("p", "3"),
        Mem("p", "2"),
        Mem("p", "1"),
        Mem("p", "0"),
    ],
]

target_addr = [
    Mem("p", "7"),
    Mem("p", "6"),
    Mem("p", "5"),
    Mem("p", "4"),
    Mem("p", "3"),
    Mem("p", "2"),
    Mem("p", "1"),
    Mem("p", "0"),
]

all_load_candidates = []
# O(N*M)
for mem_addr in target_addr:
    base_idx = mem_addr.idx
    for load_ins in loads:
        test_output = Intrinsic.evaluate_output(load_ins, base_idx)
        if mem_addr not in test_output:
            continue
        new_ins = copy.deepcopy(load_ins)
        new_ins.output = test_output
        new_ins.args = [get_ptr(mem_addr)]
        new_ins.output_var = new_tmp_reg()
        all_load_candidates.append(new_ins)

combinations = []
# O(N^2*M^2)
n_comb = 0
combinations = []
global_combinations = []


def get_combinations(combinations):
    addr_loaded = []
    new_comb = []
    for ins in combinations:
        for out in ins.output:
            if out in target_addr and out not in addr_loaded:
                addr_loaded.append(out)
                if ins not in new_comb:
                    new_comb.append(ins)
    if len(addr_loaded) == len(target_addr):
        return new_comb
    return []


for n in range(1, len(target_addr) + 1):
    niterations = sum(
        1 for _ in it.combinations_with_replacement(all_load_candidates, n)
    )
    with mp.Pool(processes=n_processes) as pool:
        for output in tqdm.tqdm(
            pool.istarmap(
                get_combinations,
                zip(it.combinations_with_replacement(all_load_candidates, n)),
                chunksize=50000,
            ),
            total=niterations,
        ):
            if output not in global_combinations:
                global_combinations.append(output)


if print_combinations:
    for comb in global_combinations:
        print("Combination")
        for ins in comb:
            print(f"    {ins}")
print(len(global_combinations))
