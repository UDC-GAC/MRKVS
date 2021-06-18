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
import bisect
import random

print_combinations = False
n_processes = 16


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

    # @staticmethod
    # def generate_gather(base_addr: str, vindex: list, scale=9):
    #     vindex = ""
    #     gather_ins = f"__m256 tmp = _mm256_i32gather_ps({base_addr},{vindex},{scale});"
    #     return gather_ins

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
    # Intrinsic(
    #     "_mm_loadu_ps",
    #     [Mem("p", "N")],
    #     "__m128",
    #     "float",
    #     128,
    #     [Mem("p", "N+3"), Mem("p", "N+2"), Mem("p", "N+1"), Mem("p", "N+0")],
    # ),
    # Intrinsic(
    #     "_mm_loadu_ps1",
    #     [Mem("p", "N")],
    #     "__m128",
    #     "float",
    #     128,
    #     [Mem("p", "N+0"), Mem("p", "N+0"), Mem("p", "N+0"), Mem("p", "N+0")],
    # ),
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


# shuffles = [
#     Intrinsic(
#         "_mm_shuffle_ps",
#         [Mem("p", "N"), Mem("p", "N"), "IMM8"],
#         "__m128",
#         "float",
#         128,
#         [Mem("p", "N+3"), Mem("p", "N+2"), Mem("p", "N+1"), Mem("p", "N+0")],
#     )
# ]

tmp_reg = 0


def new_tmp_reg():
    global tmp_reg
    n = f"__tmp{tmp_reg}"
    tmp_reg += 1
    return n


def get_ptr(mem):
    return f"&{mem.p}[{mem.idx}]"


def get_combinations(combinations):
    addr_loaded = []
    new_comb = []
    for ins in combinations:
        for out in ins.output:
            if out in target_addr and out not in addr_loaded:
                addr_loaded.append(out)
                if ins not in new_comb:
                    new_comb.append(ins)
    # print(mp.current_process())
    if len(addr_loaded) == len(target_addr):
        return new_comb
    return []


def gen_all_load_candidates(target_addr):
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
    # sample_size = 10 if len(all_load_candidates) > 10 else len(all_load_candidates)
    # all_load_candidates = random.sample(all_load_candidates, sample_size)
    # print(sample_size)
    return all_load_candidates


def combinations_with_replacement_loads(all_load_candidates, cl=2, end=9):
    global_combinations = []
    for n in range(cl, end):
        n_candidates = 5
        niterations = sum(
            1 for _ in it.combinations_with_replacement(all_load_candidates, n)
        )
        if len(global_combinations) > n_candidates:
            return global_combinations
        with mp.Pool(processes=n_processes) as pool:
            for output in tqdm.tqdm(
                pool.istarmap(
                    get_combinations,
                    zip(it.combinations_with_replacement(all_load_candidates, n)),
                    chunksize=50000,
                ),
                total=niterations,
            ):
                # if output != [] and output not in global_combinations:
                if output != []:
                    global_combinations.append(output)
                    if len(global_combinations) > n_candidates:
                        pool.terminate()
                        return global_combinations
                # bisect.insort(global_combinations, output)
    return global_combinations


#################### MAIN
# target_addresses = [
#     # # 8 cl
#     # [
#     #     Mem("p", "112"),
#     #     Mem("p", "96"),
#     #     Mem("p", "80"),
#     #     Mem("p", "64"),
#     #     Mem("p", "48"),
#     #     Mem("p", "32"),
#     #     Mem("p", "16"),
#     #     Mem("p", "0"),
#     # ],
#     # [
#     #     Mem("p", "98"),
#     #     Mem("p", "65"),
#     #     Mem("p", "48"),
#     #     Mem("p", "32"),
#     #     Mem("p", "17"),
#     #     Mem("p", "16"),
#     #     Mem("p", "1"),
#     #     Mem("p", "0"),
#     # ],
#     # [
#     #     Mem("p", "98"),
#     #     Mem("p", "48"),
#     #     Mem("p", "33"),
#     #     Mem("p", "32"),
#     #     Mem("p", "17"),
#     #     Mem("p", "16"),
#     #     Mem("p", "1"),
#     #     Mem("p", "0"),
#     # ],
#     # # 4 cl
#     # [
#     #     Mem("p", "49"),
#     #     Mem("p", "48"),
#     #     Mem("p", "33"),
#     #     Mem("p", "32"),
#     #     Mem("p", "17"),
#     #     Mem("p", "16"),
#     #     Mem("p", "1"),
#     #     Mem("p", "0"),
#     # ],
#     # # 3 cl
#     # [
#     #     Mem("p", "49"),
#     #     Mem("p", "48"),
#     #     Mem("p", "17"),
#     #     Mem("p", "16"),
#     #     Mem("p", "3"),
#     #     Mem("p", "2"),
#     #     Mem("p", "1"),
#     #     Mem("p", "0"),
#     # ],
#     # # 2 CL
#     # [
#     #     Mem("p", "19"),
#     #     Mem("p", "18"),
#     #     Mem("p", "17"),
#     #     Mem("p", "16"),
#     #     Mem("p", "3"),
#     #     Mem("p", "2"),
#     #     Mem("p", "1"),
#     #     Mem("p", "0"),
#     # ],
#     # 1 cl
#     [
#         Mem("p", "15"),
#         Mem("p", "14"),
#         Mem("p", "13"),
#         Mem("p", "12"),
#         Mem("p", "11"),
#         Mem("p", "10"),
#         Mem("p", "9"),
#         Mem("p", "8"),
#         Mem("p", "7"),
#         Mem("p", "6"),
#         Mem("p", "5"),
#         Mem("p", "4"),
#         Mem("p", "3"),
#         Mem("p", "2"),
#         Mem("p", "1"),
#         Mem("p", "0"),
#     ],
# ]

array = []
for i in range(16):
    positions = list(range(16))
    for j in range(i):
        positions[-i + j] = (j + 1) * 16

    array.append(positions)
    print(positions)

target_addresses = []
offset = 0
for comb in array:
    # for offset in [0, 1, 2, 3, 8, 9, 11]:
    new_comb = [Mem("p", f"{offset + elem}") for elem in comb]
    target_addresses.append(new_comb)
    print(new_comb)

print(len(target_addresses))

# import sys

# sys.exit(0)


def gen_shuffles(dst, A, B, mask="0b0000"):
    return f"{dst} = _mm256_shuffle_ps({A},{B},{mask});DO_NOT_TOUCH({dst});"


def get_cl(addresses):
    offsets = set()
    for addr in addresses:
        offsets.add(int(int(addr.idx) / 8))
    return len(offsets)


case_number = 0
for target_addr in target_addresses:
    all_candidates = gen_all_load_candidates(target_addr)
    cl = get_cl(target_addr)
    global_combinations = combinations_with_replacement_loads(all_candidates, cl, 9)
    if print_combinations:
        for comb in global_combinations:
            print("Combination")
            for ins in comb:
                print(f"    {ins}")
    print(f"total {len(global_combinations)}")
    import time

    from merge_sort_parallel import merge_sort_parallel as msort

    start_time = time.monotonic()
    global_combinations = sorted(global_combinations)
    stop_time = time.monotonic()
    print("seconds: ", stop_time - start_time)
    start_time = time.monotonic()
    new_list = list(k for k, _ in it.groupby(global_combinations))
    print("seconds: ", time.monotonic() - start_time)
    print(f"cleaned total {len(new_list)}")

    comb_number = 0
    for comb in new_list:
        print(f"KERNEL {case_number}-{comb_number}")
        new_comb = comb
        nloads = len(comb)
        for n in range(2, nloads):
            # new_comb.append("#ifdef MASK_SHUFFLE")
            # new_comb.append(f"mask = 256 >> {int(8 * random.random())};")
            # new_comb.append("#else")
            # new_comb.append(f"mask = {hex(int(256 * random.random()))};")
            # new_comb.append("#endif")
            A = (
                f"_mm256_castps128_ps256({comb[n-1].output_var})"
                if comb[n - 1].width == 128
                else comb[n - 1].output_var
            )
            B = (
                f"_mm256_castps128_ps256({comb[n].output_var})"
                if comb[n].width == 128
                else comb[n].output_var
            )
            dst = new_tmp_reg()
            shuffle = gen_shuffles(
                dst,
                A,
                B,
                "mask",
            )
            new_comb.append(shuffle)

        variables128 = []
        variables256 = []
        for ins in new_comb:
            if type(ins) == Intrinsic:
                if ins.width == 128 and ins.output_var not in variables128:
                    variables128.append(ins.output_var)
                if ins.width == 256 and ins.output_var not in variables256:
                    variables256.append(ins.output_var)
            else:
                variables256.append(ins.split(" = ")[0].strip())

        with open(f"kernels/kernel_{case_number}_{comb_number}.decl.c", "w") as f:
            f.write("int mask;\n")
            if len(variables128) > 0:
                f.write("__m128 ")
                for v in variables128[:-1]:
                    f.write(f"{v}, ")
                f.write(f"{variables128[-1]};\n")
            if len(variables256) > 0:
                f.write("__m256 ")
                for v in variables256[:-1]:
                    f.write(f"{v}, ")
                f.write(f"{variables256[-1]};\n")

        with open(f"kernels/kernel_{case_number}_{comb_number}.c", "w") as f:
            for ins in new_comb:
                if type(ins) == Intrinsic:
                    f.write(f"{ins.render()}\n")
                else:
                    f.write(f"{ins}\n")

        with open(f"kernels/kernel_{case_number}_{comb_number}.dce.c", "w") as f:
            if len(variables256) > 0:
                f.write("MARTA_AVOID_DCE(")
                for v in variables256[:-1]:
                    f.write(f"{v}, ")
                f.write(f"{variables256[-1]});\n")
            if len(variables128) > 0:
                f.write("MARTA_AVOID_DCE(")
                for v in variables128[:-1]:
                    f.write(f"{v}, ")
                f.write(f"{variables128[-1]});\n")

        comb_number += 1
    case_number += 1


# print(len(set(global_combinations)))
