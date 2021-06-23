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
import os
import json
from intrinsics import Mem, Intrinsic
import custom_mp
import bisect
import random

print_combinations = False
n_processes = 16


loads = [
    Intrinsic(
        "_mm_load_ss",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        [Mem("p", "-1"), Mem("p", "-1"), Mem("p", "-1"), Mem("p", "N+0")],
    ),
    Intrinsic(
        "_mm_loadu_ps",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        [
            Mem("p", "N+3"),
            Mem("p", "N+2"),
            Mem("p", "N+1"),
            Mem("p", "N+0"),
        ],
        "AVX2",
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


def gen_all_load_candidates(target_addr):
    all_load_candidates = []
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
    return all_load_candidates


def combinations_loads(all_load_candidates, cl=2, end=9):
    global_combinations = []
    for n in range(cl, end):
        n_candidates = 1
        niterations = sum(1 for _ in it.combinations(all_load_candidates, n))
        if len(global_combinations) > n_candidates:
            return global_combinations
        with mp.Pool(processes=n_processes) as pool:
            for output in tqdm.tqdm(
                pool.istarmap(
                    get_combinations,
                    zip(it.combinations(all_load_candidates, n)),
                    chunksize=25000,
                ),
                total=niterations,
            ):
                if output != []:
                    global_combinations.append(output)
                    if len(global_combinations) > n_candidates:
                        pool.terminate()
                        return global_combinations
    return global_combinations


def gen_shuffles(dst, A, B, mask="0b0000"):
    return f"{dst} = _mm256_shuffle_ps({A},{B},{mask});DO_NOT_TOUCH({dst});"


def get_cl(addresses):
    offsets = set()
    for addr in addresses:
        offsets.add(int(int(addr.idx) / 8))
    return len(offsets)


def generate_shuffle_instructions(comb):
    new_comb = comb
    nloads = len(comb)
    for n in range(2, nloads):
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
    return new_comb


def get_number_slots_ordered(candidate, target: list):
    v = {"full": 0, "low": 0, "high": 0}
    max_size = max(len(candidate), len(target))
    different_size = len(candidate) != len(target)
    for n in range(max_size):
        idx_candidate = n
        if different_size and n >= len(candidate):
            idx_candidate = n % len(candidate)
        if candidate[idx_candidate] == target[n]:
            if idx_candidate < n:
                v["high"] += 1
            else:
                v["low"] += 1
    return v


def get_blend_mask(target, source):
    m = get_number_slots_ordered(target.output, source.output)
    return "0x0"


def get_shuffle_mask(target, source):
    m = get_number_slots_ordered(target.output, source.output)
    return "0x0"


def get_cast(width, source):
    if width == "256" and source.width != 256:
        return f"_mm256_castps128_ps256({source.output_var})"
    if width == "" and source.width != 128:
        return f"_mm256_castps256_ps128({source.output_var})"


def get_blend(target_address, target_register, source):
    dst = target_register.output_var
    width = "256" if dst.width == 256 else ""
    a = target_register.output_var
    b = get_cast(width, source)
    if mask := get_blend_mask(target_address, source):
        return f"{dst} = _mm{width}_blend_ps({a},{b},{mask});"
    return None


def get_shuffle(target_address, target_register, source):
    dst = target_register.output_var
    width = "256" if dst.width == 256 else ""
    a = target_register.output_var
    b = get_cast(width, source)
    if mask := get_shuffle_mask(target_address, source):
        return f"{dst} = _mm{width}_shuffle_ps({a},{b},{mask});"
    return None


def generate_swizzle_instructions(comb, target_address: list):
    new_comb = copy.deepcopy(comb)
    main_ins = ""
    max_ordered = 0
    for load in comb:
        output = copy.deepcopy(load.output)
        output.reverse()
        m = get_number_slots_ordered(output, target_address)
        print(m)
        if m["low"] > max_ordered:
            main_ins = load
            max_ordered = m["low"]

    comb.remove(main_ins)
    # Are there any permutes?
    # TODO: permutes
    for load in comb:
        # Can it blend, otherwise just shuffle it
        if new_inst := get_blend(target_address, load, main_ins):
            comb.append(new_inst)
        else:
            comb.append(get_shuffle(target_address, load, main_ins))
    print(new_comb)
    return new_comb


def create_performance_bench(instructions) -> str:
    content = "#include <immintrin.h>\n"
    content += "#define restrict __restrict\n"
    content += '#define NO_OPT(X) asm volatile("":"+x"(X)::);\n'
    content += "void foo(float *restrict p) {\n"
    content += '    __asm volatile("# LLVM-MCA-BEGIN foo");'
    for instruction in instructions:
        if type(instruction) == Intrinsic:
            f.write(f"    {instruction.render()}\n")
        else:
            f.write(f"    {instruction}\n")
    content += '    __asm volatile("# LLVM-MCA-END foo");'
    for instruction in instructions:
        content += f"    NO_OPT({instruction.output_var})\n"
    content += "}\n"
    return content


def fix_json() -> None:
    with open("perf.json") as f:
        lines = f.readlines()
        new_lines = []
        new_lines.append("[\n")
        for l in lines:
            if "not implemented" in l or "\n" == l or "Code Region" in l:
                continue
            if l == "}\n":
                new_lines.append("},\n")
            elif l == "]\n":
                new_lines.append("],\n")
            else:
                new_lines.append(l)
        if new_lines[-1] == "},\n":
            new_lines[-1] = "}\n"
        elif new_lines[-1] == "],\n":
            new_lines[-1] = "]\n"
        new_lines.append("]\n")
    with open("perf_fixed.json", "w") as f:
        f.writelines(new_lines)


def compute_performance(combination) -> tuple:
    file_content = create_performance_bench(combination)
    with open("__tmp.c", "w") as f:
        f.write(file_content)
    os.system("gcc -c -O3 -S -mavx2 __tmp.c")
    os.system("llvm-mca-12 __tmp.s -json -o perf.json")
    fix_json()
    with open("perf_fixed.json") as f:
        dom = json.loads(f.read())
    summ = dom[1]["SummaryView"]
    ipc = summ["IPC"]
    avg_cycles = summ["Cycles"] / summ["Iterations"]

    # Be clean
    os.system("rm __tmp.c __tmp.s perf.json perf_fixed.json")
    return (ipc, avg_cycles)


def generate_new_cases() -> list:
    array = []
    for i in range(16):
        positions = list(range(16))
        for j in range(i):
            positions[-i + j] = (j + 1) * 16
        array.append(positions)

    target_addresses = []
    offset = 0
    for comb in array:
        new_comb = [Mem("p", f"{offset + elem}") for elem in comb]
        target_addresses.append(new_comb)
    return target_addresses


def generate_debug_case():
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


if __name__ == "__main__":
    # target_addresses = generate_new_cases()
    target_addresses = generate_debug_case()
    case_number = 0
    # CORE for the brute force approach
    for target_addr in target_addresses:
        target_addr.reverse()
        # Step 1.1: generate load candidates
        all_candidates = gen_all_load_candidates(target_addr)
        print(all_candidates)
        cl = get_cl(target_addr)
        # FIXME Step 1.2: generate all possible combinations with the
        # candidates. This needs to be pruned somehow.
        global_combinations = combinations_loads(
            all_candidates, 1, len(target_addr) + 1
        )
        print(global_combinations)
        # Delete duplicates
        global_combinations = sorted(global_combinations)
        new_list = list(k for k, _ in it.groupby(global_combinations))

        comb_number = 0
        for comb in new_list:
            # Step 2: generate swizzle instructions for each combination
            # considered in step 1.2
            new_comb = generate_swizzle_instructions(comb, target_addr)
            print(f"KERNEL {case_number}-{comb_number}")
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
