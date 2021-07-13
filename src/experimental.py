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

import sys
import copy
import itertools as it
import tqdm
import multiprocessing as mp
import os
import json
from intrinsics import (
    Mem,
    Intrinsic,
    MemList,
    IntrinsicsList,
    generate_debug_case,
    generate_new_cases,
)
from typing import Union, List, Tuple
import custom_mp
import bisect
import random


print_combinations = False
n_processes = 1


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
        "_mm_load_ps",
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
        True,
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
    Intrinsic(
        "_mm256_load_ps",
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
        True,
    ),
]

tmp_reg = 0


def new_tmp_reg():
    global tmp_reg
    n = f"__tmp{tmp_reg}"
    tmp_reg += 1
    return n


def get_ptr(mem: Mem):
    return f"&{mem.p}[{mem.idx}]"


def is_aligned(mem: Mem, alignment=8):
    return eval(mem.idx) % alignment != 0


def get_combinations(combinations: IntrinsicsList) -> IntrinsicsList:
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


def gen_all_load_candidates(target_addr: MemList) -> IntrinsicsList:
    all_load_candidates = []
    for mem_addr in target_addr:
        base_idx = mem_addr.idx
        for load_ins in loads:
            test_output = Intrinsic.evaluate_output(load_ins, base_idx)
            if (
                mem_addr not in test_output
                or load_ins.aligned
                and not is_aligned(mem_addr, 8)
            ):
                continue
            new_ins = copy.deepcopy(load_ins)
            new_ins.output = test_output
            new_ins.args = [get_ptr(mem_addr)]
            new_ins.output_var = new_tmp_reg()
            all_load_candidates.append(new_ins)
    return all_load_candidates


def combinations_loads(
    all_load_candidates: IntrinsicsList, cl=2, end=9, n_candidates=1, csize=25000
) -> IntrinsicsList:
    global_combinations = []
    for n in range(cl, end):
        niterations = sum(1 for _ in it.combinations(all_load_candidates, n))
        if len(global_combinations) > n_candidates:
            return global_combinations
        with mp.Pool(processes=n_processes) as pool:
            for output in tqdm.tqdm(
                pool.istarmap(
                    get_combinations,
                    zip(it.combinations(all_load_candidates, n)),
                    chunksize=csize,
                ),
                total=niterations,
            ):
                if output != []:
                    global_combinations.append(output)
                    if len(global_combinations) > n_candidates:
                        pool.terminate()
                        return global_combinations
    return global_combinations


def get_cl(addresses: MemList) -> int:
    offsets = set()
    for addr in addresses:
        offsets.add(int(int(addr.idx) / 8))
    return len(offsets)


def get_number_slots_ordered(candidate: list, target: MemList) -> dict:
    v = {"low": 0, "high": 0}
    max_size = max(len(candidate), len(target))
    different_size = len(candidate) != len(target)
    offset = 0
    while sum(v.values()) == 0 and offset <= max_size:
        for n in range(max_size - 1, offset - 1, -1):
            idx_candidate = n - offset
            if offset >= len(candidate):
                continue
            if different_size and n >= len(candidate):
                idx_candidate = n % len(candidate)
            if candidate[idx_candidate] == target[n]:
                if n in range(0, int(len(target) / 2)):
                    v["high"] += 1
                else:
                    v["low"] += 1
        offset += 1
    if sum(v.values()) == 0:
        if target[0:4] == candidate[0:4]:
            v["high"] = 4
        if target[4:8] == candidate[0:4]:
            v["low"] = 4
        if not different_size:
            if target[0:4] == candidate[4:8]:
                v["high"] += 4
            if target[4:8] == candidate[4:8]:
                v["low"] += 4

    return v


def get_blend_mask(target: MemList, source: Intrinsic) -> Union[None, str]:
    m = get_number_slots_ordered(
        source.output,
        target,
    )
    if m["low"] == 0 and m["high"] == 0:
        return None
    mask = 0x0
    min_length = min(len(target), len(source.output))
    for n in range(min_length):
        if target[n].idx == source.output[n].idx:
            # if n == int(target[n].idx):
            mask |= 1 << (-n + min_length - 1)
    format_hex = "0x{:02x}".format(mask)
    return format_hex


def get_shuffle_mask(positions: Tuple[int, int], size) -> str:
    idx_source, idx_target = positions
    mask = idx_source << (2 * ((-idx_target + size) % 4))
    return str(hex(mask))


def get_cast(width: str, source: Intrinsic) -> str:
    if width == "256" and source.width != 256:
        return f"_mm256_castps128_ps256({source.output_var})"
    if width == "" and source.width != 128:
        return f"_mm256_castps256_ps128({source.output_var})"
    return source.output_var


def get_blend(
    target_address: MemList, target_register: Intrinsic, source: Intrinsic, result=False
) -> Union[None, str]:
    width = "256"
    b = get_cast(width, source)
    if result:
        dst = "result"
        a = "result"
    else:
        dst = target_register.output_var
        if target_register.width == 128:
            dst = "result"
        a = get_cast(width, target_register)
    if mask := get_blend_mask(target_address, source):
        new_inst = Intrinsic(
            f"_mm{width}_blend_ps",
            [a, b, mask],
            "__m256",
            "float",
            256,
            [],
            "AVX2",
        )
        new_inst.output_var = dst
        return new_inst
    return None


def get_shuffle(
    positions: tuple, target_register: Intrinsic, source: Intrinsic
) -> Union[None, str]:
    dst = target_register.output_var
    width = "256" if target_register.width == 256 else ""
    a = target_register.output_var
    _, idx_target = positions
    if idx_target in [0, 1, 4, 5]:
        a = get_cast(width, source)
    b = get_cast(width, source)
    size = len(source.output)
    if mask := get_shuffle_mask(positions, size):
        return f"{dst} = _mm{width}_shuffle_ps({a},{b},{mask});"
    return None


def get_permute_index(
    positions: Tuple[int, int], source: Intrinsic
) -> Tuple[str, MemList]:
    idx_source, idx_target = positions
    signature = "_mm256_set_epi32"
    indices = ""
    memlist = []
    offset_output = len(source.output) - 1
    for i in range(7, -1, -1):
        if i == idx_target:
            indices += f"{idx_source},"
            memlist.append(source.output[-idx_source + offset_output])
        else:
            indices += f"{i},"
            if i >= len(source.output):
                memlist.append(Mem("p", "-1"))
            else:
                memlist.append(source.output[-i + offset_output])
    return f"{signature}({indices[:-1]})", memlist


def get_permute(
    positions: Tuple[int, int], source: Intrinsic
) -> Union[None, Intrinsic]:
    a = get_cast("256", source)
    dst = source.output_var
    if source.width == 128:
        dst = new_tmp_reg()
    if val := get_permute_index(positions, source):
        mask, output = val
        new_inst = Intrinsic(
            "_mm256_permutevar8x32_ps",
            [a, mask],
            "__m256",
            "float",
            256,
            output,
            "AVX2",
        )
        new_inst.output_var = dst
        return new_inst
    return None


def get_inserts(loads: list, target_address: list) -> Union[str, None]:
    # _mm256_set_m128()
    low = None
    high = None
    for load in loads:
        m = get_number_slots_ordered(load.output, target_address)
        if m["low"] == 4:
            low = load
        if m["high"] == 4:
            high = load

    if low is not None and high is not None:
        dst = "result"
        if low.width == 256:
            dst = low.output_var
        if high.width == 256:
            dst = high.output_var
        return f"{dst} = _mm256_set_m128({get_cast('',high)}, {get_cast('',low)});"

    return None


def get_swizzle_instruction(
    target_address: MemList, load: Intrinsic, main_ins: Intrinsic
) -> list:
    instruction = []
    output = load.output
    # find positions
    positions = []
    for idx_target in range(len(target_address)):
        if target_address[idx_target] not in output:
            continue
        idx_load = (len(output) - 1) - output.index(target_address[idx_target])
        positions.append((idx_load, (len(target_address) - 1 - idx_target)))

    shuffles = []
    for (idx_load, idx_target) in positions:
        # Blending
        if idx_load == idx_target:
            new_blend = get_blend(target_address, main_ins, load)
            instruction.append(new_blend)
            continue
        # if same_128b, then we could avoid shuffling
        same_128b = int(idx_load / 4) == int(idx_target / 4)
        if same_128b:
            # shuffle:
            skip = [
                1
                for sidx, si in shuffles
                if idx_load == sidx * 2 and idx_target == si * 2
            ]
            if sum(skip) > 0:
                continue
            new_shuffle = get_shuffle((idx_load, idx_target), main_ins, load)
            instruction.append(new_shuffle)
            shuffles.append((idx_load, idx_target))
        else:
            # permute + blend
            permute = get_permute((idx_load, idx_target), load)
            instruction.append(permute)
            instruction.append(get_blend(target_address, main_ins, permute, True))

    return instruction


def generate_swizzle_instructions(
    comb: IntrinsicsList, target_address: MemList
) -> IntrinsicsList:
    new_comb = copy.deepcopy(comb)
    main_ins = ""
    max_ordered = 0

    # TODO: check if main_ins size is equal to the target list, or if there are
    # needed more than one register, e.g. in AVX2 packing 16 floats

    # Inserts
    if inserts := get_inserts(comb, target_address):
        new_comb.append(inserts)
        return new_comb

    for load in comb:
        m = get_number_slots_ordered(load.output, target_address)
        if m["low"] > max_ordered:
            main_ins = load
            max_ordered = m["low"]

    comb.remove(main_ins)
    for load in comb:
        new_comb += get_swizzle_instruction(target_address, load, main_ins)
    return new_comb


def get_variables_from_comb(new_comb: IntrinsicsList) -> Tuple[List, List]:
    variables128 = []
    variables256 = []
    for ins in new_comb:
        if type(ins) == Intrinsic:
            if (
                ins.width == 128
                and ins.output_var not in variables128
                and ins.output_var not in variables256
            ):
                variables128.append(ins.output_var)
            if (
                ins.width == 256
                and ins.output_var not in variables256
                and ins.output_var not in variables128
            ):
                variables256.append(ins.output_var)
        elif ins != None:
            var = ins.split("=")[0].strip()
            if var != None and var not in variables256 and var not in variables128:
                variables256.append(var)
    return variables128, variables256


def get_gather(target_addr: MemList) -> List[str]:
    gathers = []
    for i in range(0, len(target_addr), 8):
        signature = "_mm256_i32gather_ps"
        index = "_mm256_set_epi32"
        indices = ""
        for addr in target_addr[i : i + 8]:
            indices += f"{addr.idx},"
        index = f"{index}({indices[:-1]})"
        dst = "result"
        base_addr = get_ptr(target_addr[0])
        gathers.append(f"{dst} = {signature}({base_addr},{index},4);\n")
    return gathers


def create_performance_bench(
    case_number: int,
    comb_number: Union[int, str],
    name_bench: str,
    variables128: list,
    variables256: list,
    instructions: IntrinsicsList,
) -> str:
    with open(f"{name_bench}.c", "w+") as f:
        f.write("#include <immintrin.h>\n")
        f.write("#define restrict __restrict\n")
        f.write('#define MARTA_AVOID_DCE(X) asm volatile("":"+x"(X)::)\n')
        f.write("void foo(float *restrict p) {\n")
        write_variables(f, variables128, variables256)
        f.write('    __asm volatile("# LLVM-MCA-BEGIN foo");\n')
        if type(instructions) == str:
            f.write(f"    {instructions}\n")
        else:
            for instruction in list(instructions):
                if type(instruction) == Intrinsic:
                    f.write(f"    {instruction.render()}\n")
                elif type(instruction) == str:
                    f.write(f"    {instruction}\n")
        f.write('    __asm volatile("# LLVM-MCA-END foo");\n')
    write_dce(
        case_number,
        comb_number,
        variables128,
        variables256,
        "a",
        "perf_kernels/__tmp_",
        ".c",
    )
    with open(f"{name_bench}.c", "a") as f:
        f.write("}")
    return name_bench


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


def compute_performance(
    variables128: list,
    variables256: list,
    combination: IntrinsicsList,
    case_number: Union[str, int],
    comb_number: Union[str, int],
) -> dict:
    name_bench = f"perf_kernels/__tmp_{case_number}_{comb_number}"
    create_performance_bench(
        case_number, comb_number, name_bench, variables128, variables256, combination
    )
    os.system(
        f"gcc -c -O3 -mavx2 -march=cascadelake -mtune=cascadelake -S {name_bench}.c"
    )
    os.system(f"mv __tmp_{case_number}_{comb_number}.s perf_kernels/")
    os.system(
        f"llvm-mca-12 -march=x86-64 -mcpu=cascadelake -iterations=1 {name_bench}.s -json -o perf.json"
    )
    fix_json()
    os.system(f"mv perf_fixed.json {name_bench}.json")
    with open(f"{name_bench}.json") as f:
        dom = json.loads(f.read())
    summ = dom[1]["SummaryView"]
    d = {}
    d.update({"IPC": summ["IPC"]})
    d.update({"CyclesPerIteration": summ["TotalCycles"] / summ["Iterations"]})
    d.update({"uOpsPerCycle": summ["uOpsPerCycle"]})

    # Be clean
    os.system(f"rm perf.json")
    return d


def write_variables(f, variables128, variables256):
    variables128 = [
        v for v in variables128 if (v != " " and v != "" and v.startswith("__"))
    ]
    variables256 = [
        v for v in variables256 if (v != " " and v != "" and v.startswith("__"))
    ]
    if len(variables128) > 0:
        f.write("    __m128 ")
        for v in variables128[:-1]:
            f.write(f"{v}, ")
        f.write(f"{variables128[-1]};\n")
    if len(variables256) > 0:
        f.write("    __m256 ")
        for v in variables256[:-1]:
            f.write(f"{v}, ")
        f.write(f"{variables256[-1]}, result;\n")
    else:
        f.write("    __m256 result;\n")


def write_decl(
    case_number: int,
    comb_number: Union[int, str],
    variables128: list,
    variables256: list,
):
    with open(f"kernels/kernel_{case_number}_{comb_number}.decl.c", "w") as f:
        write_variables(f, variables128, variables256)


def write_kernel(case_number: int, comb_number: int, new_comb: list) -> None:
    with open(f"kernels/kernel_{case_number}_{comb_number}.c", "w") as f:
        for instruction in new_comb:
            if type(instruction) == Intrinsic:
                f.write(f"{instruction.render()}\n")
            elif type(instruction) == str:
                f.write(f"{instruction}\n")


def write_dce(
    case_number: int,
    comb_number: Union[int, str],
    variables128: list,
    variables256: list,
    open_type="w",
    file_name="kernels/kernel_",
    suffix=".dce.c",
) -> None:
    full_file_name = f"{file_name}{case_number}_{comb_number}{suffix}"
    with open(full_file_name, open_type) as f:
        if len(variables128) > 0:
            for v in variables128[:-1]:
                f.write(f"    MARTA_AVOID_DCE({v});\n")
        if len(variables256) > 0:
            for v in variables256:
                f.write(f"    MARTA_AVOID_DCE({v});\n")


def write_to_files(
    case_number: int,
    comb_number: Union[int, str],
    variables128: list,
    variables256: list,
    ins: list,
) -> None:
    write_decl(case_number, comb_number, variables128, variables256)
    write_kernel(case_number, comb_number, ins)
    write_dce(case_number, comb_number, variables128, variables256)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        target_addresses = generate_debug_case()
    else:
        target_addresses = generate_new_cases()

    case_number = 0
    full_experiments = {}
    for target_addr in target_addresses:
        target_addr.reverse()
        # TODO: prune based on e-graphs, could be done?
        # Step 1.1: generate load candidates
        all_candidates = gen_all_load_candidates(target_addr)
        cl = get_cl(target_addr)
        # Step 1.2: generate all possible combinations with the
        # candidates. This needs to be pruned somehow.
        global_combinations = combinations_loads(
            all_candidates, 1, len(target_addr) + 1, 10
        )

        # Naïve pruning: delete duplicates
        global_combinations = sorted(global_combinations)
        new_list = list(k for k, _ in it.groupby(global_combinations))
        target_addr.reverse()
        comb_number = 0
        for comb in new_list:
            # Step 2: generate swizzle instructions for each combination
            # considered in step 1.2. This needs to be redone.
            new_comb = generate_swizzle_instructions(comb, target_addr)
            variables128, variables256 = get_variables_from_comb(new_comb)

            write_to_files(
                case_number, comb_number, variables128, variables256, new_comb
            )
            new_dict = compute_performance(
                variables128, variables256, new_comb, case_number, comb_number
            )
            new_dict = {f"KERNEL-{case_number}-{comb_number}": new_dict}
            full_experiments.update(new_dict)
            print(new_dict)
            comb_number += 1
        # DO GATHER
        new_comb = get_gather(target_addr)
        variables128, variables256 = get_variables_from_comb(new_comb)
        write_to_files(case_number, "gather", variables128, variables256, new_comb)
        new_dict = compute_performance(
            variables128, variables256, new_comb, case_number, "gather"
        )
        new_dict = {f"KERNEL-{case_number}-gather": new_dict}
        full_experiments.update(new_dict)
        print(new_dict)
        case_number += 1
