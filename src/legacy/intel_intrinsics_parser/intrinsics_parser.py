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

#!/usr/bin/python3
import operation_parser as OParser
import xml.etree.ElementTree as ET
import sys
import pandas as pd
import json
import requests
from tqdm import tqdm
import pathlib

DEBUG = False


class Parameter:
    d_etypes = {
        "FP16": 16,
        "FP32": 32,
        "FP64": 64,
        "IMM": 8,
        "M128": 128,
        "M256": 256,
        "M512": 512,
        "MASK": -1,
        "SI16": 16,
        "SI32": 32,
        "SI64": 64,
        "SI8": 8,
        "UI16": 16,
        "UI32": 32,
        "UI64": 64,
        "UI8": 8,
    }

    d_ctypes = {
        "__int16": 16,
        "__int32": 32,
        "__int32*": 32,
        "__int64": 64,
        "__int64*": 64,
        "__int8": 8,
        "__m128": 128,
        "__m128*": 128,
        "__m128d": 128,
        "__m128d*": 128,
        "__m128i": 128,
        "__m128i*": 128,
        "__m256": 256,
        "__m256d": 256,
        "__m256i": 256,
        "__m256i*": 256,
        "__m512": 512,
        "__m512*": 512,
        "__m512bh": 512,
        "__m512d": 512,
        "__m512d*": 512,
        "__m512i": 512,
        "__m64": 64,
        "__m64*": 64,
        "__mmask16": 16,
        "__mmask16*": 16,
        "__mmask32": 32,
        "__mmask32*": 32,
        "__mmask64": 64,
        "__mmask64*": 64,
        "__mmask8": 8,
        "__mmask8*": 8,
        "char": 8,
        "char*": 8,
        "double": 64,
        "double*": 64,
        "float": 32,
        "float*": 32,
        "int": 32,
        "int*": 32,
        "longlong": 64,
        "short": 16,
        "size_t": 32,
        "size_t*": 32,
        "unsigned__int64": 64,
        "unsigned__int64*": 64,
        "unsignedchar": 8,
        "unsignedchar*": 8,
        "unsignedlong": 32,
        "unsignedint": 32,
        "unsignedint*": 32,
        "unsignedshort": 16,
        "unsignedshort*": 16,
        "void": 64,
        "void*": 64,
        "void**": 64,
    }

    def get_number_of_slots(self):
        # Memory address
        try:
            if self.memwidth != None and self.etype:
                return int(self.memwidth) // self.d_etypes[self.etype]
            if self.immwidth != None:
                return self.immwidth
            if self.etype != None:
                return self.d_ctypes[self.ret] // self.d_etypes[self.etype]
        except KeyError:
            return -1
        # Masks
        return 1

    def __init__(self, node):
        self.ret = node.attrib["type"].replace("const", "").replace(" ", "")
        self.varname = node.attrib["varname"] if "varname" in node.attrib else None
        self.etype = node.attrib["etype"] if "etype" in node.attrib else None
        self.memwidth = node.attrib["memwidth"] if "memwidth" in node.attrib else None
        self.immwidth = node.attrib["immwidth"] if "immwidth" in node.attrib else None
        self.slots = self.get_number_of_slots()

    def __str__(self):
        s = self.ret
        if self.varname != None:
            s += f" {self.varname}"
        if self.etype != None:
            s += f" {self.etype}"
        if self.memwidth != None:
            s += f" {self.memwidth}"
        s += f" {self.slots}"
        return s


class IntrinsicsInstruction:
    @staticmethod
    def get_multivalued_key(node, key):
        tag = []
        for val in node.iter(key):
            tag.append(val.text)
        return tag

    @staticmethod
    def get_forms(node):
        form = ""
        if "sequence" in node.keys():
            asm = xed_iform = "seq"
        else:
            asm = xed_iform = ""

        for descr in node.iter("description"):
            if (
                "does not generate any instructions" in descr.text
                or "zero latency" in descr.text
            ):
                asm = xed_iform = "none"
        for i in node.iter("instruction"):
            asm = i.attrib["name"]
            if "xed" in i.attrib.keys():
                xed_iform = i.attrib["xed"]
            else:
                xed_iform = asm
            if "form" in i.attrib.keys():
                form = i.attrib["form"]
            else:
                form = ""
        return form, xed_iform, asm

    @staticmethod
    def get_parameters(node, type="parameter"):
        l = []
        for p in node.iter(type):
            l.append(Parameter(p))

        return l

    def get_as_row(self):
        return [
            self.intrinsics_name,
            self.asm,
            self.xed_iform,
            self.isa,
            self.cpuids,
            self.form,
            self.ret,
            self.args,
        ]

    def get_operation(self, node):
        for op in node.iter("operation"):
            self.operation = op.text.replace("\t", "").split("\n")
            if self.operation[0] == "":
                self.operation = self.op[1:]

    def evaluate_operation(self, operation):
        self.operation = operation
        if (self.operation == []) or (
            (not "Load" in self.category) and (not "Swizzle" in self.category)
        ):
            return
        oparser = OParser.OParser(self)
        # TODO: TESTING:
        if self.intrinsics_name != "_mm256_loadu_ps":
            return
        self.input_vals, self.output_vals = oparser.evaluate()

    def __init__(self, node, operation):
        self.intrinsics_name = node.attrib["name"]
        self.ret = self.get_parameters(node, "return")
        self.args = self.get_parameters(node, "parameter")
        self.isa = node.attrib["tech"].replace("-", "")
        self.form, self.xed_iform, self.asm = self.get_forms(node)
        self.cpuids = self.get_multivalued_key(node, "CPUID")
        self.category = self.get_multivalued_key(node, "category")
        self.evaluate_operation(operation)

    def __str__(self):
        s = f"{self.intrinsics_name}:\n"
        s += f"\t{self.xed_iform}\n"
        if "Load" in self.category:
            s += "\tMEM:\n"
            for arg in self.args:
                s += f"\t\t{arg}\n"
        return s


class IntrinsicsParser:
    csv_sep = "|"
    url = "https://software.intel.com/sites/landingpage/IntrinsicsGuide/files/data-latest.xml"
    intrinsics_file = "intel_intrinsics_info.xml"
    supported_isas = [
        "BASE",
        "MMX",
        "SSE",
        "SSE2",
        "SSE3",
        "SSSE3",
        "SSE4.1",
        "SSE4.2",
        "AVX",
        "AVX2",
        "AVX512",
    ]

    @staticmethod
    def is_in_isa(isa, min_isa):
        if not (min_isa in IntrinsicsParser.supported_isas):
            if not (min_isa.upper() in IntrinsicsParser.supported_isas):
                print("Error: wrong ISA")
                sys.exit(1)
            else:
                min_isa = min_isa.upper()

        if min_isa == "all":
            return True
        if isa not in IntrinsicsParser.supported_isas:
            return False
        return IntrinsicsParser.supported_isas.index(
            isa
        ) <= IntrinsicsParser.supported_isas.index(min_isa)

    @staticmethod
    def get_root(intrinsics_file, download_file):
        f = pathlib.Path(intrinsics_file)
        if f.exists():
            root = ET.parse(intrinsics_file)
        else:
            resp = requests.get(IntrinsicsParser.url)
            root = ET.fromstring(resp.content)
            if download_file != "":
                with open(download_file, "wb") as file:
                    file.write(resp.content)
            else:
                with open(intrinsics_file, "wb") as file:
                    file.write(resp.content)
        return root

    @classmethod
    def get_list_intrinsics(cls, min_isa, download_file="", category=["all"]):
        root = cls.get_root(cls.intrinsics_file, download_file)
        intrinsics_list = []
        with open("operation_python.json") as f:
            operations_list = json.load(f)

        # Convert to dict
        d_operations = {}
        for operation in operations_list:
            if (
                not "Load" in operation["category"]
                and not "Swizzle" in operation["category"]
            ):
                continue
            print(operation["name"])
            d_operations[operation["name"]] = {
                "operation": operation["op"],
            }

        etypes = []
        for list_node in root.iter("intrinsics_list"):
            # This approach is ad-hoc: all children in intrinsics_list are intrinsics
            it = len(list(list_node))
            with tqdm(total=it) as pbar:
                for intr_node in list_node.iter("intrinsic"):
                    pbar.update(1)
                    operation = []
                    try:
                        operation = d_operations[intr_node.attrib["name"]]
                    except KeyError:
                        pass

                    ins = IntrinsicsInstruction(intr_node, operation)
                    etypes.append(ins.ret[0].ret)
                    for arg in ins.args:
                        etypes.append(arg.ret)
                    if not cls.is_in_isa(ins.isa, min_isa):
                        continue
                    if ("all" in category) or (category in ins.category):
                        intrinsics_list.append(ins)

        if DEBUG:
            import numpy as np

            etypes = list(filter(None.__ne__, etypes))
            print(np.unique(etypes))

        return intrinsics_list

    @classmethod
    def get_csv(cls, min_isa, download_file=""):
        root = cls.get_root(intrinsics_file, download_file)
        cols = ["intrinsics", "ASM", "XED_iform", "ISA", "CPUID", "form", "ret", "op"]
        df = pd.DataFrame(columns=cols)
        for list_node in root.iter("intrinsics_list"):
            # This approach is ad-hoc: all children in intrinsics_list are intrinsics
            it = len(list(list_node.getchildren()))
            with tqdm(total=it) as pbar:
                for intr_node in list_node.iter("intrinsic"):
                    pbar.update(1)
                    ins = IntrinsicsInstruction(intr_node)
                    if not cls.is_in_isa(ins.isa, min_isa):
                        continue
                    row = ins.get_as_row()
                    df = df.append(pd.DataFrame([row], columns=cols), ignore_index=True)
        return df


if __name__ == "__main__":
    download_file = ""
    if len(sys.argv) > 1:
        download_file = sys.argv[1]
    intrinsics_file = IntrinsicsParser.intrinsics_file.split("/")[-1]
    # df = IntrinsicsParser.get_csv("AVX2", download_file)
    # csv_file = "intrinsics.csv"
    # df.to_csv(csv_file, sep=IntrinsicsParser.csv_sep)
    l = IntrinsicsParser.get_list_intrinsics("AVX2", download_file, "Load")
    if DEBUG:
        for ins in l:
            print(ins)
