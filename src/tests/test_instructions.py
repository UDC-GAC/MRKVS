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

from ..instructions import (
    _mm_loadu_ps,
    _mm256_loadu_ps,
    _mm_load_ss,
    _mv_insert_mem_ps,
    _mv_blend_mem_ps,
    _mv256_blend_mem_ps,
    set_4_float_elements,
    set_8_float_elements,
)
from z3 import *
from ..sat.x86_sat.evaluate import check, Var
import pytest


@pytest.fixture
def load_4_elements_0():
    return _mm_loadu_ps(0)


def test_loadu():
    res, _ = check(set_4_float_elements(3, 2, 1, 0) == _mm_loadu_ps(0))
    assert res == sat


def test_256loadu():
    res, _ = check(set_8_float_elements(7, 6, 5, 4, 3, 2, 1, 0) == _mm256_loadu_ps(0))
    assert res == sat


def test_loadss():
    res, _ = check(set_4_float_elements(-1, -1, -1, 0) == _mm_load_ss(0))
    assert res == sat


def test_blend():
    __tmp = _mm_loadu_ps(0)
    i = Var("i", "int")
    res, _ = check(set_4_float_elements(6, 5, 1, 0) == _mv_blend_mem_ps(__tmp, 3, i))
    assert res == sat
