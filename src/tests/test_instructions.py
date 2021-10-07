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

from ..instructions import _mm_loadu_ps, set_4_float_elements, intrinsics
from z3 import *
from ..sat.x86_sat.evaluate import check


# import pytest


# @pytest.fixture
# def load_4_elements_0():
#     return _mm_loadu_ps(0)


def test_load_4():
    print(set_4_float_elements(3, 2, 1, 0) == _mm_loadu_ps(0))
    print(check(set_4_float_elements(3, 2, 1, 0) == _mm_loadu_ps(0)))

    res, _ = check(set_4_float_elements(3, 2, 1, 0) == _mm_loadu_ps(0))
    assert res == sat


def test_inserts():
    assert True


def test_blends():
    assert True
