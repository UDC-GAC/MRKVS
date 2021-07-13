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

from sat.x86_sat.parse import *

regex = "|".join(
    [
        r"_mm256_set_epi32",
        r"_mm256_permutevar8x32_epi32",
    ]
)

index = Var("index", "__m256i")
intrinsics = parse_whitelist("sat/data-latest.xml", regex=regex)
globals().update(intrinsics)

print(
    check(
        _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1)
        == _mm256_permutevar8x32_epi32(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0), index)
    )
)

print(
    check(
        _mm256_set_epi32(6, 1, 1, 1, 1, 1, 1, 1)
        == _mm256_permutevar8x32_epi32(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0), index)
    )
)
