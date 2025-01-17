#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from polygraphy.logger import G_LOGGER


def test_logger_severity():
    assert G_LOGGER.severity == G_LOGGER.module_severity.get()
    with G_LOGGER.verbosity():
        assert G_LOGGER.severity == G_LOGGER.CRITICAL


def test_debug_diff_tactics(poly_debug):
    status = poly_debug(["diff-tactics"])
    assert "debug diff-tactics is deprecated and will be removed" in status.stdout + status.stderr
