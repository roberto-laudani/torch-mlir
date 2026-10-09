# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torchao

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class TorchaoQuantizeAffinePerToken(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([1, 4, 16], torch.float32, True),
            ([1, 4], torch.float32, True),
            ([1, 4], torch.int32, True),
        ]
    )
    def forward(self, x, scale, zero_point):
        return torch.ops.torchao.quantize_affine.default(
            x, [1, 1, 16], scale.to(torch.bfloat16), zero_point, torch.int8
        )


@register_test_case(module_factory=lambda: TorchaoQuantizeAffinePerToken())
def TorchaoQuantizeAffinePerToken_basic(module, tu: TestUtils):
    scale = (tu.rand(1, 4) + 0.01).to(torch.bfloat16)
    zero_point = tu.randint(1, 4, low=-128, high=127).to(torch.int32)
    # Values on .5 ties of x * (1 / scale), partly outside the int8 range.
    ties = tu.randint(1, 4, 16, low=-150, high=150) + 0.5
    module.forward(ties / (1 / scale).float().unsqueeze(-1), scale.float(), zero_point)


# ==============================================================================


class TorchaoDequantizeAffinePerGroup(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.int8, True),
            ([4, 2], torch.float32, True),
            ([4, 2], torch.float32, True),
        ]
    )
    def forward(self, x, scale, zero_point):
        return torch.ops.torchao.dequantize_affine.default(
            x, [1, 8], scale.to(torch.bfloat16), zero_point, torch.int8, -128, 127
        )


@register_test_case(module_factory=lambda: TorchaoDequantizeAffinePerGroup())
def TorchaoDequantizeAffinePerGroup_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 16, low=-128, high=128).to(torch.int8),
        tu.rand(4, 2) + 0.01,
        tu.rand(4, 2, low=-10, high=10),
    )
