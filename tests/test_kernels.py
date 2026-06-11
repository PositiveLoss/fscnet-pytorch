import os
import unittest
from unittest import mock

import torch

from fscnet_pytorch.kernels import (
    _apply_inverse_rope_torch,
    _apply_rope_torch,
    activated_kernel_names,
    fused_global_layer_norm_pyptx,
    fused_rope_qk_norm_pyptx,
    make_progressive_targets_pyptx,
    pyptx_disabled,
    reset_activated_kernel_names,
)


class KernelFallbackTest(unittest.TestCase):
    def test_pyptx_is_opt_in(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(pyptx_disabled())
        with mock.patch.dict(os.environ, {"FSCNET_ENABLE_PYPTX": "1"}, clear=True):
            self.assertFalse(pyptx_disabled())
        with mock.patch.dict(
            os.environ, {"FSCNET_ENABLE_PYPTX": "1", "NO_PYPTX": "1"}, clear=True
        ):
            self.assertTrue(pyptx_disabled())

    def test_cpu_tensors_use_progressive_target_fallback(self) -> None:
        input_ri = torch.randn(1, 2, 9, 3)
        target_ri = torch.randn(1, 2, 9, 3)

        out = make_progressive_targets_pyptx(input_ri, target_ri, windows=(5, 1))

        self.assertIsNone(out)

    def test_global_layer_norm_kernel_fallback_preserves_manual_path(self) -> None:
        x = torch.randn(2, 4, 5, 3)
        weight = torch.ones(1, 4, 1, 1)
        bias = torch.zeros(1, 4, 1, 1)

        out = fused_global_layer_norm_pyptx(x, weight, bias, eps=1.0e-5)

        self.assertIsNone(out)

    def test_rope_qk_kernel_fallback_on_cpu(self) -> None:
        q = torch.randn(2, 2, 5, 8)
        k = torch.randn(2, 2, 5, 8)

        out = fused_rope_qk_norm_pyptx(q, k)

        self.assertIsNone(out)

    def test_rope_torch_inverse_roundtrip(self) -> None:
        x = torch.randn(2, 3, 5, 8)

        y = _apply_inverse_rope_torch(_apply_rope_torch(x))

        torch.testing.assert_close(y, x)

    def test_activated_kernel_names_reset(self) -> None:
        reset_activated_kernel_names()

        self.assertEqual(activated_kernel_names(), ())


if __name__ == "__main__":
    unittest.main()
