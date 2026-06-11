import os
import unittest
from unittest import mock

import torch

from fscnet_pytorch.kernels import (
    _apply_inverse_rope_torch,
    _apply_rope_torch,
    _can_use_global_layer_norm_kernel,
    _can_use_progressive_target_kernel,
    _can_use_rope_qk_norm_kernel,
    _mark_kernel_active,
    activated_kernel_names,
    fused_global_layer_norm_pyptx,
    fused_rope_qk_norm_pyptx,
    make_progressive_targets_pyptx,
    pyptx_disabled,
    reset_activated_kernel_names,
)


class KernelFallbackTest(unittest.TestCase):
    def test_pyptx_is_opt_in(self) -> None:
        enabled_values = ("1", "true", "yes", "on")
        disabled_values = ("1", "true", "yes", "on", "anything")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(pyptx_disabled())
        for value in enabled_values:
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ, {"FSCNET_ENABLE_PYPTX": value}, clear=True
                ):
                    self.assertFalse(pyptx_disabled())
        for value in disabled_values:
            with self.subTest(no_pyptx=value):
                with mock.patch.dict(
                    os.environ,
                    {"FSCNET_ENABLE_PYPTX": "1", "NO_PYPTX": value},
                    clear=True,
                ):
                    self.assertTrue(pyptx_disabled())

    def test_progressive_target_kernel_eligibility_requires_safe_cuda_inputs(
        self,
    ) -> None:
        input_ri = torch.randn(1, 2, 9, 3)
        target_ri = torch.randn(1, 2, 9, 3)

        self.assertFalse(_can_use_progressive_target_kernel(input_ri, target_ri))
        self.assertFalse(
            _can_use_progressive_target_kernel(input_ri.requires_grad_(), target_ri)
        )
        self.assertFalse(
            _can_use_progressive_target_kernel(input_ri[:, :1], target_ri[:, :1])
        )

    def test_cpu_tensors_use_progressive_target_fallback(self) -> None:
        input_ri = torch.randn(1, 2, 9, 3)
        target_ri = torch.randn(1, 2, 9, 3)

        with mock.patch.dict(os.environ, {"FSCNET_ENABLE_PYPTX": "1"}, clear=True):
            out = make_progressive_targets_pyptx(input_ri, target_ri, windows=(5, 1))

        self.assertIsNone(out)

    def test_global_layer_norm_kernel_fallback_preserves_manual_path(self) -> None:
        x = torch.randn(2, 4, 5, 3)
        weight = torch.ones(1, 4, 1, 1)
        bias = torch.zeros(1, 4, 1, 1)

        with mock.patch.dict(
            os.environ,
            {"FSCNET_ENABLE_PYPTX": "1", "FSCNET_ENABLE_PYPTX_NORM": "1"},
            clear=True,
        ):
            out = fused_global_layer_norm_pyptx(x, weight, bias, eps=1.0e-5)

        self.assertIsNone(out)
        self.assertFalse(_can_use_global_layer_norm_kernel(x, weight, bias))
        self.assertFalse(
            _can_use_global_layer_norm_kernel(x, torch.ones(4), torch.zeros(4))
        )

    def test_rope_qk_kernel_fallback_on_cpu(self) -> None:
        q = torch.randn(2, 2, 5, 8)
        k = torch.randn(2, 2, 5, 8)

        with mock.patch.dict(
            os.environ,
            {"FSCNET_ENABLE_PYPTX": "1", "FSCNET_ENABLE_PYPTX_ROPE_QK": "1"},
            clear=True,
        ):
            out = fused_rope_qk_norm_pyptx(q, k)

        self.assertIsNone(out)
        self.assertFalse(_can_use_rope_qk_norm_kernel(q, k))
        self.assertFalse(_can_use_rope_qk_norm_kernel(q, k[..., :7]))

    def test_rope_torch_inverse_roundtrip(self) -> None:
        x = torch.randn(2, 3, 5, 8)

        rotated = _apply_rope_torch(x)
        y = _apply_inverse_rope_torch(rotated)

        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
        torch.testing.assert_close(y, x)

    def test_activated_kernel_names_reset(self) -> None:
        reset_activated_kernel_names()
        _mark_kernel_active("z_kernel")
        _mark_kernel_active("a_kernel")

        self.assertEqual(activated_kernel_names(), ("a_kernel", "z_kernel"))

        reset_activated_kernel_names()

        self.assertEqual(activated_kernel_names(), ())


if __name__ == "__main__":
    unittest.main()
