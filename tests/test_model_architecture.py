import unittest

import torch

from fscnet_pytorch.model import (
    DepthwiseConv2dHead,
    FSCNet,
    FSCNetConfig,
    FastFourierConv,
    GlobalLayerNorm,
    IntraFrameRNN,
    TFFFCBlock,
    count_parameters,
    cws_merge,
    cws_split,
)
from fscnet_pytorch.config import get_model_preset
from fscnet_pytorch.losses import make_progressive_targets


class ModelArchitectureTest(unittest.TestCase):
    def test_compact_preset_matches_article_constants(self) -> None:
        preset = get_model_preset("compact")

        self.assertEqual(preset.config.num_blocks, 5)
        self.assertEqual(preset.config.subbands, 3)
        self.assertEqual(preset.config.n_fft, 1536)
        self.assertEqual(preset.config.win_length, 1536)
        self.assertEqual(preset.config.hop_length, 768)
        self.assertEqual(preset.progressive_windows, (257, 65, 17, 5, 1))

    def test_cws_split_merge_roundtrip(self) -> None:
        x = torch.randn(2, 2, 10, 4)

        split, pad = cws_split(x, subbands=3)
        merged = cws_merge(split, subbands=3, original_freq=x.shape[2])

        self.assertEqual(pad, 2)
        self.assertEqual(split.shape, (2, 6, 4, 4))
        torch.testing.assert_close(merged, x)

    def test_cws_split_and_merge_validate_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected"):
            cws_split(torch.randn(2, 10, 4), subbands=3)

        with self.assertRaisesRegex(ValueError, "not divisible"):
            cws_merge(torch.randn(2, 5, 4, 4), subbands=3, original_freq=10)

    def test_global_layer_norm_matches_manual_normalization(self) -> None:
        module = GlobalLayerNorm(channels=2)
        x = torch.randn(3, 2, 4, 5, requires_grad=True)
        module.weight.data.copy_(torch.tensor([[[[1.5]], [[0.5]]]]))
        module.bias.data.copy_(torch.tensor([[[[-0.25]], [[0.75]]]]))

        y = module(x)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        expected = (x - mean) * torch.rsqrt(var + module.eps)
        expected = expected * module.weight + module.bias

        torch.testing.assert_close(y, expected)
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.bias.grad)

    def test_intra_frame_rnn_preserves_shape(self) -> None:
        module = IntraFrameRNN(channels=8, hidden=5, dropout=0.0)
        x = torch.randn(2, 8, 7, 3)

        y = module(x)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(module.unfold_channels, 24)

    def test_ffc_matches_article_local_plus_global_equation(self) -> None:
        module = FastFourierConv(channels=8)
        x = torch.randn(2, 8, 7, 5)

        y = module(x)
        expected = module.local(x) + module.global_branch(x)

        self.assertFalse(hasattr(module, "l2l"))
        self.assertFalse(hasattr(module, "l2g"))
        self.assertFalse(hasattr(module, "g2l"))
        torch.testing.assert_close(y, expected)

    def test_tf_ffc_block_uses_paper_intra_frame_rnn(self) -> None:
        cfg = FSCNetConfig(channels=8, rnn_hidden=5, attention_heads=2)

        block = TFFFCBlock(cfg)

        self.assertIsInstance(block.intra_rnn, IntraFrameRNN)

    def test_fscnet_uses_ffc_out_and_dconv2d_heads(self) -> None:
        cfg = FSCNetConfig(
            n_fft=64,
            win_length=64,
            hop_length=32,
            channels=8,
            num_blocks=2,
            rnn_hidden=5,
            attention_heads=2,
        )

        model = FSCNet(cfg)

        self.assertEqual(len(model.stage_heads), 2)
        self.assertTrue(hasattr(model, "ffc_out"))
        self.assertTrue(
            all(isinstance(head, DepthwiseConv2dHead) for head in model.stage_heads)
        )
        self.assertGreater(count_parameters(model), 0)

    def test_forward_returns_one_progressive_output_per_block(self) -> None:
        cfg = FSCNetConfig(
            n_fft=64,
            win_length=64,
            hop_length=32,
            channels=8,
            num_blocks=2,
            rnn_hidden=5,
            attention_heads=2,
        )
        model = FSCNet(cfg)
        wav = torch.randn(1, 256)

        outputs, input_ri = model(wav, return_all=True)

        self.assertEqual(len(outputs), cfg.num_blocks)
        self.assertEqual(input_ri.shape[1], 2)
        self.assertTrue(all(output.shape == input_ri.shape for output in outputs))
        self.assertEqual(model(wav).shape, input_ri.shape)

    def test_enhance_preserves_single_waveform_rank_and_length(self) -> None:
        cfg = FSCNetConfig(
            n_fft=64,
            win_length=64,
            hop_length=32,
            channels=8,
            num_blocks=1,
            rnn_hidden=5,
            attention_heads=2,
        )
        model = FSCNet(cfg)
        wav = torch.randn(256)

        enhanced = model.enhance(wav)

        self.assertEqual(enhanced.shape, wav.shape)

    def test_progressive_targets_match_frequency_smoothed_residuals(self) -> None:
        input_ri = torch.tensor([[[[1.0], [2.0], [3.0]], [[0.0], [0.0], [0.0]]]])
        target_ri = torch.tensor([[[[3.0], [4.0], [9.0]], [[0.0], [0.0], [0.0]]]])

        targets = make_progressive_targets(input_ri, target_ri, windows=(3, 1))

        self.assertEqual(len(targets), 2)
        expected_smoothed = torch.tensor(
            [[[[7.0 / 3.0], [16.0 / 3.0], [17.0 / 3.0]], [[0.0], [0.0], [0.0]]]]
        )
        torch.testing.assert_close(targets[0], expected_smoothed)
        torch.testing.assert_close(targets[-1], target_ri)


if __name__ == "__main__":
    unittest.main()
