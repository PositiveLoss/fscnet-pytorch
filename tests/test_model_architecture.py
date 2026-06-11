import unittest

import torch

from fscnet_pytorch.model import (
    DepthwiseConv2dHead,
    FSCNet,
    FSCNetConfig,
    IntraFrameRNN,
    TFFFCBlock,
    count_parameters,
    cws_merge,
    cws_split,
)


class ModelArchitectureTest(unittest.TestCase):
    def test_cws_split_merge_roundtrip(self) -> None:
        x = torch.randn(2, 2, 10, 4)

        split, pad = cws_split(x, subbands=3)
        merged = cws_merge(split, subbands=3, original_freq=x.shape[2])

        self.assertEqual(pad, 2)
        self.assertEqual(split.shape, (2, 6, 4, 4))
        torch.testing.assert_close(merged, x)

    def test_intra_frame_rnn_preserves_shape(self) -> None:
        module = IntraFrameRNN(channels=8, hidden=5, dropout=0.0)
        x = torch.randn(2, 8, 7, 3)

        y = module(x)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(module.unfold_channels, 24)

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
        self.assertTrue(all(isinstance(head, DepthwiseConv2dHead) for head in model.stage_heads))
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


if __name__ == "__main__":
    unittest.main()
