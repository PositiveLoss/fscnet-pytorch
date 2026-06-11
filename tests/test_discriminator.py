import unittest
from typing import cast

import torch
from torch import nn

from fscnet_pytorch.discriminator import (
    DiscriminatorScaleOutput,
    MultiScaleDiscriminator,
    discriminator_lsgan_loss,
    generator_lsgan_fm_loss,
)


class MeanScoreDiscriminator(nn.Module):
    def forward(
        self, waveform: torch.Tensor, spec_ri: torch.Tensor
    ) -> list[DiscriminatorScaleOutput]:
        wave_score = waveform.mean().reshape(1, 1, 1)
        spec_score = spec_ri.mean().reshape(1, 1, 1, 1)
        return [
            DiscriminatorScaleOutput(
                wave_score=wave_score,
                spec_score=spec_score,
                wave_features=[wave_score.reshape(1, 1)],
                spec_features=[spec_score.reshape(1, 1)],
            )
        ]


def mean_score_discriminator() -> MultiScaleDiscriminator:
    return cast(MultiScaleDiscriminator, MeanScoreDiscriminator())


class DiscriminatorTest(unittest.TestCase):
    def test_multiscale_discriminator_accepts_waveform_spectrogram_pair(self) -> None:
        disc = MultiScaleDiscriminator(num_scales=2, base_channels=4)
        waveform = torch.randn(2, 256)
        spec_ri = torch.randn(2, 2, 33, 9)

        outputs = disc(waveform, spec_ri)

        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertIsInstance(output, DiscriminatorScaleOutput)
            self.assertEqual(output.wave_score.shape[0], waveform.shape[0])
            self.assertEqual(output.wave_score.shape[1], 1)
            self.assertEqual(output.spec_score.shape[0], waveform.shape[0])
            self.assertEqual(output.spec_score.shape[1], 1)
            self.assertGreater(len(output.wave_features), 0)
            self.assertGreater(len(output.spec_features), 0)

    def test_multiscale_discriminator_rejects_bad_input_rank(self) -> None:
        disc = MultiScaleDiscriminator(num_scales=1, base_channels=4)
        spec_ri = torch.randn(2, 2, 33, 9)

        with self.assertRaisesRegex(ValueError, "Expected waveform"):
            disc(torch.randn(2, 1, 1, 256), spec_ri)

        with self.assertRaisesRegex(ValueError, "Expected spectrogram"):
            disc(torch.randn(2, 256), torch.randn(2, 33, 9))

    def test_lsgan_losses_match_closed_form_average(self) -> None:
        disc = mean_score_discriminator()
        real_waveform = torch.ones(2, 4)
        fake_waveform = torch.zeros(2, 4)
        real_spec_ri = torch.full((2, 2, 3, 4), 2.0)
        fake_spec_ri = torch.full((2, 2, 3, 4), -1.0)

        d_loss = discriminator_lsgan_loss(
            disc, real_waveform, real_spec_ri, fake_waveform, fake_spec_ri
        )
        adv, fm = generator_lsgan_fm_loss(
            disc,
            real_waveform,
            real_spec_ri,
            fake_waveform,
            fake_spec_ri,
            fm_weight=0.5,
        )

        torch.testing.assert_close(d_loss, torch.tensor(1.0))
        torch.testing.assert_close(adv, torch.tensor(2.5))
        torch.testing.assert_close(fm, torch.tensor(1.0))

    def test_discriminator_loss_detaches_fake_inputs(self) -> None:
        disc = mean_score_discriminator()
        real_waveform = torch.ones(2, 4, requires_grad=True)
        fake_waveform = torch.zeros(2, 4, requires_grad=True)
        real_spec_ri = torch.full((2, 2, 3, 4), 2.0, requires_grad=True)
        fake_spec_ri = torch.full((2, 2, 3, 4), -1.0, requires_grad=True)

        d_loss = discriminator_lsgan_loss(
            disc, real_waveform, real_spec_ri, fake_waveform, fake_spec_ri
        )

        d_loss.backward()

        self.assertIsNotNone(real_waveform.grad)
        self.assertIsNotNone(real_spec_ri.grad)
        self.assertIsNone(fake_waveform.grad)
        self.assertIsNone(fake_spec_ri.grad)

    def test_generator_loss_backpropagates_through_fake_inputs_only(self) -> None:
        disc = mean_score_discriminator()
        real_waveform = torch.ones(2, 4, requires_grad=True)
        fake_waveform = torch.zeros(2, 4, requires_grad=True)
        real_spec_ri = torch.full((2, 2, 3, 4), 2.0, requires_grad=True)
        fake_spec_ri = torch.full((2, 2, 3, 4), -1.0, requires_grad=True)

        adv, fm = generator_lsgan_fm_loss(
            disc, real_waveform, real_spec_ri, fake_waveform, fake_spec_ri
        )

        (adv + fm).backward()

        self.assertIsNone(real_waveform.grad)
        self.assertIsNone(real_spec_ri.grad)
        self.assertIsNotNone(fake_waveform.grad)
        self.assertIsNotNone(fake_spec_ri.grad)


if __name__ == "__main__":
    unittest.main()
