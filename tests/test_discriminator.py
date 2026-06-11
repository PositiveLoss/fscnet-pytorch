import unittest

import torch

from fscnet_pytorch.discriminator import (
    DiscriminatorScaleOutput,
    MultiScaleDiscriminator,
    discriminator_lsgan_loss,
    generator_lsgan_fm_loss,
)


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

    def test_lsgan_losses_use_real_and_fake_pairs(self) -> None:
        disc = MultiScaleDiscriminator(num_scales=1, base_channels=4)
        real_waveform = torch.randn(2, 256)
        fake_waveform = torch.randn(2, 256)
        real_spec_ri = torch.randn(2, 2, 33, 9)
        fake_spec_ri = torch.randn(2, 2, 33, 9)

        d_loss = discriminator_lsgan_loss(
            disc, real_waveform, real_spec_ri, fake_waveform, fake_spec_ri
        )
        adv, fm = generator_lsgan_fm_loss(
            disc, real_waveform, real_spec_ri, fake_waveform, fake_spec_ri
        )

        self.assertEqual(d_loss.ndim, 0)
        self.assertEqual(adv.ndim, 0)
        self.assertEqual(fm.ndim, 0)

    def test_losses_average_wave_and_spectrogram_modalities(self) -> None:
        disc = MultiScaleDiscriminator(num_scales=2, base_channels=4)
        real_waveform = torch.randn(2, 512)
        fake_waveform = torch.randn(2, 512)
        real_spec_ri = torch.randn(2, 2, 65, 17)
        fake_spec_ri = torch.randn(2, 2, 65, 17)

        outputs = disc(fake_waveform, fake_spec_ri)
        score_count = sum(
            int(output.wave_score is not None) + int(output.spec_score is not None)
            for output in outputs
        )
        d_loss = discriminator_lsgan_loss(
            disc, real_waveform, real_spec_ri, fake_waveform, fake_spec_ri
        )
        adv, fm = generator_lsgan_fm_loss(
            disc, real_waveform, real_spec_ri, fake_waveform, fake_spec_ri
        )

        self.assertEqual(score_count, 4)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(d_loss.ndim, 0)
        self.assertEqual(adv.ndim, 0)
        self.assertEqual(fm.ndim, 0)


if __name__ == "__main__":
    unittest.main()
