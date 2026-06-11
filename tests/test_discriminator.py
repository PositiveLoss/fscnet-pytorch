import unittest

import torch

from fscnet_pytorch.discriminator import (
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
        for score, features in outputs:
            self.assertEqual(score.shape[0], waveform.shape[0])
            self.assertEqual(score.shape[1], 1)
            self.assertGreater(len(features), 0)

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


if __name__ == "__main__":
    unittest.main()
