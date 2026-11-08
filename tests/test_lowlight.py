from __future__ import annotations

import unittest

import numpy as np

from lowlight import (
    enhance_adaptive,
    make_synthetic_lowlight,
    mean_luminance,
    psnr,
    ssim,
)


class LowLightTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[0:96, 0:128]
        self.reference = np.stack(
            [
                np.clip(80 + x * 1.1, 0, 255),
                np.clip(60 + y * 1.5, 0, 255),
                np.clip(40 + (x + y) * 0.8, 0, 255),
            ],
            axis=2,
        ).astype(np.uint8)

    def test_output_shape_and_dtype(self) -> None:
        lowlight = make_synthetic_lowlight(self.reference, 1.7, 2024)
        enhanced = enhance_adaptive(lowlight)
        self.assertEqual(enhanced.shape, self.reference.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreaterEqual(int(enhanced.min()), 0)
        self.assertLessEqual(int(enhanced.max()), 255)

    def test_adaptive_method_lifts_dark_input(self) -> None:
        lowlight = make_synthetic_lowlight(self.reference, 1.7, 2024)
        enhanced = enhance_adaptive(lowlight)
        self.assertGreater(mean_luminance(enhanced), mean_luminance(lowlight))

    def test_black_image_stays_finite(self) -> None:
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        enhanced = enhance_adaptive(black)
        self.assertTrue(np.isfinite(enhanced).all())
        self.assertEqual(int(enhanced.max()), 0)

    def test_metrics_identity(self) -> None:
        self.assertEqual(psnr(self.reference, self.reference), float("inf"))
        self.assertAlmostEqual(ssim(self.reference, self.reference), 1.0, places=5)

    def test_synthetic_degradation_is_reproducible(self) -> None:
        first = make_synthetic_lowlight(self.reference, 1.7, 123)
        second = make_synthetic_lowlight(self.reference, 1.7, 123)
        self.assertTrue(np.array_equal(first, second))

    def test_invalid_inputs_raise_clear_errors(self) -> None:
        for image in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((32, 32), np.uint8)):
            with self.assertRaises(ValueError):
                enhance_adaptive(image)


if __name__ == "__main__":
    unittest.main()
