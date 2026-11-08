"""Low-light enhancement experiments built on OpenCV's image-processing API."""

from __future__ import annotations

import cv2
import numpy as np


def _lab_luminance(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError("Expected a non-empty BGR image with three channels")
    if image_bgr.dtype != np.uint8:
        raise ValueError("Expected an 8-bit image")
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return lab, lab[:, :, 0]


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Estimate high-frequency noise from low-gradient pixels using MAD."""
    gray_f = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(gray_f, (0, 0), 1.0)
    residual = gray_f - smooth
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    flat = cv2.magnitude(gx, gy) < 12.0
    values = residual[flat] if np.count_nonzero(flat) >= 128 else residual.ravel()
    center = float(np.median(values))
    sigma = float(np.median(np.abs(values - center)) / 0.6745)
    return float(np.clip(sigma, 0.0, 20.0))


def _match_mean_gamma(luminance: np.ndarray, target_mean: float) -> tuple[np.ndarray, float]:
    normalized = luminance.astype(np.float32) / 255.0
    mean = float(normalized.mean())
    if mean <= 1e-8 or abs(mean - target_mean) < 1e-4:
        return normalized, 1.0

    low, high = 0.2, 2.5
    for _ in range(24):
        gamma = (low + high) / 2.0
        candidate_mean = float(np.power(normalized, gamma).mean())
        if candidate_mean > target_mean:
            low = gamma
        else:
            high = gamma

    gamma = (low + high) / 2.0
    return np.power(normalized, gamma), gamma


def enhance_histogram_equalization(image_bgr: np.ndarray) -> np.ndarray:
    lab, luminance = _lab_luminance(image_bgr)
    lab[:, :, 0] = cv2.equalizeHist(luminance)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    lab, luminance = _lab_luminance(image_bgr)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(luminance)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_adaptive(
    image_bgr: np.ndarray,
    *,
    use_denoise: bool = True,
    use_local_contrast: bool = True,
    use_highlight: bool = True,
) -> np.ndarray:
    """Denoise and lift dark luminance while limiting local contrast and highlights."""
    lab, luminance = _lab_luminance(image_bgr)
    luminance_f = luminance.astype(np.float32) / 255.0
    input_mean = float(luminance_f.mean())
    if input_mean <= 1e-8:
        return image_bgr.copy()
    target_mean = max(
        input_mean,
        min(0.58, max(input_mean * 1.7, input_mean + 0.10)),
    )

    noise_sigma = estimate_noise_sigma(luminance)
    denoise_h = float(np.clip(2.0 + 0.65 * noise_sigma, 2.0, 9.0))
    denoised = (
        cv2.fastNlMeansDenoising(
            luminance, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21
        )
        if use_denoise else luminance.copy()
    )
    corrected, _gamma = _match_mean_gamma(denoised, target_mean)

    adaptive_clip = float(
        np.clip(1.2 + 5.0 * (target_mean - input_mean), 1.2, 3.2)
        / (1.0 + noise_sigma / 30.0)
    )
    clahe = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=(8, 8))
    local_contrast = clahe.apply(denoised).astype(np.float32) / 255.0

    dark_weight = np.clip((target_mean - luminance_f) / max(target_mean, 1e-5), 0.0, 1.0)
    dark_weight = cv2.GaussianBlur(dark_weight, (0, 0), 7.0)
    noise_factor = float(np.clip(1.0 / (1.0 + noise_sigma / 10.0), 0.45, 1.0))
    local_weight = (0.15 + 0.30 * dark_weight) * noise_factor
    enhanced = corrected.copy()
    if use_local_contrast:
        enhanced += local_weight * (local_contrast - denoised.astype(np.float32) / 255.0)

    highlight = np.clip((luminance_f - 0.72) / 0.23, 0.0, 1.0)
    highlight = cv2.GaussianBlur(highlight, (0, 0), 5.0)
    if use_highlight:
        enhanced = enhanced * (1.0 - 0.85 * highlight) + luminance_f * (0.85 * highlight)

    lab[:, :, 0] = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def make_synthetic_lowlight(
    reference_bgr: np.ndarray,
    gamma: float,
    seed: int,
    exposure_scale: float = 0.72,
    noise_sigma: float = 2.5,
) -> np.ndarray:
    """Darken only the LAB luminance channel and add deterministic Gaussian noise."""
    lab, luminance = _lab_luminance(reference_bgr)
    normalized = luminance.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_sigma, size=luminance.shape).astype(np.float32)
    dark = exposure_scale * np.power(normalized, gamma) * 255.0 + noise
    lab[:, :, 0] = np.clip(dark, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10((255.0 * 255.0) / mse))


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute mean channel-wise SSIM with an 11x11 Gaussian window."""
    x = reference.astype(np.float32)
    y = candidate.astype(np.float32)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return float(np.mean(score))


def mean_luminance(image_bgr: np.ndarray) -> float:
    _, luminance = _lab_luminance(image_bgr)
    return float(luminance.mean() / 255.0)
