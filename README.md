# OpenCV Adaptive Low-Light Lab

An educational extension built with the OpenCV image-processing API. It compares global histogram equalization, fixed CLAHE, and an adaptive pipeline for low-light images. The extension is maintained separately from the upstream OpenCV repository.

## Upstream project

- GitHub: https://github.com/opencv/opencv
- Main image-processing API: OpenCV `imgproc`
- CLAHE API used as the baseline: `cv2.createCLAHE`
- This project uses the installed OpenCV Python bindings and does not patch OpenCV's C++ core.

## Method

The adaptive pipeline works on LAB luminance so it does not independently stretch the B, G, and R channels. It estimates high-frequency noise on low-gradient pixels, applies non-local means denoising, chooses a per-image luminance target from the measured input mean, solves for a gamma curve that approaches that target, then adds noise-adjusted CLAHE detail. A soft mask returns bright highlights toward their input luminance.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 evaluate.py
python3 enhance.py results/cases/astronaut_g1p7/input.png results/astronaut_enhanced.png
python3 -m unittest discover -s tests -v
```

`evaluate.py` creates nine paired synthetic cases from three sample images and three gamma levels. It writes per-case metrics, a summary table, an image comparison, a PSNR chart, an ablation table, full-resolution input/output images, and environment details under `results/`.

## Dataset and limits

The reference images are scikit-image v0.25.2 sample data (`astronaut.png`, `chelsea.png`, `coffee.png`). The low-light inputs are generated from the reference LAB L channel as `0.72 * (L/255)^gamma * 255 + GaussianNoise(sigma=2.5)` for gamma values 1.4, 1.7, and 2.0, with fixed seeds. This controlled paired experiment tests the code path and parameter behavior; it does not establish performance on real camera night scenes or replace a benchmark such as LOL.

The sample data originates from the scikit-image project: https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data. See `THIRD_PARTY_NOTICES.md` for attribution and license details.

## Server verification

The tests and evaluator were run on the GYM server with Python 3.14.7, OpenCV 5.0.0, NumPy 2.5.4, and an Intel Xeon Gold 6326 CPU. The server has RTX 4090 D GPUs, but this OpenCV build reports zero CUDA devices, so the reported image-processing timings are CPU timings. Exact results are recorded in `results/summary.csv` and `results/metrics.csv`.
