from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from lowlight import (
    enhance_adaptive,
    enhance_clahe,
    enhance_histogram_equalization,
    make_synthetic_lowlight,
    mean_luminance,
    psnr,
    ssim,
)


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "data" / "source"
RESULT_DIR = ROOT / "results"
GAMMAS = (1.4, 1.7, 2.0)
METHODS = (
    ("Input", lambda image: image),
    ("Global HE", enhance_histogram_equalization),
    ("Fixed CLAHE", enhance_clahe),
    ("Adaptive", enhance_adaptive),
)
ABLATIONS = (
    ("Without denoise", {"use_denoise": False}),
    ("Without local contrast", {"use_local_contrast": False}),
    ("Without highlight", {"use_highlight": False}),
)


def comparison_montage(reference: np.ndarray, lowlight: np.ndarray) -> np.ndarray:
    outputs = [fn(lowlight) for _, fn in METHODS]
    panels = [lowlight, *outputs[1:], reference]
    labels = ["Input", "Global HE", "Fixed CLAHE", "Adaptive", "Reference"]
    panel_w, panel_h, label_h = 256, 256, 34
    canvas = np.full((panel_h + label_h, panel_w * len(panels), 3), 245, np.uint8)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        resized = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        x0 = index * panel_w
        canvas[label_h:, x0 : x0 + panel_w] = resized
        cv2.putText(canvas, label, (x0 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def draw_metric_chart(summary: dict[str, dict[str, float]]) -> np.ndarray:
    width, height = 1500, 760
    canvas = np.full((height, width, 3), 250, np.uint8)
    plot_x0, plot_y0, plot_w, plot_h = 110, 100, 1330, 550
    methods = list(summary)
    values = [summary[name]["psnr_mean"] for name in methods]
    minimum, maximum = 0.0, 30.0
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (90, 90, 90), 1)
    for tick in range(7):
        y = plot_y0 + plot_h - round(tick * plot_h / 6)
        value = minimum + (maximum - minimum) * tick / 6
        cv2.line(canvas, (plot_x0, y), (plot_x0 + plot_w, y), (220, 220, 220), 1)
        cv2.putText(canvas, f"{value:.0f}", (36, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), 1, cv2.LINE_AA)
    colors = [(140, 140, 140), (95, 160, 210), (80, 180, 100), (50, 80, 210)]
    group_w = plot_w // len(methods)
    for index, (name, value, color) in enumerate(zip(methods, values, colors)):
        bar_h = round((value - minimum) / (maximum - minimum) * plot_h)
        x0 = plot_x0 + index * group_w + 65
        x1 = plot_x0 + (index + 1) * group_w - 65
        y0 = plot_y0 + plot_h - bar_h
        cv2.rectangle(canvas, (x0, y0), (x1, plot_y0 + plot_h), color, -1)
        error_h = round(summary[name]["psnr_std"] / (maximum - minimum) * plot_h)
        xc = (x0 + x1) // 2
        cv2.line(canvas, (xc, y0 - error_h), (xc, y0 + error_h), (40, 40, 40), 2)
        for error_y in (y0 - error_h, y0 + error_h):
            cv2.line(canvas, (xc - 14, error_y), (xc + 14, error_y), (40, 40, 40), 2)
        cv2.putText(canvas, f"{value:.2f}", (xc - 40, y0 - error_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, name, (x0 - 6, plot_y0 + plot_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PSNR (dB), mean +/- sample SD over 9 synthetic cases", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(SOURCE_DIR.glob("*.*"))
    if not source_paths:
        raise SystemExit(f"No source images found in {SOURCE_DIR}")

    records: list[dict[str, object]] = []
    ablation_records: list[dict[str, object]] = []
    runtimes: dict[str, list[float]] = {name: [] for name, _ in METHODS}
    astronaut_example: tuple[np.ndarray, np.ndarray] | None = None

    for image_index, path in enumerate(source_paths):
        reference = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if reference is None:
            raise SystemExit(f"Could not read {path}")
        for gamma_index, gamma in enumerate(GAMMAS):
            seed = 2024 + image_index * 10 + gamma_index
            lowlight = make_synthetic_lowlight(reference, gamma, seed)
            case_dir = RESULT_DIR / "cases" / f"{path.stem}_g{str(gamma).replace('.', 'p')}"
            case_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(case_dir / "reference.png"), reference)
            cv2.imwrite(str(case_dir / "input.png"), lowlight)
            if path.stem == "astronaut" and gamma == 1.7:
                astronaut_example = (reference, lowlight)

            for method_name, method in METHODS:
                method(lowlight)
                start = time.perf_counter()
                repetitions = 5
                output = lowlight
                for _ in range(repetitions):
                    output = method(lowlight)
                elapsed_ms = (time.perf_counter() - start) * 1000.0 / repetitions
                runtimes[method_name].append(elapsed_ms)
                output_name = method_name.lower().replace(" ", "_")
                cv2.imwrite(str(case_dir / f"{output_name}.png"), output)
                records.append(
                    {
                        "image": path.name,
                        "gamma": gamma,
                        "seed": seed,
                        "method": method_name,
                        "psnr_db": psnr(reference, output),
                        "ssim": ssim(reference, output),
                        "mean_luminance": mean_luminance(output),
                        "luminance_error": abs(mean_luminance(reference) - mean_luminance(output)),
                        "runtime_ms": elapsed_ms,
                    }
                )

            for ablation_name, switches in ABLATIONS:
                output = enhance_adaptive(lowlight, **switches)
                ablation_records.append({
                    "image": path.name, "gamma": gamma, "method": ablation_name,
                    "psnr_db": psnr(reference, output), "ssim": ssim(reference, output),
                    "luminance_error": abs(mean_luminance(reference) - mean_luminance(output)),
                })

    summary: dict[str, dict[str, float]] = {}
    for method_name, _ in METHODS:
        method_rows = [row for row in records if row["method"] == method_name]
        psnr_values = np.array([float(row["psnr_db"]) for row in method_rows])
        ssim_values = np.array([float(row["ssim"]) for row in method_rows])
        brightness_values = np.array([float(row["luminance_error"]) for row in method_rows])
        summary[method_name] = {
            "psnr_mean": float(psnr_values.mean()),
            "psnr_std": float(psnr_values.std(ddof=1)),
            "ssim_mean": float(ssim_values.mean()),
            "ssim_std": float(ssim_values.std(ddof=1)),
            "luminance_error_mean": float(brightness_values.mean()),
            "runtime_ms_mean": float(np.mean(runtimes[method_name])),
        }

    with (RESULT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with (RESULT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["method", *next(iter(summary.values())).keys()]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method_name, values in summary.items():
            writer.writerow({"method": method_name, **values})

    with (RESULT_DIR / "ablations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ablation_records[0]))
        writer.writeheader()
        writer.writerows(ablation_records)

    cpu = platform.processor() or platform.machine()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    environment = {
        "python": platform.python_version(), "opencv": cv2.__version__,
        "numpy": np.__version__, "cpu": cpu,
        "opencv_threads": cv2.getNumThreads(), "timing_repetitions": 5,
        "cuda_devices": cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0,
    }
    (RESULT_DIR / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    if astronaut_example is not None:
        montage = comparison_montage(*astronaut_example)
        cv2.imwrite(str(RESULT_DIR / "comparison_astronaut.png"), montage)
    chart = draw_metric_chart(summary)
    cv2.imwrite(str(RESULT_DIR / "psnr_summary.png"), chart)

    print(f"Evaluated {len(source_paths)} images x {len(GAMMAS)} exposure levels = {len(source_paths) * len(GAMMAS)} paired cases")
    print("method,psnr_mean,psnr_std,ssim_mean,ssim_std,luminance_error_mean,runtime_ms_mean")
    for method_name, values in summary.items():
        print(
            f"{method_name},{values['psnr_mean']:.4f},{values['psnr_std']:.4f},"
            f"{values['ssim_mean']:.5f},{values['ssim_std']:.5f},"
            f"{values['luminance_error_mean']:.5f},{values['runtime_ms_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
