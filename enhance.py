from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lowlight import enhance_adaptive


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive OpenCV low-light enhancement")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("output", type=Path, help="Output image path")
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Cannot read image: {args.input}")
    result = enhance_adaptive(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        parser.error(f"Cannot write image: {args.output}")


if __name__ == "__main__":
    main()
