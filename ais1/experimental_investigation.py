"""Reproducible experiments for the five face-detection hypotheses."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from mediapipe_face_detection import create_detector, detect_faces, detect_haar


Box = tuple[float, float, float, float]


def load_annotations(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        annotations = json.load(file)
    if not isinstance(annotations, list):
        raise ValueError("Annotations must be a JSON list.")
    for item in annotations:
        if not {"image", "boxes"} <= item.keys():
            raise ValueError("Each annotation needs image and boxes fields.")
    return annotations


def as_box(box) -> Box:
    return tuple(float(value) for value in box)


def media_boxes(frame, detector) -> list[Box]:
    result = detect_faces(frame, detector)
    return [
        as_box(
            (
                detection.bounding_box.origin_x,
                detection.bounding_box.origin_y,
                detection.bounding_box.width,
                detection.bounding_box.height,
            )
        )
        for detection in result.detections
    ]


def haar_boxes(frame, cascade_dir: Path) -> list[Box]:
    return [as_box(face) for face in detect_haar(frame, cascade_dir)]


def iou(first: Box, second: Box) -> float:
    fx, fy, fw, fh = first
    sx, sy, sw, sh = second
    left = max(fx, sx)
    top = max(fy, sy)
    right = min(fx + fw, sx + sw)
    bottom = min(fy + fh, sy + sh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = fw * fh + sw * sh - intersection
    return intersection / union if union else 0.0


def score(predicted: list[Box], actual: list[Box], threshold: float) -> tuple[int, int, int]:
    matches = [
        (iou(prediction, truth), prediction_index, truth_index)
        for prediction_index, prediction in enumerate(predicted)
        for truth_index, truth in enumerate(actual)
        if iou(prediction, truth) >= threshold
    ]
    matches.sort(reverse=True)
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    true_positives = 0
    for _, prediction_index, truth_index in matches:
        if prediction_index not in used_predictions and truth_index not in used_truth:
            used_predictions.add(prediction_index)
            used_truth.add(truth_index)
            true_positives += 1
    return true_positives, len(actual) - true_positives, len(predicted) - true_positives


def darken(frame, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def equalize(frame) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)


def add_noise(frame, sigma: float, seed: int) -> np.ndarray:
    if sigma == 0:
        return frame.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def resize_case(frame, boxes: list[Box], factor: float) -> tuple[np.ndarray, list[Box]]:
    height, width = frame.shape[:2]
    resized = cv2.resize(frame, (round(width * factor), round(height * factor)))
    resized_boxes = [
        (x * factor, y * factor, box_width * factor, box_height * factor)
        for x, y, box_width, box_height in boxes
    ]
    return resized, resized_boxes


def run_detector(name, frame, detector, cascade_dir) -> list[Box]:
    if name == "mediapipe":
        return media_boxes(frame, detector)
    return haar_boxes(frame, cascade_dir)


def benchmark(
    detector_name: str,
    frame,
    detector,
    cascade_dir: Path,
    repetitions: int,
) -> tuple[float, float]:
    run_detector(detector_name, frame, detector, cascade_dir)
    runtimes = []
    for _ in range(repetitions):
        start = time.perf_counter()
        run_detector(detector_name, frame, detector, cascade_dir)
        runtimes.append((time.perf_counter() - start) * 1000)
    return statistics.median(runtimes), statistics.mean(runtimes)


def evaluate(
    rows: list[dict],
    image_name: str,
    condition: str,
    frame,
    actual: list[Box],
    detector,
    cascade_dir: Path,
    threshold: float,
) -> None:
    for detector_name in ("mediapipe", "haar"):
        predicted = run_detector(detector_name, frame, detector, cascade_dir)
        true_positives, missed, false_positives = score(
            predicted, actual, threshold
        )
        rows.append(
            {
                "hypothesis": condition.split(":", 1)[0],
                "image": image_name,
                "condition": condition,
                "detector": detector_name,
                "true_positives": true_positives,
                "missed": missed,
                "false_positives": false_positives,
            }
        )


def run_experiments(
    annotations: list[dict],
    model_path: Path,
    cascade_dir: Path,
    iou_threshold: float,
    noise_seed: int,
    repetitions: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    timing_rows: list[dict] = []
    with create_detector(model_path, 0.5) as detector:
        for item in annotations:
            image_name = item["image"]
            frame = cv2.imread(str(Path(image_name)))
            if frame is None:
                raise FileNotFoundError(f"Could not read image: {image_name}")
            actual = [as_box(box) for box in item["boxes"]]

            for factor in (1.0, 0.6, 0.3):
                dark = darken(frame, factor)
                evaluate(rows, image_name, f"H1:dark_{factor:.1f}", dark, actual,
                         detector, cascade_dir, iou_threshold)
                evaluate(rows, image_name, f"H1:dark_{factor:.1f}_equalized",
                         equalize(dark), actual, detector, cascade_dir, iou_threshold)

            for factor in (1.0, 0.75, 0.5, 0.25):
                resized, resized_actual = resize_case(frame, actual, factor)
                evaluate(rows, image_name, f"H3:scale_{factor:.2f}", resized,
                         resized_actual, detector, cascade_dir, iou_threshold)
                for detector_name in ("mediapipe", "haar"):
                    median_ms, mean_ms = benchmark(
                        detector_name, resized, detector, cascade_dir, repetitions
                    )
                    timing_rows.append(
                        {
                            "hypothesis": "H3",
                            "image": image_name,
                            "condition": f"scale_{factor:.2f}",
                            "detector": detector_name,
                            "repetitions": repetitions,
                            "median_ms": median_ms,
                            "mean_ms": mean_ms,
                        }
                    )

            for sigma in (0.0, 10.0, 25.0):
                noisy = add_noise(frame, sigma, noise_seed)
                for filter_name, filtered in (
                    ("none", noisy),
                    ("gaussian", cv2.GaussianBlur(noisy, (5, 5), 0)),
                    ("bilateral", cv2.bilateralFilter(noisy, 9, 75, 75)),
                ):
                    evaluate(rows, image_name, f"H4:sigma_{sigma:.0f}_{filter_name}",
                             filtered, actual, detector, cascade_dir, iou_threshold)

            pose = item.get("pose", "unlabelled")
            if pose != "unlabelled":
                evaluate(rows, image_name, f"H2:pose_{pose}", frame, actual,
                         detector, cascade_dir, iou_threshold)

            for detector_name in ("mediapipe", "haar"):
                median_ms, mean_ms = benchmark(
                    detector_name, frame, detector, cascade_dir, repetitions
                )
                timing_rows.append({
                    "hypothesis": "H5",
                    "image": image_name,
                    "condition": "baseline",
                    "detector": detector_name,
                    "repetitions": repetitions,
                    "median_ms": median_ms,
                    "mean_ms": mean_ms,
                })
    return rows, timing_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=project_dir / "blaze_face_short_range.tflite")
    parser.add_argument("--cascade-dir", type=Path, default=project_dir / "cascades")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, timing_rows = run_experiments(
        load_annotations(args.annotations),
        args.model,
        args.cascade_dir,
        args.iou_threshold,
        args.noise_seed,
        args.repetitions,
    )
    write_csv(args.output_dir / "detection_results.csv", rows)
    write_csv(args.output_dir / "timing_results.csv", timing_rows)
    print(f"Wrote {len(rows)} detection rows and {len(timing_rows)} timing rows.")


if __name__ == "__main__":
    main()
