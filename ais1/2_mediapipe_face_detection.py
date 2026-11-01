"""MediaPipe BlazeFace detection for images and webcam frames."""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp


def create_detector(model_path: Path, confidence: float):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Download "
            "blaze_face_short_range.tflite first."
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=confidence,
    )
    return mp.tasks.vision.FaceDetector.create_from_options(options)


def detect_faces(frame, detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return detector.detect(image)


def draw_detections(frame, result):
    height, width = frame.shape[:2]
    for detection in result.detections:
        box = detection.bounding_box
        x1 = max(0, box.origin_x)
        y1 = max(0, box.origin_y)
        x2 = min(width - 1, x1 + box.width)
        y2 = min(height - 1, y1 + box.height)
        score = detection.categories[0].score

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"MediaPipe {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        for keypoint in detection.keypoints:
            point = (round(keypoint.x * width), round(keypoint.y * height))
            cv2.circle(frame, point, 3, (255, 0, 0), -1)

    return frame


def detect_haar(frame, cascade_dir: Path):
    cascade = cv2.CascadeClassifier(
        str(cascade_dir / "haarcascade_frontalface_default.xml")
    )
    if cascade.empty():
        raise FileNotFoundError(f"Could not load Haar cascade from {cascade_dir}.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )


def draw_haar(frame, faces):
    for x, y, width, height in faces:
        cv2.rectangle(
            frame, (x, y), (x + width, y + height), (0, 0, 255), 2
        )
        cv2.putText(
            frame,
            "Haar",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return frame


def process_frame(frame, detector, cascade_dir=None):
    result = detect_faces(frame, detector)
    media_frame = draw_detections(frame.copy(), result)
    if cascade_dir is None:
        return media_frame, len(result.detections)

    haar_faces = detect_haar(frame, cascade_dir)
    haar_frame = draw_haar(frame.copy(), haar_faces)
    comparison = cv2.hconcat((media_frame, haar_frame))
    return comparison, len(result.detections), len(haar_faces)


def run_image(image_path, detector, cascade_dir=None):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    processed = process_frame(frame, detector, cascade_dir)
    if cascade_dir is None:
        output, media_count = processed
        print(f"MediaPipe detected {media_count} face(s) in {image_path}.")
    else:
        output, media_count, haar_count = processed
        print(
            f"MediaPipe detected {media_count} face(s); "
            f"Haar detected {haar_count} face(s) in {image_path}."
        )

    cv2.imshow("MediaPipe (left) / Haar (right)", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(camera_index, detector, cascade_dir=None):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam {camera_index}.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a webcam frame.")

            output = process_frame(frame, detector, cascade_dir)[0]
            cv2.imshow("MediaPipe (left) / Haar (right)", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="image to process; omit for webcam")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "blaze_face_short_range.tflite",
    )
    parser.add_argument(
        "--compare-haar",
        action="store_true",
        help="show MediaPipe and Haar results side by side",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    cascade_dir = project_dir / "cascades" if args.compare_haar else None
    with create_detector(args.model, args.confidence) as detector:
        if args.image:
            run_image(args.image, detector, cascade_dir)
        else:
            run_camera(args.camera, detector, cascade_dir)


if __name__ == "__main__":
    main()
