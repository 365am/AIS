"""Viola-Jones face detection using a webcam or a still image."""

import argparse
from pathlib import Path

import cv2


def load_cascades(cascade_dir: Path):
    cascade_dir = Path(cascade_dir)
    frontal = cv2.CascadeClassifier(
        str(cascade_dir / "haarcascade_frontalface_default.xml")
    )
    profile = cv2.CascadeClassifier(
        str(cascade_dir / "haarcascade_profileface.xml")
    )

    if frontal.empty() or profile.empty():
        raise FileNotFoundError(
            "Could not load the frontal/profile Haar cascades from "
            f"{cascade_dir}."
        )
    return frontal, profile


def detect_and_display(frame, frontal, profile):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    frontal_faces = frontal.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    profile_faces = profile.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, width, height in (*frontal_faces, *profile_faces):
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

    return frame, len(frontal_faces) + len(profile_faces)


def run_image(image_path, frontal, profile):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    frame, count = detect_and_display(frame, frontal, profile)
    print(f"Detected {count} face(s) in {image_path}.")
    cv2.imshow("Viola-Jones face detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(camera_index, frontal, profile):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam {camera_index}.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a webcam frame.")

            frame, _ = detect_and_display(frame, frontal, profile)
            cv2.imshow("Viola-Jones face detection (q/Esc to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        help="image to process, for example oscar.jpg; omit for webcam mode",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--cascade-dir",
        type=Path,
        default=project_dir / "cascades",
        help="directory containing the two face Haar XML files",
    )
    args = parser.parse_args()

    frontal, profile = load_cascades(args.cascade_dir)
    if args.image:
        run_image(args.image, frontal, profile)
    else:
        run_camera(args.camera, frontal, profile)


if __name__ == "__main__":
    main()
