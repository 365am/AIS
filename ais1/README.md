# Viola-Jones Face Detection

This project demonstrates automatic face detection with OpenCV's
Viola-Jones method and Haar cascade classifiers.

The script can process:

- live frames from a web camera;
- a still image such as `oscar.jpg`.

For every input image, the script converts the OpenCV BGR image to grayscale
with `cv2.cvtColor()` and calls `detectMultiScale()`. Detected faces are
returned as bounding boxes in the form `(x, y, w, h)` and displayed with
rectangles.

## Requirements

- Python 3
- OpenCV for Python (`cv2`)
- Two Haar cascade XML files:
  - `haarcascade_frontalface_default.xml`
  - `haarcascade_profileface.xml`

The script uses OpenCV's default cascade directory by default. If the XML
files are not included in that installation, download them from the OpenCV
repository and place both files in the same folder.

## Running the script

Open PowerShell in the project directory.

### Webcam

```powershell
.\Scripts\python.exe viola_jones.py --cascade-dir "C:\path\to\cascades"
```

The default camera index is `0`. To use another camera:

```powershell
.\Scripts\python.exe viola_jones.py --camera 1 --cascade-dir "C:\path\to\cascades"
```

Press `q` or `Esc` to close the webcam window.

### Image

Use the `--image` option to process a still image:

```powershell
.\Scripts\python.exe viola_jones.py `
    --image "C:\path\to\oscar.jpg" `
    --cascade-dir "C:\path\to\cascades"
```

Press any key while the image window is focused to close it.

## Command-line options

| Option | Description | Default |
|---|---|---|
| `--image PATH` | Process an image instead of the webcam | Webcam mode |
| `--camera INDEX` | Select the webcam index | `0` |
| `--cascade-dir PATH` | Directory containing both Haar XML files | OpenCV's cascade directory |

View all options with:

```powershell
.\Scripts\python.exe viola_jones.py --help
```

## Detection versus tracking

In webcam mode, the classifiers run independently on every frame. This is
repeated object detection, not object or person tracking. The script does not
assign persistent identities to faces or predict their positions between
frames.

## Main functions

- `load_classifiers()` loads and validates the Haar cascade files.
- `detect_faces()` converts a frame to grayscale and detects frontal and
  profile faces.
- `draw_faces()` draws the returned bounding boxes and classifier labels.
- `show_webcam()` reads and processes webcam frames.
- `show_image()` processes one image.
- `main()` parses command-line arguments and selects the input mode.

## MediaPipe Face Detector

The MediaPipe script uses the BlazeFace short-range CNN model. Download the
model once in the project directory:

```powershell
curl.exe -L -o blaze_face_short_range.tflite `
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
```

Run MediaPipe on `oscar.jpg` and compare it with the frontal Haar cascade:

```powershell
.\Scripts\python.exe mediapipe_face_detection.py `
    --image .\oscar.jpg --compare-haar
```

Run it on webcam frames:

```powershell
.\Scripts\python.exe mediapipe_face_detection.py --compare-haar
```

The comparison window shows MediaPipe on the left and Haar on the right.
MediaPipe draws a confidence score and six keypoints; press `q` or `Esc` to
close webcam mode.

MediaPipe learns features from training data with a CNN, whereas Viola-Jones
uses hand-designed Haar-like features and a cascade of classifiers. BlazeFace
usually handles moderate changes in pose and lighting more robustly and
provides keypoints, while Haar is simpler and can be faster on small images.
The actual comparison should report the face count and note missed or extra
detections observed in the image/webcam experiment.

## Interpretation

For a low-cost embedded device intended to detect people at short range, I
would select the MediaPipe BlazeFace detector. Although its CNN requires more
computation than a Haar cascade, my observations suggest that it gives more
reliable detections when the face is not perfectly frontal or the lighting
changes, so the reduction in missed detections is worth the extra cost. Haar
would be preferable only when the device is extremely resource-constrained
and the camera view can be tightly controlled.

Before deployment, I would run both detectors on a representative video
dataset using the actual target device and measure missed detections, false
positives, latency, and sustained frames per second across expected lighting
and face poses. This would verify that BlazeFace's reliability advantage is
large enough for the available compute and power budget.

## Experimental investigation

Create a copy of `annotations.example.json`, review the face boxes, and add
additional images and pose labels when available. The annotation boxes are
`[x, y, width, height]` in source-image pixels. Then run all five experiments:

```powershell
Copy-Item .\annotations.example.json .\annotations.json
.\Scripts\python.exe experimental_investigation.py `
    --annotations .\annotations.json `
    --output-dir .\results
```

The runner uses IoU >= 0.30 for a correct match and writes
`detection_results.csv` (true positives, missed faces, and false positives)
and `timing_results.csv` (warm-up excluded, 30 repeated calls by default).
The timing file includes every H3 resolution and the H5 baseline.
Noise uses a fixed seed. H2 is evaluated only for images with a pose label;
the runner never synthetically rotates a 2D image.
