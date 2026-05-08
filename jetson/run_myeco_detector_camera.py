#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import math
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from identity_manager import IdentityManager, IdentityState
from recognition_client import CppFaceRecognizer, DisabledFaceRecognizer


PERSON_MODEL_DIR = Path("person_detection_update") / "pedestrian_detection"


def parse_args():
    parser = argparse.ArgumentParser(description="Run detector-assisted MyECO on a live camera.")
    parser.add_argument("--camera", default="0", help="Camera index, video path, or GStreamer pipeline.")
    parser.add_argument("--video-input", type=Path, default=None, help="Optional video file path for replay. Overrides --camera when set.")
    parser.add_argument("--gstreamer", action="store_true", help="Open --camera as a GStreamer pipeline.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width for index cameras.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height for index cameras.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS and output FPS fallback.")
    parser.add_argument("--input-resize", default=None, help="Optional WIDTH,HEIGHT resize applied to each input frame before detection/tracking.")
    parser.add_argument("--input-rotate", choices=["none", "90cw", "90ccw", "180"], default="none", help="Optional rotation applied to each input frame before detection/tracking.")
    parser.add_argument("--tracker-name", default="eco", help="PyTracking tracker name.")
    parser.add_argument("--param", default="verified_otb936_run_update", help="PyTracking parameter name.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for CSV/video output.")
    parser.add_argument("--save-video", action="store_true", help="Save annotated camera output when --output-dir is set.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit. 0 means run until Q/Esc.")
    parser.add_argument("--window-name", default="MyECO Detector Camera", help="OpenCV display window name.")
    parser.add_argument("--detector-backend", choices=["auto", "cpp_ncnn", "ncnn", "onnx", "torchscript", "hog"], default="auto", help="Person detector backend.")
    parser.add_argument("--model-path", type=Path, default=None, help="Detector model path. For ncnn, this is the directory containing ssdperson10695.param/bin.")
    parser.add_argument("--detector-bin", type=Path, default=PERSON_MODEL_DIR / "build" / "detect_person_stdin", help="C++ NCNN stdin detector executable.")
    parser.add_argument("--detector-model-dir", type=Path, default=PERSON_MODEL_DIR, help="Directory containing ssdperson10695.param/bin for cpp_ncnn.")
    parser.add_argument("--input-size", type=int, default=300, help="Detector input size. NCNN SSD uses 300.")
    parser.add_argument("--detector-conf", type=float, default=0.75, help="Person detector confidence threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.45, help="NMS threshold for ONNX/TorchScript YOLO-style detectors.")
    parser.add_argument("--recognizer-backend", choices=["auto", "off", "cpp_ncnn"], default="auto", help="Face recognition backend for identity gating.")
    parser.add_argument("--recognizer-bin", type=Path, default=PERSON_MODEL_DIR / "build" / "face_recognize_stdin", help="C++ NCNN stdin face recognizer executable.")
    parser.add_argument("--recognizer-model-dir", type=Path, default=PERSON_MODEL_DIR / "models", help="Directory containing retinaface and mbv2facenet NCNN models.")
    parser.add_argument("--recognizer-cpu", action="store_true", help="Run the C++ face recognizer without NCNN Vulkan.")
    parser.add_argument("--recognizer-face-conf", type=float, default=0.80, help="RetinaFace confidence threshold.")
    parser.add_argument("--recognizer-face-nms", type=float, default=0.40, help="RetinaFace NMS threshold.")
    parser.add_argument("--recognize-interval", type=int, default=30, help="Verify identity every N stable tracking frames.")
    parser.add_argument("--recognize-fast-interval", type=int, default=3, help="Verify identity every N frames while tracking is uncertain.")
    parser.add_argument("--identity-match-threshold", type=float, default=0.35, help="Cosine similarity needed to verify target identity.")
    parser.add_argument("--identity-reject-threshold", type=float, default=0.12, help="Cosine similarity low enough to count as identity rejection.")
    parser.add_argument("--identity-reject-confirm-frames", type=int, default=2, help="Consecutive clear rejects before dropping the current target.")
    parser.add_argument("--identity-stale-frames", type=int, default=90, help="Frames after last match before identity becomes STALE.")
    parser.add_argument("--identity-probation-frames", type=int, default=45, help="Frames to keep uncertain identity in PROBATION before STALE.")
    parser.add_argument("--allow-probation-reinit", action="store_true", help="Allow detector reinit without identity match when no face is visible.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device for torchscript backend.")
    parser.add_argument("--ncnn-use-gpu", action="store_true", help="Use NCNN Vulkan GPU when Python ncnn binding is available.")
    parser.add_argument("--hog-scale", type=float, default=1.05, help="Scale factor for fallback OpenCV HOG detector.")
    parser.add_argument("--hog-win-stride", type=int, default=8, help="Window stride for fallback OpenCV HOG detector.")
    parser.add_argument("--detect-interval", type=int, default=5, help="Run detector every N tracking frames.")
    parser.add_argument("--lost-detect-interval", type=int, default=2, help="Run detector every N frames while searching/lost.")
    parser.add_argument("--low-score-threshold", type=float, default=0.50, help="Tracker score threshold for LOW_CONFIDENCE state.")
    parser.add_argument("--very-low-score-threshold", type=float, default=0.25, help="Tracker score threshold for SUSPECTED_LOST/LOST state.")
    parser.add_argument("--suspect-frames", type=int, default=10, help="Consecutive suspicious frames before SUSPECTED_LOST state.")
    parser.add_argument("--lost-frames", type=int, default=20, help="Consecutive suspicious frames before LOST state.")
    parser.add_argument("--bbox-margin", type=float, default=0.15, help="Allowed bbox margin outside frame as a ratio of bbox size.")
    parser.add_argument("--jump-threshold", type=float, default=0.45, help="Center jump threshold as a ratio of frame diagonal.")
    parser.add_argument("--area-change-threshold", type=float, default=2.80, help="Large bbox area-ratio threshold between adjacent tracking frames.")
    parser.add_argument("--confirm-iou-threshold", type=float, default=0.20, help="Detector/tracker IoU needed to confirm target.")
    parser.add_argument("--confirm-center-threshold", type=float, default=0.35, help="Detector/tracker center distance threshold as frame diagonal ratio.")
    parser.add_argument("--reinit-iou-threshold", type=float, default=0.08, help="Hard reinit if selected detector is below this IoU during low confidence/lost.")
    parser.add_argument("--size-ratio-threshold", type=float, default=2.20, help="Hard reinit if tracker/detector area ratio exceeds this value.")
    parser.add_argument("--shrink-ratio-threshold", type=float, default=1.50, help="Soft reinitialize from detector when tracker bbox is much larger than a confirmed detector bbox.")
    parser.add_argument("--reinit-cooldown-frames", type=int, default=60, help="Minimum frames between detector-driven reinitializations unless target is lost or background-locked.")
    parser.add_argument("--shrink-confirm-frames", type=int, default=3, help="Require this many consecutive confirmed oversized tracker boxes before shrink reinitialization.")
    parser.add_argument("--detector-weight", type=float, default=0.65, help="Soft correction weight for detector bbox.")
    parser.add_argument("--control-box-smoothing", type=float, default=0.35, help="Smoothing weight for the display/control bbox used by later recognition or robot-control stages.")
    parser.add_argument("--control-center-threshold", type=float, default=0.20, help="Maximum center distance, as a frame diagonal ratio, for tracker updates to move the control bbox.")
    parser.add_argument("--ambiguous-margin", type=float, default=0.15, help="If top two detection ranks are closer than this margin, do not reinitialize blindly.")
    parser.add_argument("--metrics-interval", type=int, default=30, help="Write camera_metrics.txt every N frames. 0 means only on exit.")
    parser.add_argument("--wheel-log-only", action="store_true", help="Compute wheel-control commands from the CONTROL box and log them without sending real motor commands.")
    parser.add_argument("--wheel-target-height-ratio", type=float, default=0.70, help="Desired CONTROL-box height ratio used as a distance proxy for wheel log-only control.")
    parser.add_argument("--wheel-kp-linear", type=float, default=0.45, help="Proportional gain for wheel log-only linear control.")
    parser.add_argument("--wheel-kp-angular", type=float, default=0.65, help="Proportional gain for wheel log-only angular control.")
    parser.add_argument("--wheel-max-linear", type=float, default=0.35, help="Clamp limit for wheel log-only linear command.")
    parser.add_argument("--wheel-max-angular", type=float, default=0.45, help="Clamp limit for wheel log-only angular command.")
    parser.add_argument("--wheel-deadband-center", type=float, default=0.05, help="Deadband for normalized CONTROL-box center error before angular command is applied.")
    parser.add_argument("--wheel-deadband-distance", type=float, default=0.05, help="Deadband for normalized CONTROL-box distance proxy error before linear command is applied.")
    parser.set_defaults(eco_prewarm=False)
    parser.add_argument("--eco-prewarm", dest="eco_prewarm", action="store_true", help="Warm ECO with real camera frames before normal detection/tracking starts.")
    parser.add_argument("--no-eco-prewarm", dest="eco_prewarm", action="store_false", help="Skip the ECO prewarm gate.")
    parser.add_argument("--eco-prewarm-bbox", "--prewarm-bbox", dest="eco_prewarm_bbox", default="180,80,260,390", help="Fallback x,y,w,h bbox used if no detector bbox is found during ECO prewarm.")
    parser.add_argument("--eco-prewarm-reinit-runs", "--prewarm-repeats", dest="eco_prewarm_reinit_runs", type=int, default=1, help="Number of extra ECO reinitialize calls to run before READY_WARMED.")
    parser.add_argument("--eco-prewarm-track-frames", type=int, default=30, help="Number of tracker.track frames to run after each ECO prewarm initialize.")
    parser.add_argument("--eco-prewarm-timeout-frames", type=int, default=30, help="Frames to wait for a detector-selected prewarm bbox before using the fallback bbox.")
    parser.add_argument("--reset-video-after-prewarm", action="store_true", help="If the source is a video file, seek back to frame 0 after READY_WARMED before the main loop.")
    return parser.parse_args()


def parse_xywh(value, name):
    parts = str(value).split(",")
    if len(parts) != 4:
        raise ValueError("%s must be x,y,w,h" % name)
    try:
        box = [float(part.strip()) for part in parts]
    except ValueError:
        raise ValueError("%s must contain numeric x,y,w,h values" % name)
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError("%s width and height must be positive" % name)
    return box


def parse_width_height(value, name):
    parts = str(value).split(",")
    if len(parts) != 2:
        raise ValueError("%s must be WIDTH,HEIGHT" % name)
    try:
        width = int(parts[0].strip())
        height = int(parts[1].strip())
    except ValueError:
        raise ValueError("%s must contain integer WIDTH,HEIGHT values" % name)
    if width <= 0 or height <= 0:
        raise ValueError("%s width and height must be positive" % name)
    return width, height


def resolve_project_root(script_path):
    for candidate in [script_path] + list(script_path.parents):
        if (candidate / "pytracking").is_dir():
            return candidate
        if (candidate / "MyECOTracker" / "pytracking").is_dir():
            return candidate / "MyECOTracker"
    raise RuntimeError("Could not locate pytracking project root from %s" % script_path)


def setup_pytracking(project_root):
    pytracking_dir = project_root / "pytracking"
    pytracking_str = str(pytracking_dir)
    if pytracking_str not in sys.path:
        sys.path.insert(0, pytracking_str)


def create_tracker(project_root, tracker_name, param_name):
    setup_pytracking(project_root)
    from pytracking.evaluation.tracker import Tracker

    wrapper = Tracker(tracker_name, param_name)
    params = wrapper.get_parameters()
    params.debug = 0
    params.visualization = False
    tracker = wrapper.create_tracker(params)
    if hasattr(tracker, "initialize_features"):
        tracker.initialize_features()
    return tracker


def resolve_input_source(args):
    if args.video_input is not None:
        video_path = args.video_input.expanduser().resolve()
        return {
            "capture_arg": str(video_path),
            "source_label": str(video_path),
            "video_path": str(video_path),
            "source_is_video": 1,
            "use_gstreamer": False,
            "camera_label": args.camera,
        }
    camera_text = str(args.camera)
    if args.gstreamer:
        return {
            "capture_arg": camera_text,
            "source_label": camera_text,
            "video_path": "",
            "source_is_video": 0,
            "use_gstreamer": True,
            "camera_label": camera_text,
        }
    try:
        int(camera_text)
        is_video = 0
    except ValueError:
        expanded = Path(camera_text).expanduser()
        if expanded.exists():
            camera_text = str(expanded.resolve())
            is_video = 1 if expanded.is_file() else 0
        else:
            is_video = 0
    return {
        "capture_arg": camera_text,
        "source_label": camera_text,
        "video_path": camera_text if is_video else "",
        "source_is_video": int(is_video),
        "use_gstreamer": bool(args.gstreamer),
        "camera_label": str(args.camera),
    }


def open_capture(camera_arg, use_gstreamer, width, height, fps):
    if use_gstreamer:
        cap = cv2.VideoCapture(camera_arg, cv2.CAP_GSTREAMER)
    else:
        try:
            camera_source = int(camera_arg)
        except ValueError:
            camera_source = camera_arg
        cap = cv2.VideoCapture(camera_source)
        if isinstance(camera_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            cap.set(cv2.CAP_PROP_FPS, float(fps))
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera source: %s" % camera_arg)
    return cap


def rotate_frame(frame, mode):
    if mode == "none":
        return frame
    if mode == "90cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "90ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError("Unknown rotate mode: %s" % mode)


def preprocess_input_frame(frame_bgr, args):
    frame = rotate_frame(frame_bgr, args.input_rotate)
    if args.input_resize is None:
        return frame
    resize_w, resize_h = parse_width_height(args.input_resize, "--input-resize")
    if frame.shape[1] == resize_w and frame.shape[0] == resize_h:
        return frame
    return cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)


def read_processed_frame(cap, args):
    source_frame_index = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return False, None, None
    source_info = {
        "source_frame_index": max(0.0, source_frame_index),
        "source_timestamp_s": float("nan"),
    }
    pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    if math.isfinite(pos_msec) and pos_msec >= 0.0:
        source_info["source_timestamp_s"] = pos_msec / 1000.0
    return True, preprocess_input_frame(frame_bgr, args), source_info


def enrich_source_info(source_info, source_is_video, source_fps):
    row = dict(source_info or {})
    row["source_is_video"] = int(bool(source_is_video))
    row["source_fps"] = float(source_fps)
    frame_index = row.get("source_frame_index", float("nan"))
    timestamp_s = row.get("source_timestamp_s", float("nan"))
    if (not math.isfinite(timestamp_s) or timestamp_s < 0.0) and source_is_video and math.isfinite(source_fps) and source_fps > 0.0 and math.isfinite(frame_index):
        row["source_timestamp_s"] = float(frame_index) / float(source_fps)
    if not math.isfinite(row.get("source_timestamp_s", float("nan"))):
        row["source_timestamp_s"] = float("nan")
    if not math.isfinite(row.get("source_frame_index", float("nan"))):
        row["source_frame_index"] = float("nan")
    return row


def clip_xywh(box_xywh, frame_shape):
    height, width = frame_shape[:2]
    x, y, bw, bh = [float(v) for v in box_xywh]
    x = max(0.0, min(x, width - 1.0))
    y = max(0.0, min(y, height - 1.0))
    bw = max(1.0, min(bw, width - x))
    bh = max(1.0, min(bh, height - y))
    return [x, y, bw, bh]


def bbox_center(box):
    x, y, w, h = [float(v) for v in box]
    return x + w / 2.0, y + h / 2.0


def bbox_area(box):
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def box_iou(box_a, box_b):
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def center_distance(box_a, box_b):
    acx, acy = bbox_center(box_a)
    bcx, bcy = bbox_center(box_b)
    return float(math.hypot(acx - bcx, acy - bcy))


def area_ratio(box_a, box_b):
    area_a = max(1.0, bbox_area(box_a))
    area_b = max(1.0, bbox_area(box_b))
    return max(area_a, area_b) / min(area_a, area_b)


def bbox_out_of_frame(box, frame_shape, margin_ratio):
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = [float(v) for v in box]
    margin_x = max(0.0, w * float(margin_ratio))
    margin_y = max(0.0, h * float(margin_ratio))
    return x + w < -margin_x or y + h < -margin_y or x > frame_w + margin_x or y > frame_h + margin_y


def blend_boxes(box_a, box_b, box_b_weight):
    alpha = float(box_b_weight)
    beta = 1.0 - alpha
    return [beta * float(a) + alpha * float(b) for a, b in zip(box_a, box_b)]


def finite_or_nan(value):
    if value is None:
        return float("nan")
    value = float(value)
    if math.isfinite(value):
        return value
    return float("nan")


def clamp(value, low, high):
    return max(float(low), min(float(value), float(high)))


def compute_wheel_control(control_box, frame_width, frame_height, state, identity_state, recognizer_enabled, eco_ready_warmed, args):
    result = {
        "enabled": int(bool(args.wheel_log_only)),
        "allowed": 0,
        "center_error_norm": 0.0,
        "distance_error_norm": 0.0,
        "linear_cmd": 0.0,
        "angular_cmd": 0.0,
        "left_cmd": 0.0,
        "right_cmd": 0.0,
        "reason": "disabled",
    }
    if not args.wheel_log_only:
        return result
    if int(eco_ready_warmed) != 1:
        result["reason"] = "not_ready"
        return result
    if control_box is None:
        result["reason"] = "no_control_box"
        return result
    if state not in ("TRACKING", "DETECTOR_CONFIRMED"):
        result["reason"] = "unsafe_state"
        return result
    if recognizer_enabled and identity_state != IdentityState.VERIFIED:
        result["reason"] = "identity_not_verified"
        return result

    center_x, _ = bbox_center(control_box)
    frame_width = max(1.0, float(frame_width))
    frame_height = max(1.0, float(frame_height))
    center_error_norm = (center_x - frame_width / 2.0) / (frame_width / 2.0)
    current_height_ratio = float(control_box[3]) / frame_height
    distance_error_norm = float(args.wheel_target_height_ratio) - current_height_ratio
    angular_error = 0.0 if abs(center_error_norm) < float(args.wheel_deadband_center) else center_error_norm
    linear_error = 0.0 if abs(distance_error_norm) < float(args.wheel_deadband_distance) else distance_error_norm
    angular_cmd = clamp(float(args.wheel_kp_angular) * angular_error, -float(args.wheel_max_angular), float(args.wheel_max_angular))
    linear_cmd = clamp(float(args.wheel_kp_linear) * linear_error, -float(args.wheel_max_linear), float(args.wheel_max_linear))
    left_cmd = clamp(linear_cmd - angular_cmd, -1.0, 1.0)
    right_cmd = clamp(linear_cmd + angular_cmd, -1.0, 1.0)

    result.update({
        "allowed": 1,
        "center_error_norm": center_error_norm,
        "distance_error_norm": distance_error_norm,
        "linear_cmd": linear_cmd,
        "angular_cmd": angular_cmd,
        "left_cmd": left_cmd,
        "right_cmd": right_cmd,
        "reason": "ok",
    })
    return result


def percentile(values, percent):
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * float(percent) / 100.0
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - pos) + sorted_values[high] * (pos - low)


def resolve_detector_backend(model_path, backend, detector_bin=None, model_path_was_explicit=False):
    if backend != "auto":
        return backend
    if detector_bin is not None and detector_bin.exists():
        return "cpp_ncnn"
    if not model_path_was_explicit or model_path is None:
        return "hog"
    if model_path.is_dir():
        return "ncnn"
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".torchscript":
        return "torchscript"
    return "ncnn"


class CppNcnnStdinDetector(object):
    def __init__(self, detector_bin, model_dir, stderr_path=None):
        self.detector_bin = Path(detector_bin).expanduser().resolve()
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.detector_bin.exists():
            raise RuntimeError("Missing C++ NCNN detector executable: %s" % self.detector_bin)
        if not (self.model_dir / "ssdperson10695.param").exists() or not (self.model_dir / "ssdperson10695.bin").exists():
            raise RuntimeError("Missing NCNN person model files in %s" % self.model_dir)
        self.stderr_file = None
        stderr_target = None
        if stderr_path is not None:
            self.stderr_file = Path(stderr_path).open("w", encoding="utf-8")
            stderr_target = self.stderr_file
        self.process = subprocess.Popen(
            [str(self.detector_bin), "--model-dir", str(self.model_dir), "--conf", "0.0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            bufsize=0,
        )

    def close(self):
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
            except Exception:
                pass
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        stderr_file = getattr(self, "stderr_file", None)
        if stderr_file is not None:
            stderr_file.close()

    def detect(self, frame_bgr, conf_thresh, nms_thresh):
        del nms_thresh
        if self.process.poll() is not None:
            raise RuntimeError("C++ NCNN detector exited with code %s" % self.process.returncode)
        if frame_bgr.dtype != np.uint8 or len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            raise RuntimeError("C++ NCNN detector expects uint8 BGR frames")
        frame = frame_bgr if frame_bgr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame_bgr)
        height, width = frame.shape[:2]
        self.process.stdin.write(struct.pack("<ii", int(width), int(height)))
        self.process.stdin.write(frame.tobytes())
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("C++ NCNN detector produced no response")
        text = line.decode("utf-8", "replace").strip()
        parts = text.split()
        if not parts:
            raise RuntimeError("C++ NCNN detector produced an empty response")
        if parts[0] == "ERR":
            raise RuntimeError("C++ NCNN detector error: %s" % " ".join(parts[1:]))
        if parts[0] != "OK" or len(parts) < 2:
            raise RuntimeError("Unexpected C++ NCNN detector response: %s" % text)
        count = int(parts[1])
        values = parts[2:]
        if len(values) != count * 5:
            raise RuntimeError("Malformed C++ NCNN detector response: %s" % text)
        detections = []
        for index in range(count):
            offset = index * 5
            x, y, w, h, conf = [float(v) for v in values[offset:offset + 5]]
            if conf < float(conf_thresh):
                continue
            detections.append({"box_xywh": clip_xywh([x, y, w, h], frame_bgr.shape), "conf": conf})
        return detections


class NcnnPersonDetector(object):
    def __init__(self, model_dir, input_size, use_gpu=True):
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError(
                "Python module 'ncnn' is not installed in this environment. "
                "Use --detector-backend hog for a no-install smoke test, or run an ONNX/TorchScript detector with --model-path. "
                "The existing Repo 3 NCNN detector was built as C++, not as a Python module."
            ) from exc

        self.ncnn = ncnn
        self.input_size = int(input_size)
        self.model_dir = Path(model_dir).expanduser().resolve()
        param_path = self.model_dir / "ssdperson10695.param"
        bin_path = self.model_dir / "ssdperson10695.bin"
        if not param_path.exists() or not bin_path.exists():
            raise RuntimeError("Missing NCNN person model files in %s" % self.model_dir)
        if use_gpu:
            try:
                ncnn.create_gpu_instance()
            except Exception:
                use_gpu = False
        self.use_gpu = bool(use_gpu)
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = bool(use_gpu)
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

    def close(self):
        try:
            self.net.clear()
        except Exception:
            pass
        if self.use_gpu:
            try:
                self.ncnn.destroy_gpu_instance()
            except Exception:
                pass

    def detect(self, frame_bgr, conf_thresh, nms_thresh):
        del nms_thresh
        img_h, img_w = frame_bgr.shape[:2]
        mat_in = self.ncnn.Mat.from_pixels_resize(
            frame_bgr,
            self.ncnn.Mat.PixelType.PIXEL_BGR,
            img_w,
            img_h,
            self.input_size,
            self.input_size,
        )
        mean_vals = [127.5, 127.5, 127.5]
        norm_vals = [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        ex = self.net.create_extractor()
        ex.set_light_mode(True)
        ex.input("data", mat_in)
        ret, out = ex.extract("detection_out")
        if ret != 0:
            return []
        detections = []
        for i in range(out.h):
            values = out.row(i)
            conf = float(values[1])
            if conf < conf_thresh:
                continue
            x1 = float(values[2]) * img_w
            y1 = float(values[3]) * img_h
            x2 = float(values[4]) * img_w
            y2 = float(values[5]) * img_h
            box = clip_xywh([x1, y1, x2 - x1, y2 - y1], frame_bgr.shape)
            detections.append({"box_xywh": box, "conf": conf})
        return detections


class HOGPersonDetector(object):
    def __init__(self, win_stride, scale):
        self.win_stride = int(win_stride)
        self.scale = float(scale)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def close(self):
        pass

    def detect(self, frame_bgr, conf_thresh, nms_thresh):
        del nms_thresh
        rects, weights = self.hog.detectMultiScale(
            frame_bgr,
            winStride=(self.win_stride, self.win_stride),
            padding=(8, 8),
            scale=self.scale,
        )
        detections = []
        for rect, weight in zip(rects, weights):
            conf = float(weight)
            if conf < float(conf_thresh):
                continue
            x, y, w, h = [float(v) for v in rect]
            detections.append({"box_xywh": clip_xywh([x, y, w, h], frame_bgr.shape), "conf": conf})
        return detections


class YoloDnnDetector(object):
    def __init__(self, model_path, input_size):
        self.model_path = Path(model_path).expanduser().resolve()
        self.input_size = int(input_size)
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def close(self):
        pass

    def detect(self, frame_bgr, conf_thresh, nms_thresh):
        orig_h, orig_w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0]
        return decode_yolo_output(out, frame_bgr.shape, orig_w, orig_h, self.input_size, conf_thresh, nms_thresh)


class TorchscriptDetector(object):
    def __init__(self, model_path, input_size, device):
        import torch

        if device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            torch_device = device
        self.torch = torch
        self.device = torch_device
        self.input_size = int(input_size)
        self.model_path = Path(model_path).expanduser().resolve()
        self.model = torch.jit.load(str(self.model_path), map_location=torch_device)
        self.model.eval()

    def close(self):
        pass

    def detect(self, frame_bgr, conf_thresh, nms_thresh):
        resized = cv2.resize(frame_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            out = self.model(tensor)
        out = out.detach().cpu().numpy()[0]
        orig_h, orig_w = frame_bgr.shape[:2]
        return decode_yolo_output(out, frame_bgr.shape, orig_w, orig_h, self.input_size, conf_thresh, nms_thresh)


def decode_yolo_output(out, frame_shape, orig_w, orig_h, input_size, conf_thresh, nms_thresh):
    boxes = []
    scores = []
    for i in range(out.shape[1]):
        xc, yc, bw, bh, conf = [float(v) for v in out[:, i]]
        if conf < conf_thresh:
            continue
        x = (xc - bw / 2.0) * orig_w / float(input_size)
        y = (yc - bh / 2.0) * orig_h / float(input_size)
        box_w = bw * orig_w / float(input_size)
        box_h = bh * orig_h / float(input_size)
        boxes.append([x, y, box_w, box_h])
        scores.append(conf)
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
    if indices is None or len(indices) == 0:
        return []
    detections = []
    for idx in np.array(indices).reshape(-1).tolist():
        detections.append({"box_xywh": clip_xywh(boxes[idx], frame_shape), "conf": float(scores[idx])})
    return detections


def resolve_path(project_root, path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def load_detector(project_root, args, output_dir=None):
    model_path = args.model_path
    model_path_was_explicit = model_path is not None
    if model_path is None:
        model_path = project_root / PERSON_MODEL_DIR
    else:
        model_path = resolve_path(project_root, model_path)
    detector_bin = resolve_path(project_root, args.detector_bin)
    detector_model_dir = resolve_path(project_root, args.detector_model_dir)
    backend = resolve_detector_backend(model_path, args.detector_backend, detector_bin, model_path_was_explicit)
    if backend == "cpp_ncnn":
        stderr_path = output_dir / "cpp_detector_stderr.log" if output_dir is not None else None
        return backend, detector_model_dir.resolve(), detector_bin.resolve(), CppNcnnStdinDetector(detector_bin, detector_model_dir, stderr_path)
    if backend == "ncnn":
        return backend, model_path.resolve(), Path(""), NcnnPersonDetector(model_path, args.input_size, use_gpu=bool(args.ncnn_use_gpu))
    if backend == "hog":
        return backend, Path("opencv_hog"), Path(""), HOGPersonDetector(args.hog_win_stride, args.hog_scale)
    if backend == "onnx":
        if not model_path.exists():
            raise RuntimeError("Missing ONNX model: %s" % model_path)
        return backend, model_path.resolve(), Path(""), YoloDnnDetector(model_path, args.input_size)
    if backend == "torchscript":
        if not model_path.exists():
            raise RuntimeError("Missing TorchScript model: %s" % model_path)
        return backend, model_path.resolve(), Path(""), TorchscriptDetector(model_path, args.input_size, args.device)
    raise RuntimeError("Unsupported detector backend: %s" % backend)


def resolve_recognizer_backend(project_root, args):
    if args.recognizer_backend == "off":
        return "off"
    recognizer_bin = resolve_path(project_root, args.recognizer_bin)
    model_dir = resolve_path(project_root, args.recognizer_model_dir)
    required = ["retinaface.param", "retinaface.bin", "mbv2facenet.param", "mbv2facenet.bin"]
    models_exist = all((model_dir / name).exists() for name in required)
    if args.recognizer_backend == "cpp_ncnn":
        return "cpp_ncnn"
    if recognizer_bin.exists() and models_exist:
        return "cpp_ncnn"
    return "off"


def load_recognizer(project_root, args, output_dir=None):
    backend = resolve_recognizer_backend(project_root, args)
    recognizer_bin = resolve_path(project_root, args.recognizer_bin)
    model_dir = resolve_path(project_root, args.recognizer_model_dir)
    if backend == "off":
        return backend, model_dir.resolve(), recognizer_bin.resolve(), DisabledFaceRecognizer()
    if backend == "cpp_ncnn":
        stderr_path = output_dir / "cpp_recognizer_stderr.log" if output_dir is not None else None
        return (
            backend,
            model_dir.resolve(),
            recognizer_bin.resolve(),
            CppFaceRecognizer(
                recognizer_bin,
                model_dir,
                face_conf=args.recognizer_face_conf,
                face_nms=args.recognizer_face_nms,
                use_gpu=not bool(args.recognizer_cpu),
                stderr_path=stderr_path,
            ),
        )
    raise RuntimeError("Unsupported recognizer backend: %s" % backend)


def rank_detections(detections, reference_box, frame_shape):
    if not detections:
        return []
    if reference_box is None:
        return sorted([(det["conf"], det) for det in detections], key=lambda item: item[0], reverse=True)
    height, width = frame_shape[:2]
    diag = max(1.0, float(math.hypot(width, height)))
    ranked = []
    for det in detections:
        det_box = det["box_xywh"]
        iou = box_iou(reference_box, det_box)
        dist_ratio = center_distance(reference_box, det_box) / diag
        proximity = max(0.0, 1.0 - dist_ratio)
        size_penalty = min(1.0, abs(math.log(max(area_ratio(reference_box, det_box), 1e-9))) / math.log(4.0))
        rank = 2.2 * iou + 0.8 * proximity + 0.4 * det["conf"] - 0.4 * size_penalty
        ranked.append((rank, det))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def select_detection(detections, reference_box, frame_shape, ambiguous_margin):
    ranked = rank_detections(detections, reference_box, frame_shape)
    if not ranked:
        return None, False, 0.0
    best_rank, best_det = ranked[0]
    ambiguous = False
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < float(ambiguous_margin):
        ambiguous = True
    return best_det, ambiguous, float(best_rank)


def detector_match_metrics(tracker_box, selected_det, frame_diag):
    if selected_det is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    det_conf = selected_det["conf"]
    if tracker_box is None:
        return det_conf, float("nan"), float("nan"), float("nan")
    det_box = selected_det["box_xywh"]
    return (
        det_conf,
        box_iou(tracker_box, det_box),
        center_distance(tracker_box, det_box) / max(1.0, float(frame_diag)),
        area_ratio(tracker_box, det_box),
    )


def find_detection_index(detections, selected_det):
    if selected_det is None:
        return -1
    for idx, det in enumerate(detections):
        if det is selected_det:
            return idx
    selected_box = selected_det.get("box_xywh")
    for idx, det in enumerate(detections):
        if det.get("box_xywh") == selected_box:
            return idx
    return -1


def recognition_boxes_from_candidates(candidates):
    return [item["box_xywh"] for item in candidates if item is not None and item.get("box_xywh") is not None]


def detector_confirms_track(best_iou, center_ratio, area_ratio_value, args):
    iou_ok = math.isfinite(best_iou) and best_iou >= float(args.confirm_iou_threshold)
    overlap_ok = math.isfinite(best_iou) and best_iou >= float(args.reinit_iou_threshold)
    center_ok = math.isfinite(center_ratio) and center_ratio <= float(args.confirm_center_threshold)
    scale_ok = (not math.isfinite(area_ratio_value)) or area_ratio_value <= float(args.size_ratio_threshold)
    return iou_ok or (center_ok and overlap_ok and scale_ok)


def run_identity_recognition(recognizer, identity, frame_index, frame_bgr, candidates, enroll, metrics, danger=False):
    boxes = recognition_boxes_from_candidates(candidates)
    if not identity.enabled or not boxes:
        return -1, 0.0, {"target_ready": identity.target_ready, "enrolled_index": -1, "results": []}
    start = time.perf_counter()
    response = recognizer.enroll(frame_bgr, boxes) if enroll else recognizer.verify(frame_bgr, boxes)
    elapsed = time.perf_counter() - start
    metrics["recognition_calls"] += 1
    metrics["recognition_times"].append(elapsed)
    if enroll:
        candidate_index = identity.observe_enroll(frame_index, response)
        metrics["identity_enrollments"] = identity.enrollments
    else:
        candidate_index = identity.observe_verify(frame_index, response, danger=danger)
    return candidate_index, elapsed, response


class TrackerInitializer(threading.Thread):
    def __init__(self, tracker, frame_bgr, init_box):
        threading.Thread.__init__(self)
        self.daemon = True
        self.tracker = tracker
        self.frame_bgr = frame_bgr.copy()
        self.init_box = list(map(float, init_box))
        self.output = None
        self.error = None
        self.elapsed = 0.0

    def run(self):
        start = time.perf_counter()
        try:
            frame_rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
            self.output = self.tracker.initialize(frame_rgb, {"init_bbox": self.init_box}) or {}
        except Exception as exc:
            self.error = exc
        finally:
            self.elapsed = time.perf_counter() - start


def draw_box(image, box, color, thickness=2, label=None):
    x, y, w, h = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    if label:
        label_y = max(18, y - 6)
        cv2.putText(image, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def draw_label(image, text, point, color):
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def draw_header(image, frame_index, fps_value, state, score, det_count, det_conf, best_iou, identity_state=None, identity_score=None):
    score_text = "nan" if score is None or not math.isfinite(float(score)) else "%.3f" % score
    det_text = "nan" if det_conf is None or not math.isfinite(float(det_conf)) else "%.2f" % det_conf
    iou_text = "nan" if best_iou is None or not math.isfinite(float(best_iou)) else "%.2f" % best_iou
    text = "frame=%d fps=%.2f state=%s score=%s dets=%d det=%s iou=%s" % (frame_index, fps_value, state, score_text, det_count, det_text, iou_text)
    if identity_state is not None:
        id_text = "nan" if identity_score is None or not math.isfinite(float(identity_score)) else "%.2f" % identity_score
        text += " id=%s idscore=%s" % (identity_state, id_text)
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "Auto detect person -> init ECO. R: re-detect. S: screenshot. Q/Esc: quit", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 2, cv2.LINE_AA)


def centered_fallback_box(frame_shape, fallback_box):
    height, width = frame_shape[:2]
    _, _, box_w, box_h = [float(v) for v in fallback_box]
    box_w = min(max(1.0, box_w), float(width))
    box_h = min(max(1.0, box_h), float(height))
    x = (float(width) - box_w) / 2.0
    y = (float(height) - box_h) / 2.0
    return clip_xywh([x, y, box_w, box_h], frame_shape)


def draw_prewarm_frame(frame_bgr, state, init_box, elapsed, repeat_index, total_runs):
    display = frame_bgr.copy()
    if init_box is not None:
        draw_box(display, init_box, (0, 165, 255), 2, "PREWARM")
    draw_label(display, "%s %d/%d %.1fs" % (state, repeat_index + 1, total_runs, elapsed), (12, 82), (0, 165, 255))
    draw_header(display, 0, 0.0, state, None, 0, float("nan"), float("nan"))
    return display


def run_prewarm_initialize(tracker, cap, args, init_box, run_index, total_runs, started, metrics):
    ok, frame_bgr, _ = read_processed_frame(cap, args)
    if not ok or frame_bgr is None:
        raise RuntimeError("Camera frame read failed during ECO prewarm.")
    metrics["frame_width"] = frame_bgr.shape[1]
    metrics["frame_height"] = frame_bgr.shape[0]
    init_box = clip_xywh(init_box, frame_bgr.shape)
    worker = TrackerInitializer(tracker, frame_bgr, init_box)
    worker.start()
    while worker.is_alive():
        display = draw_prewarm_frame(frame_bgr, "WARMING_ECO", init_box, time.perf_counter() - started, run_index, total_runs)
        cv2.imshow(args.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return None, None, None, False
    if worker.error is not None:
        raise worker.error

    prev_output = dict(worker.output or {})
    current_box = [float(v) for v in prev_output.get("target_bbox", init_box)]
    for track_index in range(max(0, int(args.eco_prewarm_track_frames))):
        ok, track_frame_bgr, _ = read_processed_frame(cap, args)
        if not ok or track_frame_bgr is None:
            raise RuntimeError("Camera frame read failed during ECO prewarm tracking.")
        frame_rgb = cv2.cvtColor(track_frame_bgr, cv2.COLOR_BGR2RGB)
        prev_output = dict(tracker.track(frame_rgb, {"previous_output": prev_output}) or {})
        current_box = [float(v) for v in prev_output.get("target_bbox", current_box)]
        display = draw_prewarm_frame(track_frame_bgr, "WARMING_ECO_TRACK", current_box, time.perf_counter() - started, run_index, total_runs)
        cv2.imshow(args.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return None, None, None, False
    return worker.elapsed, current_box, prev_output, True


def select_prewarm_box(cap, detector, args, metrics):
    fallback_box = parse_xywh(args.eco_prewarm_bbox, "--eco-prewarm-bbox")
    timeout_frames = max(0, int(args.eco_prewarm_timeout_frames))
    last_frame = None
    for frame_offset in range(timeout_frames):
        ok, frame_bgr, _ = read_processed_frame(cap, args)
        if not ok or frame_bgr is None:
            raise RuntimeError("Camera frame read failed while selecting ECO prewarm bbox.")
        last_frame = frame_bgr
        metrics["frame_width"] = frame_bgr.shape[1]
        metrics["frame_height"] = frame_bgr.shape[0]
        detections = detector.detect(frame_bgr, args.detector_conf, args.nms_threshold)
        selected_det, ambiguous_detection, _ = select_detection(detections, None, frame_bgr.shape, args.ambiguous_margin)
        display = frame_bgr.copy()
        if selected_det is not None:
            draw_box(display, selected_det["box_xywh"], (255, 0, 0), 2, "PREWARM_DET")
        draw_label(display, "WARMING_ECO_SELECT %d/%d" % (frame_offset + 1, max(1, timeout_frames)), (12, 82), (0, 165, 255))
        draw_header(display, 0, 0.0, "WARMING_ECO_SELECT", None, len(detections), selected_det["conf"] if selected_det is not None else float("nan"), float("nan"))
        cv2.imshow(args.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return None, False
        if selected_det is not None and not ambiguous_detection:
            print("eco_prewarm_detector_bbox=%.3f,%.3f,%.3f,%.3f det_conf=%.3f" % tuple(selected_det["box_xywh"] + [selected_det["conf"]]), flush=True)
            return clip_xywh(selected_det["box_xywh"], frame_bgr.shape), True
    if last_frame is None:
        ok, last_frame, _ = read_processed_frame(cap, args)
        if not ok or last_frame is None:
            raise RuntimeError("Camera frame read failed while building ECO prewarm fallback bbox.")
    fallback = centered_fallback_box(last_frame.shape, fallback_box)
    print("eco_prewarm_fallback_bbox=%.3f,%.3f,%.3f,%.3f" % tuple(fallback), flush=True)
    return fallback, True


def prewarm_eco_tracker(tracker, cap, detector, args, metrics):
    reinit_runs = max(0, int(args.eco_prewarm_reinit_runs))
    total_runs = 1 + reinit_runs
    if not args.eco_prewarm:
        return {"completed": True, "standby_warmed": False, "standby_box": None, "standby_prev_output": {}}

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    started = time.perf_counter()
    metrics["eco_prewarm_enabled"] = 1
    print("eco_prewarm_start reinit_runs=%d track_frames=%d timeout_frames=%d" % (reinit_runs, int(args.eco_prewarm_track_frames), int(args.eco_prewarm_timeout_frames)), flush=True)
    prewarm_box, should_continue = select_prewarm_box(cap, detector, args, metrics)
    if not should_continue:
        return {"completed": False, "standby_warmed": False, "standby_box": None, "standby_prev_output": {}}

    first_elapsed, standby_box, standby_prev_output, should_continue = run_prewarm_initialize(tracker, cap, args, prewarm_box, 0, total_runs, started, metrics)
    if not should_continue:
        return {"completed": False, "standby_warmed": False, "standby_box": None, "standby_prev_output": {}}
    metrics["eco_prewarm_first_init_time_s"] = first_elapsed
    print("eco_prewarm_first_init_time_s=%.3f" % first_elapsed, flush=True)

    reinit_times = []
    for reinit_index in range(reinit_runs):
        elapsed, standby_box, standby_prev_output, should_continue = run_prewarm_initialize(tracker, cap, args, standby_box, reinit_index + 1, total_runs, started, metrics)
        if not should_continue:
            return {"completed": False, "standby_warmed": False, "standby_box": None, "standby_prev_output": {}}
        reinit_times.append(elapsed)
        print("eco_prewarm_reinit_index=%d init_time_s=%.3f" % (reinit_index, elapsed), flush=True)

    metrics["eco_prewarm_reinit_count"] = len(reinit_times)
    metrics["eco_prewarm_reinit_times"] = reinit_times
    metrics["eco_prewarm_reinit_avg_s"] = sum(reinit_times) / float(len(reinit_times)) if reinit_times else 0.0
    metrics["eco_prewarm_total_time_s"] = time.perf_counter() - started
    metrics["eco_ready_warmed"] = 1
    metrics["eco_standby_warmed"] = 1 if standby_prev_output else 0

    ok, frame_bgr, _ = read_processed_frame(cap, args)
    if ok and frame_bgr is not None:
        display = frame_bgr.copy()
        draw_label(display, "READY_WARMED", (12, 82), (0, 255, 0))
        draw_header(display, 0, 0.0, "READY_WARMED", None, 0, float("nan"), float("nan"))
        cv2.imshow(args.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return {"completed": False, "standby_warmed": False, "standby_box": None, "standby_prev_output": {}}
    print("READY_WARMED eco_prewarm_total_time_s=%.3f reinit_avg_s=%.3f standby_warmed=%d" % (metrics["eco_prewarm_total_time_s"], metrics["eco_prewarm_reinit_avg_s"], metrics["eco_standby_warmed"]), flush=True)
    return {
        "completed": True,
        "standby_warmed": bool(standby_prev_output),
        "standby_box": standby_box[:] if standby_box is not None else None,
        "standby_prev_output": dict(standby_prev_output or {}),
    }


def write_run_info(output_dir, args, detector_backend, detector_model_path, detector_bin_path, recognizer_backend=None, recognizer_model_path=None, recognizer_bin_path=None, source_info=None):
    if output_dir is None:
        return
    info_path = output_dir / "run_info.txt"
    source_info = source_info or {}
    with info_path.open("w", encoding="utf-8") as f:
        f.write("script=%s\n" % Path(__file__).name)
        f.write("output_dir=%s\n" % output_dir)
        f.write("detector_backend=%s\n" % detector_backend)
        f.write("detector_model=%s\n" % detector_model_path)
        f.write("detector_bin=%s\n" % detector_bin_path)
        if recognizer_backend is not None:
            f.write("recognizer_backend=%s\n" % recognizer_backend)
            f.write("recognizer_model=%s\n" % recognizer_model_path)
            f.write("recognizer_bin=%s\n" % recognizer_bin_path)
        f.write("input_source=%s\n" % source_info.get("source_label", args.camera))
        f.write("input_source_is_video=%s\n" % int(bool(source_info.get("source_is_video", 0))))
        f.write("input_video_path=%s\n" % source_info.get("video_path", ""))
        f.write("input_rotate=%s\n" % args.input_rotate)
        f.write("input_resize=%s\n" % (args.input_resize if args.input_resize is not None else "none"))
        f.write("reset_video_after_prewarm=%s\n" % int(bool(args.reset_video_after_prewarm)))
        f.write("predictions_csv=%s\n" % (output_dir / "camera_predictions.csv"))
        f.write("metrics_summary=%s\n" % (output_dir / "camera_metrics.txt"))
        f.write("argv=%s\n" % " ".join(sys.argv))
        f.write("args=%s\n" % vars(args))
    print("run_info=%s" % info_path, flush=True)


def write_metrics_summary(output_dir, metrics):
    if output_dir is None:
        return
    duration_s = max(metrics["end_time"] - metrics["start_time"], 1e-9)
    scores = metrics["scores"]
    center_deltas = metrics["center_deltas"]
    area_ratios = metrics["area_ratios"]
    detector_times = metrics["detector_times"]
    recognition_times = metrics["recognition_times"]
    init_times = metrics["init_times"]
    summary_path = output_dir / "camera_metrics.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for key in [
            "camera", "tracker", "param", "detector_backend", "detector_model", "detector_bin", "frame_width", "frame_height",
            "input_size", "detector_conf", "detect_interval", "lost_detect_interval", "low_score_threshold",
            "very_low_score_threshold", "suspect_frames_threshold", "lost_frames_threshold", "confirm_iou_threshold",
            "confirm_center_threshold", "reinit_iou_threshold", "size_ratio_threshold", "shrink_ratio_threshold",
            "reinit_cooldown_frames", "shrink_confirm_frames", "tracker_load_time_s", "detector_load_time_s",
            "recognizer_backend", "recognizer_model", "recognizer_bin", "recognizer_load_time_s",
            "recognize_interval", "recognize_fast_interval", "identity_match_threshold", "identity_reject_threshold",
            "control_box_smoothing", "control_center_threshold", "source_label", "source_is_video", "source_fps",
            "input_rotate", "input_resize", "reset_video_after_prewarm", "eco_prewarm_enabled", "eco_ready_warmed",
            "eco_standby_warmed", "eco_first_live_handoff_mode", "eco_first_live_acquisition_time_s",
            "eco_prewarm_first_init_time_s", "eco_prewarm_reinit_count", "eco_prewarm_reinit_avg_s",
            "eco_prewarm_total_time_s", "wheel_control_enabled",
        ]:
            value = metrics[key]
            if isinstance(value, float):
                f.write("%s=%.6f\n" % (key, value))
            else:
                f.write("%s=%s\n" % (key, value))
        f.write("eco_prewarm_reinit_times_s=%s\n" % ",".join("%.6f" % value for value in metrics["eco_prewarm_reinit_times"]))
        f.write("duration_s=%.6f\n" % duration_s)
        f.write("frames_total=%d\n" % metrics["frames_total"])
        f.write("overall_loop_fps=%.6f\n" % (float(metrics["frames_total"]) / duration_s))
        f.write("track_frames=%d\n" % metrics["track_frames"])
        f.write("tracking_fps_excluding_idle=%.6f\n" % (float(max(metrics["track_frames"], 1)) / max(metrics["track_time_total_s"], 1e-9)))
        f.write("track_time_total_s=%.6f\n" % metrics["track_time_total_s"])
        f.write("track_time_avg_s=%.6f\n" % (metrics["track_time_total_s"] / float(max(metrics["track_frames"], 1))))
        f.write("detector_calls=%d\n" % metrics["detector_calls"])
        f.write("detector_time_total_s=%.6f\n" % sum(detector_times))
        f.write("detector_time_avg_s=%.6f\n" % (sum(detector_times) / float(len(detector_times)) if detector_times else 0.0))
        f.write("detections_total=%d\n" % metrics["detections_total"])
        f.write("detections_avg_per_call=%.6f\n" % (float(metrics["detections_total"]) / float(max(metrics["detector_calls"], 1))))
        f.write("recognition_calls=%d\n" % metrics["recognition_calls"])
        f.write("recognition_time_total_s=%.6f\n" % sum(recognition_times))
        f.write("recognition_time_avg_s=%.6f\n" % (sum(recognition_times) / float(len(recognition_times)) if recognition_times else 0.0))
        f.write("identity_enrollments=%d\n" % metrics["identity_enrollments"])
        f.write("identity_verified_frames=%d\n" % metrics["identity_verified_frames"])
        f.write("identity_stale_frames=%d\n" % metrics["identity_stale_frames"])
        f.write("identity_probation_frames=%d\n" % metrics["identity_probation_frames"])
        f.write("identity_rejected_frames=%d\n" % metrics["identity_rejected_frames"])
        f.write("identity_reinit_blocks=%d\n" % metrics["identity_reinit_blocks"])
        f.write("initializations_completed=%d\n" % metrics["initializations_completed"])
        f.write("init_time_avg_s=%.6f\n" % (sum(init_times) / float(len(init_times)) if init_times else 0.0))
        f.write("init_time_max_s=%.6f\n" % (max(init_times) if init_times else 0.0))
        f.write("soft_reinitializations=%d\n" % metrics["soft_reinitializations"])
        f.write("hard_reinitializations=%d\n" % metrics["hard_reinitializations"])
        f.write("hard_reinit_out_of_frame=%d\n" % metrics["hard_reinit_out_of_frame"])
        f.write("hard_reinit_lost_suspicious=%d\n" % metrics["hard_reinit_lost_suspicious"])
        f.write("hard_reinit_low_iou=%d\n" % metrics["hard_reinit_low_iou"])
        f.write("hard_reinit_area_ratio=%d\n" % metrics["hard_reinit_area_ratio"])
        f.write("hard_reinit_background_lock=%d\n" % metrics["hard_reinit_background_lock"])
        f.write("hard_reinit_identity_verified=%d\n" % metrics["hard_reinit_identity_verified"])
        f.write("control_box_frames=%d\n" % metrics["control_box_frames"])
        f.write("control_box_from_detector=%d\n" % metrics["control_box_from_detector"])
        f.write("control_box_from_tracker=%d\n" % metrics["control_box_from_tracker"])
        f.write("control_box_held=%d\n" % metrics["control_box_held"])
        f.write("wheel_control_allowed_frames=%d\n" % metrics["wheel_control_allowed_frames"])
        f.write("wheel_control_blocked_frames=%d\n" % metrics["wheel_control_blocked_frames"])
        allowed_frames = max(metrics["wheel_control_allowed_frames"], 1)
        f.write("wheel_linear_abs_avg=%.6f\n" % (metrics["wheel_linear_abs_sum"] / float(allowed_frames) if metrics["wheel_control_allowed_frames"] > 0 else 0.0))
        f.write("wheel_angular_abs_avg=%.6f\n" % (metrics["wheel_angular_abs_sum"] / float(allowed_frames) if metrics["wheel_control_allowed_frames"] > 0 else 0.0))
        f.write("wheel_left_abs_max=%.6f\n" % metrics["wheel_left_abs_max"])
        f.write("wheel_right_abs_max=%.6f\n" % metrics["wheel_right_abs_max"])
        f.write("ambiguous_reinit_skips=%d\n" % metrics["ambiguous_reinit_skips"])
        f.write("reinit_cooldown_skips=%d\n" % metrics["reinit_cooldown_skips"])
        f.write("shrink_reinit_cooldown_skips=%d\n" % metrics["shrink_reinit_cooldown_skips"])
        f.write("shrink_confirm_count_max=%d\n" % metrics["shrink_confirm_count_max"])
        f.write("detector_confirmed_frames=%d\n" % metrics["detector_confirmed_frames"])
        f.write("detector_missing_frames=%d\n" % metrics["detector_missing_frames"])
        f.write("background_lock_events=%d\n" % metrics["background_lock_events"])
        f.write("lost_events=%d\n" % metrics["lost_events"])
        f.write("score_count=%d\n" % len(scores))
        f.write("score_min=%.6f\n" % (min(scores) if scores else float("nan")))
        f.write("score_p10=%.6f\n" % percentile(scores, 10))
        f.write("score_median=%.6f\n" % percentile(scores, 50))
        f.write("score_p90=%.6f\n" % percentile(scores, 90))
        f.write("score_mean=%.6f\n" % (sum(scores) / float(len(scores)) if scores else float("nan")))
        f.write("score_max=%.6f\n" % (max(scores) if scores else float("nan")))
        f.write("center_delta_avg_px=%.6f\n" % (sum(center_deltas) / float(len(center_deltas)) if center_deltas else 0.0))
        f.write("center_delta_max_px=%.6f\n" % (max(center_deltas) if center_deltas else 0.0))
        f.write("area_ratio_avg=%.6f\n" % (sum(area_ratios) / float(len(area_ratios)) if area_ratios else 1.0))
        f.write("area_ratio_max=%.6f\n" % (max(area_ratios) if area_ratios else 1.0))
        f.write("low_score_frames=%d\n" % metrics["low_score_frames"])
        f.write("very_low_score_frames=%d\n" % metrics["very_low_score_frames"])
        f.write("low_confidence_frames=%d\n" % metrics["low_confidence_frames"])
        f.write("suspected_lost_frames=%d\n" % metrics["suspected_lost_frames"])
        f.write("lost_frames=%d\n" % metrics["lost_frames"])
        f.write("out_of_frame_frames=%d\n" % metrics["out_of_frame_frames"])
        f.write("large_jump_frames=%d\n" % metrics["large_jump_frames"])
        f.write("large_area_change_frames=%d\n" % metrics["large_area_change_frames"])
        f.write("max_consecutive_suspicious=%d\n" % metrics["max_consecutive_suspicious"])
    print("metrics_summary=%s" % summary_path, flush=True)


def main():
    args = parse_args()
    if args.input_resize is not None:
        parse_width_height(args.input_resize, "--input-resize")
    input_source = resolve_input_source(args)
    output_dir = None
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print("output_dir=%s" % output_dir, flush=True)

    print("Resolving project root...", flush=True)
    project_root = resolve_project_root(Path(__file__).resolve().parent)
    print("project_root=%s" % project_root, flush=True)

    print("Loading detector...", flush=True)
    detector_load_start = time.perf_counter()
    detector_backend, detector_model_path, detector_bin_path, detector = load_detector(project_root, args, output_dir)
    detector_load_time = time.perf_counter() - detector_load_start
    print("Detector ready backend=%s model=%s bin=%s time=%.3fs." % (detector_backend, detector_model_path, detector_bin_path, detector_load_time), flush=True)

    print("Loading recognizer...", flush=True)
    recognizer_load_start = time.perf_counter()
    recognizer_backend, recognizer_model_path, recognizer_bin_path, recognizer = load_recognizer(project_root, args, output_dir)
    recognizer_load_time = time.perf_counter() - recognizer_load_start
    print("Recognizer ready backend=%s model=%s bin=%s time=%.3fs." % (recognizer_backend, recognizer_model_path, recognizer_bin_path, recognizer_load_time), flush=True)
    recognizer_enabled = recognizer_backend != "off"
    identity = IdentityManager(
        enabled=recognizer_enabled,
        match_threshold=args.identity_match_threshold,
        reject_threshold=args.identity_reject_threshold,
        reject_confirm_frames=args.identity_reject_confirm_frames,
        stale_frames=args.identity_stale_frames,
        probation_frames=args.identity_probation_frames,
    )
    write_run_info(output_dir, args, detector_backend, detector_model_path, detector_bin_path, recognizer_backend, recognizer_model_path, recognizer_bin_path, input_source)

    print("Creating tracker %s/%s..." % (args.tracker_name, args.param), flush=True)
    tracker_load_start = time.perf_counter()
    tracker = create_tracker(project_root, args.tracker_name, args.param)
    tracker_load_time = time.perf_counter() - tracker_load_start
    print("Tracker ready in %.3fs." % tracker_load_time, flush=True)

    print("Opening camera source: %s" % input_source["source_label"], flush=True)
    cap = open_capture(input_source["capture_arg"], input_source["use_gstreamer"], args.width, args.height, args.fps)
    print("Camera opened.", flush=True)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = float(args.fps)

    csv_file = None
    writer_csv = None
    video_writer = None
    if output_dir is not None:
        csv_file = (output_dir / "camera_predictions.csv").open("w", newline="", encoding="utf-8")
        print("predictions_csv=%s" % (output_dir / "camera_predictions.csv"), flush=True)
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow([
            "frame_index", "timestamp_s", "source_frame_index", "source_timestamp_s", "source_fps", "source_is_video", "tracker_x", "tracker_y", "tracker_w", "tracker_h",
            "final_x", "final_y", "final_w", "final_h", "control_x", "control_y", "control_w", "control_h", "control_source", "det_x", "det_y", "det_w", "det_h",
            "tracker_score", "det_conf", "det_count", "best_detector_iou", "best_center_distance_ratio",
            "best_area_ratio", "state", "track_time_s", "detect_time_s", "init_time_s", "fps",
            "is_low_score", "is_very_low_score", "is_out_of_frame", "center_delta_px", "center_delta_ratio",
            "area_ratio", "is_large_jump", "is_large_area_change", "consecutive_suspicious", "ambiguous_detection",
            "soft_reinitializations", "hard_reinitializations", "background_lock_events",
            "identity_state", "identity_score", "identity_face_found", "identity_candidate_index",
            "identity_target_ready", "recognition_time_s", "identity_reinit_blocks",
            "eco_ready_warmed", "eco_standby_warmed", "eco_first_live_handoff_mode",
            "wheel_control_enabled", "wheel_control_allowed", "wheel_center_error_norm", "wheel_distance_error_norm",
            "wheel_linear_cmd", "wheel_angular_cmd", "wheel_left_cmd", "wheel_right_cmd", "wheel_control_reason",
        ])

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    tracking_active = False
    initializing = False
    init_worker = None
    prev_output = {}
    current_box = None
    previous_box = None
    last_detector_box = None
    control_box = None
    control_box_source = "none"
    standby_warmed = False
    standby_prev_output = {}
    standby_box = None
    standby_handoff_pending = False
    first_live_acquisition_recorded = False
    frame_index = 0
    last_time = time.perf_counter()
    run_start = time.perf_counter()
    consecutive_suspicious = 0
    consecutive_detector_missing = 0
    was_lost = False
    was_background_locked = False
    last_init_time = 0.0
    last_reinit_frame = -10**9
    shrink_confirm_count = 0
    force_redetect = False
    metrics = {
        "camera": args.camera,
        "tracker": args.tracker_name,
        "param": args.param,
        "source_label": input_source["source_label"],
        "source_is_video": int(input_source["source_is_video"]),
        "source_fps": float(source_fps),
        "input_rotate": args.input_rotate,
        "input_resize": args.input_resize if args.input_resize is not None else "none",
        "reset_video_after_prewarm": int(bool(args.reset_video_after_prewarm and input_source["source_is_video"])),
        "detector_backend": detector_backend,
        "detector_model": str(detector_model_path),
        "detector_bin": str(detector_bin_path),
        "recognizer_backend": recognizer_backend,
        "recognizer_model": str(recognizer_model_path),
        "recognizer_bin": str(recognizer_bin_path),
        "frame_width": 0,
        "frame_height": 0,
        "input_size": int(args.input_size),
        "detector_conf": float(args.detector_conf),
        "detect_interval": int(args.detect_interval),
        "lost_detect_interval": int(args.lost_detect_interval),
        "low_score_threshold": float(args.low_score_threshold),
        "very_low_score_threshold": float(args.very_low_score_threshold),
        "suspect_frames_threshold": int(args.suspect_frames),
        "lost_frames_threshold": int(args.lost_frames),
        "confirm_iou_threshold": float(args.confirm_iou_threshold),
        "confirm_center_threshold": float(args.confirm_center_threshold),
        "reinit_iou_threshold": float(args.reinit_iou_threshold),
        "size_ratio_threshold": float(args.size_ratio_threshold),
        "shrink_ratio_threshold": float(args.shrink_ratio_threshold),
        "reinit_cooldown_frames": int(args.reinit_cooldown_frames),
        "shrink_confirm_frames": int(args.shrink_confirm_frames),
        "tracker_load_time_s": tracker_load_time,
        "detector_load_time_s": detector_load_time,
        "recognizer_load_time_s": recognizer_load_time,
        "recognize_interval": int(args.recognize_interval),
        "recognize_fast_interval": int(args.recognize_fast_interval),
        "identity_match_threshold": float(args.identity_match_threshold),
        "identity_reject_threshold": float(args.identity_reject_threshold),
        "control_box_smoothing": float(args.control_box_smoothing),
        "control_center_threshold": float(args.control_center_threshold),
        "eco_prewarm_enabled": int(bool(args.eco_prewarm)),
        "wheel_control_enabled": int(bool(args.wheel_log_only)),
        "eco_ready_warmed": 0,
        "eco_standby_warmed": 0,
        "eco_first_live_handoff_mode": "none",
        "eco_first_live_acquisition_time_s": 0.0,
        "eco_prewarm_first_init_time_s": 0.0,
        "eco_prewarm_reinit_count": 0,
        "eco_prewarm_reinit_avg_s": 0.0,
        "eco_prewarm_reinit_times": [],
        "eco_prewarm_total_time_s": 0.0,
        "start_time": run_start,
        "end_time": run_start,
        "frames_total": 0,
        "track_frames": 0,
        "track_time_total_s": 0.0,
        "detector_calls": 0,
        "detector_times": [],
        "detections_total": 0,
        "recognition_calls": 0,
        "recognition_times": [],
        "identity_enrollments": 0,
        "identity_verified_frames": 0,
        "identity_stale_frames": 0,
        "identity_probation_frames": 0,
        "identity_rejected_frames": 0,
        "identity_reinit_blocks": 0,
        "initializations_completed": 0,
        "init_times": [],
        "soft_reinitializations": 0,
        "hard_reinitializations": 0,
        "hard_reinit_out_of_frame": 0,
        "hard_reinit_lost_suspicious": 0,
        "hard_reinit_low_iou": 0,
        "hard_reinit_area_ratio": 0,
        "hard_reinit_background_lock": 0,
        "hard_reinit_identity_verified": 0,
        "control_box_frames": 0,
        "control_box_from_detector": 0,
        "control_box_from_tracker": 0,
        "control_box_held": 0,
        "wheel_control_allowed_frames": 0,
        "wheel_control_blocked_frames": 0,
        "wheel_linear_abs_sum": 0.0,
        "wheel_angular_abs_sum": 0.0,
        "wheel_left_abs_max": 0.0,
        "wheel_right_abs_max": 0.0,
        "ambiguous_reinit_skips": 0,
        "reinit_cooldown_skips": 0,
        "shrink_reinit_cooldown_skips": 0,
        "shrink_confirm_count_max": 0,
        "detector_confirmed_frames": 0,
        "detector_missing_frames": 0,
        "background_lock_events": 0,
        "lost_events": 0,
        "scores": [],
        "center_deltas": [],
        "area_ratios": [],
        "low_score_frames": 0,
        "very_low_score_frames": 0,
        "low_confidence_frames": 0,
        "suspected_lost_frames": 0,
        "lost_frames": 0,
        "out_of_frame_frames": 0,
        "large_jump_frames": 0,
        "large_area_change_frames": 0,
        "max_consecutive_suspicious": 0,
    }

    try:
        prewarm_result = prewarm_eco_tracker(tracker, cap, detector, args, metrics)
        if not prewarm_result["completed"]:
            return 0
        if input_source["source_is_video"] and args.reset_video_after_prewarm:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("reset_video_after_prewarm=1", flush=True)
        run_start = time.perf_counter()
        last_time = run_start
        metrics["start_time"] = run_start
        metrics["end_time"] = run_start
        tracking_active = False
        initializing = False
        init_worker = None
        prev_output = {}
        current_box = None
        previous_box = None
        control_box = None
        control_box_source = "none"
        standby_warmed = bool(prewarm_result["standby_warmed"])
        standby_prev_output = dict(prewarm_result["standby_prev_output"])
        standby_box = prewarm_result["standby_box"][:] if prewarm_result["standby_box"] is not None else None
        standby_handoff_pending = standby_warmed

        while True:
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            ok, frame_bgr, source_info = read_processed_frame(cap, args)
            if not ok or frame_bgr is None:
                print("Camera frame read failed.", file=sys.stderr)
                break
            source_info = enrich_source_info(source_info, input_source["source_is_video"], source_fps)

            frame_height, frame_width = frame_bgr.shape[:2]
            frame_diag = max(1.0, math.hypot(float(frame_width), float(frame_height)))
            metrics["frame_width"] = frame_width
            metrics["frame_height"] = frame_height
            display = frame_bgr.copy()
            track_elapsed = 0.0
            detect_elapsed = 0.0
            recognition_elapsed = 0.0
            init_time_for_row = 0.0
            score = None
            tracker_box = current_box[:] if current_box is not None else None
            final_box = current_box[:] if current_box is not None else None
            selected_det = None
            detections = []
            det_count = 0
            det_conf = float("nan")
            best_detector_iou = float("nan")
            best_center_distance_ratio = float("nan")
            best_area_ratio = float("nan")
            ambiguous_detection = False
            identity_candidate_index = -1
            identity_face_found = False
            identity_score = identity.last_similarity
            identity_verified_detection = False
            is_low_score = False
            is_very_low_score = False
            is_out_of_frame = False
            is_large_jump = False
            is_large_area_change = False
            center_delta = 0.0
            center_delta_ratio = 0.0
            adjacent_area_ratio = 1.0
            state = "SEARCHING"
            color = (0, 255, 255)
            hard_reinit_reasons = []
            detector_confirmed = False
            detector_conflict = False
            rescue_reinit = False

            if init_worker is not None and not init_worker.is_alive():
                if init_worker.error is not None:
                    raise init_worker.error
                out = init_worker.output or {}
                prev_output = dict(out)
                current_box = [float(v) for v in out.get("target_bbox", init_worker.init_box)]
                previous_box = current_box[:]
                tracking_active = True
                initializing = False
                consecutive_suspicious = 0
                consecutive_detector_missing = 0
                was_lost = False
                was_background_locked = False
                last_init_time = init_worker.elapsed
                init_time_for_row = last_init_time
                metrics["initializations_completed"] += 1
                metrics["init_times"].append(last_init_time)
                if standby_handoff_pending:
                    metrics["eco_first_live_acquisition_time_s"] = last_init_time
                    standby_handoff_pending = False
                    first_live_acquisition_recorded = True
                elif not first_live_acquisition_recorded:
                    metrics["eco_first_live_acquisition_time_s"] = last_init_time
                    first_live_acquisition_recorded = True
                print("initialized_bbox=%.3f,%.3f,%.3f,%.3f init_time_s=%.3f" % tuple(current_box + [last_init_time]), flush=True)
                init_worker = None

            need_detection = force_redetect
            if not tracking_active and not initializing:
                need_detection = frame_index % max(1, int(args.lost_detect_interval)) == 0
            elif tracking_active:
                need_detection = frame_index % max(1, int(args.detect_interval)) == 0

            if tracking_active:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                start = time.perf_counter()
                out = tracker.track(frame_rgb, {"previous_output": prev_output}) or {}
                track_elapsed = time.perf_counter() - start
                prev_output = dict(out)
                previous_for_metrics = previous_box
                tracker_box = [float(v) for v in out["target_bbox"]]
                final_box = tracker_box[:]
                current_box = tracker_box[:]
                raw_score = getattr(tracker, "last_max_score", float("nan"))
                score = finite_or_nan(raw_score)
                is_low_score = math.isfinite(score) and score < float(args.low_score_threshold)
                is_very_low_score = math.isfinite(score) and score < float(args.very_low_score_threshold)
                is_out_of_frame = bbox_out_of_frame(tracker_box, frame_bgr.shape, args.bbox_margin)

                if previous_for_metrics is not None:
                    center_delta = center_distance(previous_for_metrics, tracker_box)
                    center_delta_ratio = center_delta / frame_diag
                    adjacent_area_ratio = area_ratio(previous_for_metrics, tracker_box)

                is_large_jump = center_delta_ratio > float(args.jump_threshold)
                is_large_area_change = adjacent_area_ratio > float(args.area_change_threshold)
                suspicious = is_out_of_frame or is_very_low_score or (is_low_score and (is_large_jump or is_large_area_change))
                weak_confidence = is_low_score or is_large_jump or is_large_area_change
                if suspicious:
                    consecutive_suspicious += 1
                else:
                    consecutive_suspicious = 0
                if is_low_score or is_very_low_score or consecutive_suspicious >= int(args.suspect_frames):
                    need_detection = True
                metrics["max_consecutive_suspicious"] = max(metrics["max_consecutive_suspicious"], consecutive_suspicious)
                metrics["track_frames"] += 1
                metrics["track_time_total_s"] += track_elapsed
                if math.isfinite(score):
                    metrics["scores"].append(score)
                metrics["center_deltas"].append(center_delta)
                metrics["area_ratios"].append(adjacent_area_ratio)
                if is_low_score:
                    metrics["low_score_frames"] += 1
                if is_very_low_score:
                    metrics["very_low_score_frames"] += 1
                if is_out_of_frame:
                    metrics["out_of_frame_frames"] += 1
                if is_large_jump:
                    metrics["large_jump_frames"] += 1
                if is_large_area_change:
                    metrics["large_area_change_frames"] += 1

            if need_detection and not initializing:
                detect_start = time.perf_counter()
                detections = detector.detect(frame_bgr, args.detector_conf, args.nms_threshold)
                detect_elapsed = time.perf_counter() - detect_start
                metrics["detector_calls"] += 1
                metrics["detector_times"].append(detect_elapsed)
                metrics["detections_total"] += len(detections)
                det_count = len(detections)
                reference_box = tracker_box if tracker_box is not None else last_detector_box
                selected_det, ambiguous_detection, _ = select_detection(detections, reference_box, frame_bgr.shape, args.ambiguous_margin)
                if selected_det is not None:
                    det_box = selected_det["box_xywh"]
                    det_conf, best_detector_iou, best_center_distance_ratio, best_area_ratio = detector_match_metrics(tracker_box, selected_det, frame_diag)
                    last_detector_box = det_box[:]
                else:
                    consecutive_detector_missing += 1
                    metrics["detector_missing_frames"] += 1

            if initializing:
                state = "INITIALIZING"
                color = (0, 165, 255)
            elif not tracking_active:
                if selected_det is not None and not ambiguous_detection:
                    can_initialize = True
                    if recognizer_enabled:
                        if not identity.target_ready:
                            identity_candidate_index, recognition_elapsed, _ = run_identity_recognition(
                                recognizer, identity, frame_index, frame_bgr, [selected_det], True, metrics, danger=True)
                            identity_face_found = identity.last_face_found
                            identity_score = identity.last_similarity
                            can_initialize = identity.target_ready and identity_candidate_index == 0
                            if not can_initialize:
                                state = "WAITING_FACE"
                                color = (0, 128, 255)
                                final_box = selected_det["box_xywh"][:]
                        else:
                            identity_candidate_index, recognition_elapsed, _ = run_identity_recognition(
                                recognizer, identity, frame_index, frame_bgr, detections, False, metrics, danger=True)
                            identity_face_found = identity.last_face_found
                            identity_score = identity.last_similarity
                            if identity_candidate_index >= 0 and identity_candidate_index < len(detections):
                                selected_det = detections[identity_candidate_index]
                                det_conf, best_detector_iou, best_center_distance_ratio, best_area_ratio = detector_match_metrics(tracker_box, selected_det, frame_diag)
                            can_initialize = identity.state == IdentityState.VERIFIED and identity_candidate_index >= 0
                            if not can_initialize:
                                state = "SEARCHING_TARGET"
                                color = (0, 128, 255)
                                final_box = selected_det["box_xywh"][:]
                    if can_initialize:
                        final_box = selected_det["box_xywh"][:]
                        standby_match_iou = float("nan")
                        standby_match_center_ratio = float("nan")
                        standby_match_area_ratio = float("nan")
                        if standby_warmed and standby_box is not None:
                            _, standby_match_iou, standby_match_center_ratio, standby_match_area_ratio = detector_match_metrics(standby_box, selected_det, frame_diag)
                        standby_matches_selected = standby_warmed and standby_prev_output and standby_box is not None and detector_confirms_track(
                            standby_match_iou,
                            standby_match_center_ratio,
                            standby_match_area_ratio,
                            args,
                        )
                        if standby_matches_selected:
                            prev_output = dict(standby_prev_output)
                            current_box = standby_box[:]
                            previous_box = standby_box[:]
                            tracking_active = True
                            initializing = False
                            standby_warmed = False
                            standby_prev_output = {}
                            standby_box = None
                            metrics["eco_standby_warmed"] = 0
                            metrics["eco_first_live_handoff_mode"] = "adopt_standby"
                            metrics["eco_first_live_acquisition_time_s"] = 0.0
                            first_live_acquisition_recorded = True
                            standby_handoff_pending = False
                            state = "DETECTOR_CONFIRMED"
                            color = (0, 255, 0)
                            print("standby_live_adopt_bbox=%.3f,%.3f,%.3f,%.3f det_conf=%.3f identity=%s id_score=%.3f match_iou=%.3f" % tuple(final_box + [selected_det["conf"], identity.state, identity.last_similarity, standby_match_iou]), flush=True)
                        else:
                            init_worker = TrackerInitializer(tracker, frame_bgr, selected_det["box_xywh"])
                            init_worker.start()
                            tracking_active = False
                            initializing = True
                            prev_output = {}
                            current_box = None
                            previous_box = None
                            standby_handoff_pending = bool(standby_warmed and standby_prev_output)
                            standby_warmed = False
                            standby_prev_output = {}
                            standby_box = None
                            metrics["eco_standby_warmed"] = 0
                            metrics["eco_first_live_handoff_mode"] = "initialize"
                            state = "DETECTED_INITIALIZING"
                            color = (0, 165, 255)
                            print("auto_initializing_bbox=%.3f,%.3f,%.3f,%.3f det_conf=%.3f identity=%s id_score=%.3f" % tuple(final_box + [selected_det["conf"], identity.state, identity.last_similarity]), flush=True)
                elif selected_det is not None and ambiguous_detection:
                    state = "AMBIGUOUS_DETECTION"
                    color = (0, 128, 255)
                    metrics["ambiguous_reinit_skips"] += 1
                    final_box = selected_det["box_xywh"][:]
                else:
                    state = "SEARCHING"
                    color = (0, 255, 255)
            else:
                detector_confirmed = False
                detector_conflict = False
                identity_drop_current = False
                identity_checked_current_box = False
                in_reinit_cooldown = frame_index - last_reinit_frame < int(args.reinit_cooldown_frames)
                if selected_det is not None:
                    detector_confirmed = detector_confirms_track(best_detector_iou, best_center_distance_ratio, best_area_ratio, args)
                    detector_conflict = not detector_confirmed
                    if detector_confirmed:
                        metrics["detector_confirmed_frames"] += 1
                        consecutive_detector_missing = 0
                    elif math.isfinite(score) and score >= float(args.low_score_threshold):
                        if not was_background_locked:
                            metrics["background_lock_events"] += 1
                        was_background_locked = True
                elif need_detection:
                    detector_conflict = True

                identity_danger = (
                    is_low_score
                    or is_very_low_score
                    or detector_conflict
                    or was_lost
                    or was_background_locked
                    or consecutive_suspicious >= int(args.suspect_frames)
                    or is_large_jump
                    or is_large_area_change
                )
                identity.refresh_staleness(frame_index, danger=identity_danger)
                if recognizer_enabled and identity.target_ready and identity.should_check(
                    frame_index,
                    args.recognize_interval,
                    args.recognize_fast_interval,
                    danger=identity_danger,
                ):
                    identity_candidates = detections[:] if detections else []
                    identity_candidate_is_detector = len(identity_candidates) > 0
                    if not identity_candidates and final_box is not None:
                        identity_candidates = [{"box_xywh": final_box, "conf": float("nan")}]
                        identity_checked_current_box = True
                    if identity_candidates:
                        identity_candidate_index, recognition_elapsed, _ = run_identity_recognition(
                            recognizer, identity, frame_index, frame_bgr, identity_candidates, False, metrics, danger=identity_danger)
                        identity_face_found = identity.last_face_found
                        identity_score = identity.last_similarity
                        if identity_candidate_index >= 0 and identity_candidate_is_detector and identity_candidate_index < len(detections):
                            selected_det = detections[identity_candidate_index]
                            ambiguous_detection = False
                            identity_verified_detection = True
                            det_conf, best_detector_iou, best_center_distance_ratio, best_area_ratio = detector_match_metrics(tracker_box, selected_det, frame_diag)
                            detector_confirmed = detector_confirms_track(best_detector_iou, best_center_distance_ratio, best_area_ratio, args)
                            detector_conflict = not detector_confirmed
                            consecutive_detector_missing = 0
                            last_detector_box = selected_det["box_xywh"][:]
                        elif identity_checked_current_box and identity.hard_rejected():
                            identity_drop_current = True

                force_hard_reinit = False
                force_shrink_reinit = False
                shrink_confirmed = False
                rescue_reinit = False
                allow_reinit = selected_det is not None and not ambiguous_detection
                if allow_reinit:
                    if is_out_of_frame:
                        force_hard_reinit = True
                        rescue_reinit = True
                        hard_reinit_reasons.append("out_of_frame")
                    if consecutive_suspicious >= int(args.lost_frames):
                        force_hard_reinit = True
                        rescue_reinit = True
                        hard_reinit_reasons.append("lost_suspicious")
                    if math.isfinite(best_detector_iou) and best_detector_iou < float(args.reinit_iou_threshold) and (is_low_score or was_lost or was_background_locked):
                        force_hard_reinit = True
                        rescue_reinit = was_lost or was_background_locked
                        hard_reinit_reasons.append("low_iou")
                        if was_background_locked:
                            hard_reinit_reasons.append("background_lock")
                    if math.isfinite(best_area_ratio) and best_area_ratio > float(args.size_ratio_threshold) and (is_low_score or detector_conflict):
                        force_hard_reinit = True
                        rescue_reinit = was_lost or was_background_locked
                        hard_reinit_reasons.append("area_ratio")
                        if was_background_locked:
                            hard_reinit_reasons.append("background_lock")
                    if detector_confirmed and tracker_box is not None:
                        tracker_area = bbox_area(tracker_box)
                        det_area = bbox_area(selected_det["box_xywh"])
                        tracker_larger = tracker_area > det_area * float(args.shrink_ratio_threshold)
                        center_close = math.isfinite(best_center_distance_ratio) and best_center_distance_ratio <= float(args.confirm_center_threshold)
                        if tracker_larger and center_close:
                            shrink_confirm_count += 1
                            metrics["shrink_confirm_count_max"] = max(metrics["shrink_confirm_count_max"], shrink_confirm_count)
                        else:
                            shrink_confirm_count = 0
                        shrink_confirmed = shrink_confirm_count >= int(args.shrink_confirm_frames)
                        if tracker_larger and center_close and (shrink_confirmed or is_low_score or was_lost or was_background_locked):
                            force_shrink_reinit = True
                    else:
                        shrink_confirm_count = 0
                    if identity_verified_detection and tracker_box is not None:
                        verified_low_overlap = math.isfinite(best_detector_iou) and best_detector_iou < float(args.confirm_iou_threshold)
                        verified_far_center = math.isfinite(best_center_distance_ratio) and best_center_distance_ratio > float(args.control_center_threshold)
                        if verified_low_overlap or verified_far_center or is_low_score or was_lost or was_background_locked:
                            force_hard_reinit = True
                            rescue_reinit = True
                            hard_reinit_reasons.append("identity_verified")
                else:
                    shrink_confirm_count = 0

                soft_reinit_requested = force_shrink_reinit or (detector_confirmed and is_low_score and (is_large_jump or is_large_area_change))
                if recognizer_enabled and identity.target_ready and (force_hard_reinit or soft_reinit_requested):
                    identity_allows_reinit = identity.state == IdentityState.VERIFIED
                    if args.allow_probation_reinit and identity.state in (IdentityState.PROBATION, IdentityState.STALE) and not identity.last_face_found:
                        identity_allows_reinit = True
                    if not identity_allows_reinit:
                        identity.block_reinit()
                        metrics["identity_reinit_blocks"] = identity.reinit_blocks
                        force_hard_reinit = False
                        force_shrink_reinit = False
                        soft_reinit_requested = False
                shrink_bypass_cooldown = force_shrink_reinit and shrink_confirmed
                if in_reinit_cooldown and allow_reinit and not rescue_reinit and not shrink_bypass_cooldown and (force_hard_reinit or soft_reinit_requested):
                    metrics["reinit_cooldown_skips"] += 1
                    if force_shrink_reinit:
                        metrics["shrink_reinit_cooldown_skips"] += 1
                    force_hard_reinit = False
                    force_shrink_reinit = False
                    soft_reinit_requested = False

                if identity_drop_current:
                    state = "IDENTITY_REJECTED"
                    color = (0, 0, 255)
                    tracking_active = False
                    current_box = None
                    previous_box = None
                    control_box = None
                    control_box_source = "none"
                    prev_output = {}
                    final_box = None
                    consecutive_suspicious = 0
                    consecutive_detector_missing = 0
                elif allow_reinit and force_hard_reinit:
                    final_box = selected_det["box_xywh"][:]
                    init_worker = TrackerInitializer(tracker, frame_bgr, final_box)
                    init_worker.start()
                    initializing = True
                    tracking_active = False
                    previous_box = None
                    consecutive_suspicious = 0
                    consecutive_detector_missing = 0
                    shrink_confirm_count = 0
                    last_reinit_frame = frame_index
                    was_lost = False
                    was_background_locked = False
                    metrics["hard_reinitializations"] += 1
                    if "out_of_frame" in hard_reinit_reasons:
                        metrics["hard_reinit_out_of_frame"] += 1
                    if "lost_suspicious" in hard_reinit_reasons:
                        metrics["hard_reinit_lost_suspicious"] += 1
                    if "low_iou" in hard_reinit_reasons:
                        metrics["hard_reinit_low_iou"] += 1
                    if "area_ratio" in hard_reinit_reasons:
                        metrics["hard_reinit_area_ratio"] += 1
                    if "background_lock" in hard_reinit_reasons:
                        metrics["hard_reinit_background_lock"] += 1
                    if "identity_verified" in hard_reinit_reasons:
                        metrics["hard_reinit_identity_verified"] += 1
                    state = "REINITIALIZING"
                    color = (255, 0, 255)
                    print("hard_reinit_bbox=%.3f,%.3f,%.3f,%.3f det_conf=%.3f" % tuple(final_box + [selected_det["conf"]]), flush=True)
                elif allow_reinit and soft_reinit_requested:
                    final_box = selected_det["box_xywh"][:] if force_shrink_reinit else blend_boxes(tracker_box, selected_det["box_xywh"], args.detector_weight)
                    init_worker = TrackerInitializer(tracker, frame_bgr, final_box)
                    init_worker.start()
                    initializing = True
                    tracking_active = False
                    previous_box = None
                    shrink_confirm_count = 0
                    last_reinit_frame = frame_index
                    metrics["soft_reinitializations"] += 1
                    state = "SOFT_REINITIALIZING"
                    color = (255, 0, 255)
                elif selected_det is not None and ambiguous_detection and (is_low_score or was_lost or was_background_locked):
                    metrics["ambiguous_reinit_skips"] += 1
                    state = "AMBIGUOUS_REDETECT"
                    color = (0, 128, 255)
                elif is_out_of_frame or consecutive_suspicious >= int(args.lost_frames):
                    state = "LOST"
                    color = (0, 0, 255)
                    if not was_lost:
                        metrics["lost_events"] += 1
                    was_lost = True
                elif was_background_locked and detector_conflict:
                    state = "BACKGROUND_LOCK"
                    color = (0, 0, 255)
                elif consecutive_suspicious >= int(args.suspect_frames) or detector_conflict:
                    state = "SUSPECTED_LOST"
                    color = (0, 128, 255)
                elif weak_confidence:
                    state = "LOW_CONFIDENCE"
                    color = (0, 165, 255)
                elif detector_confirmed:
                    state = "DETECTOR_CONFIRMED"
                    color = (0, 255, 0)
                else:
                    state = "TRACKING"
                    color = (0, 255, 0)

                if state == "LOW_CONFIDENCE":
                    metrics["low_confidence_frames"] += 1
                if state in ("SUSPECTED_LOST", "BACKGROUND_LOCK", "AMBIGUOUS_REDETECT"):
                    metrics["suspected_lost_frames"] += 1
                if state == "LOST":
                    metrics["lost_frames"] += 1
                if identity.state == IdentityState.VERIFIED:
                    metrics["identity_verified_frames"] += 1
                elif identity.state == IdentityState.STALE:
                    metrics["identity_stale_frames"] += 1
                elif identity.state in (IdentityState.PROBATION, IdentityState.WAITING_FACE):
                    metrics["identity_probation_frames"] += 1
                elif identity.state == IdentityState.REJECTED:
                    metrics["identity_rejected_frames"] += 1
                if final_box is not None:
                    current_box = final_box[:]
                    previous_box = final_box[:]

            control_candidate = None
            control_source = "none"
            tracker_control_states = ("TRACKING", "DETECTOR_CONFIRMED")
            unsafe_control_states = (
                "LOW_CONFIDENCE", "SUSPECTED_LOST", "BACKGROUND_LOCK", "LOST",
                "AMBIGUOUS_REDETECT", "AMBIGUOUS_DETECTION", "REINITIALIZING",
                "SOFT_REINITIALIZING", "INITIALIZING", "SEARCHING", "WAITING_FACE",
                "SEARCHING_TARGET", "IDENTITY_REJECTED",
            )
            detector_control_states = ("DETECTED_INITIALIZING", "REINITIALIZING", "SOFT_REINITIALIZING")
            detector_rescue_states = ("SUSPECTED_LOST", "BACKGROUND_LOCK", "LOST")
            detector_can_update_control = (
                selected_det is not None
                and not ambiguous_detection
                and (
                    control_box is None
                    or detector_confirmed
                    or state in detector_control_states
                    or (
                        state in detector_rescue_states
                        and (
                            rescue_reinit
                            or is_low_score
                            or was_lost
                            or was_background_locked
                            or consecutive_suspicious >= int(args.suspect_frames)
                        )
                    )
                )
            )
            if detector_can_update_control:
                control_candidate = selected_det["box_xywh"][:]
                control_source = "detector"
            elif final_box is not None and state in tracker_control_states:
                tracker_control_ok = True
                reference_box = control_box if control_box is not None else last_detector_box
                if reference_box is not None:
                    control_center_ratio = center_distance(reference_box, final_box) / frame_diag
                    tracker_control_ok = (
                        area_ratio(reference_box, final_box) <= float(args.shrink_ratio_threshold)
                        and control_center_ratio <= float(args.control_center_threshold)
                    )
                if tracker_control_ok:
                    control_candidate = final_box[:]
                    control_source = "tracker"
                elif control_box is not None:
                    control_source = "held"
            elif control_box is not None and state in unsafe_control_states:
                control_source = "held"

            if control_candidate is not None:
                if control_box is None or control_source == "detector":
                    control_box = control_candidate[:]
                else:
                    control_box = blend_boxes(control_box, control_candidate, args.control_box_smoothing)
                control_box = clip_xywh(control_box, frame_bgr.shape)
                control_box_source = control_source
            elif control_box is not None:
                control_box_source = "held"

            if control_box is not None:
                metrics["control_box_frames"] += 1
                if control_box_source == "detector":
                    metrics["control_box_from_detector"] += 1
                elif control_box_source == "tracker":
                    metrics["control_box_from_tracker"] += 1
                elif control_box_source == "held":
                    metrics["control_box_held"] += 1

            wheel_control = compute_wheel_control(
                control_box,
                frame_width,
                frame_height,
                state,
                identity.state,
                recognizer_enabled,
                metrics["eco_ready_warmed"],
                args,
            )
            if wheel_control["enabled"]:
                if wheel_control["allowed"]:
                    metrics["wheel_control_allowed_frames"] += 1
                    metrics["wheel_linear_abs_sum"] += abs(wheel_control["linear_cmd"])
                    metrics["wheel_angular_abs_sum"] += abs(wheel_control["angular_cmd"])
                    metrics["wheel_left_abs_max"] = max(metrics["wheel_left_abs_max"], abs(wheel_control["left_cmd"]))
                    metrics["wheel_right_abs_max"] = max(metrics["wheel_right_abs_max"], abs(wheel_control["right_cmd"]))
                else:
                    metrics["wheel_control_blocked_frames"] += 1

            for det in detections:
                label = "DET %.2f" % det.get("conf", float("nan"))
                draw_box(display, det["box_xywh"], (255, 180, 0), 1, label)
            if selected_det is not None:
                draw_box(display, selected_det["box_xywh"], (255, 0, 0), 2, "SELECTED")
            if tracker_box is not None and final_box is not None and tracker_box is not final_box:
                draw_box(display, tracker_box, (180, 180, 180), 1, "TRACK")
            if final_box is not None:
                draw_box(display, final_box, color, 2, "FINAL")
            if control_box is not None:
                draw_box(display, control_box, (255, 255, 255), 2, "CONTROL:%s" % control_box_source.upper())
            if identity.last_face_found and identity.last_face_box is not None:
                draw_box(display, identity.last_face_box, (255, 255, 0), 1, "FACE")
            draw_label(display, state, (12, 82), color)
            if recognizer_enabled:
                id_color = (0, 255, 0) if identity.state == IdentityState.VERIFIED else ((0, 0, 255) if identity.state == IdentityState.REJECTED else (0, 165, 255))
                draw_label(display, "ID:%s" % identity.state, (12, 108), id_color)
            if args.wheel_log_only:
                draw_label(display, "WHEEL LOG L/R=%.2f/%.2f" % (wheel_control["left_cmd"], wheel_control["right_cmd"]), (12, 134), (200, 200, 255))

            now = time.perf_counter()
            fps_value = 1.0 / max(now - last_time, 1e-9)
            last_time = now
            draw_header(display, frame_index, fps_value, state, score, det_count, det_conf, best_detector_iou, identity.state if recognizer_enabled else None, identity.last_similarity)
            cv2.imshow(args.window_name, display)

            if args.save_video and output_dir is not None and video_writer is None:
                height, width = display.shape[:2]
                out_fps = float(cap.get(cv2.CAP_PROP_FPS))
                if out_fps <= 0:
                    out_fps = float(args.fps)
                video_writer = cv2.VideoWriter(str(output_dir / "camera_detector_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
            if video_writer is not None:
                video_writer.write(display)

            if writer_csv is not None:
                if tracker_box is None:
                    tracker_values = [float("nan")] * 4
                else:
                    tracker_values = tracker_box
                if final_box is None:
                    final_values = [float("nan")] * 4
                else:
                    final_values = final_box
                if control_box is None:
                    control_values = [float("nan")] * 4
                    control_value_source = "none"
                else:
                    control_values = control_box
                    control_value_source = control_box_source
                if selected_det is None:
                    det_values = [float("nan")] * 4
                else:
                    det_values = selected_det["box_xywh"]
                writer_csv.writerow([
                    frame_index, now - run_start,
                    source_info["source_frame_index"], source_info["source_timestamp_s"], source_info["source_fps"], source_info["source_is_video"],
                    tracker_values[0], tracker_values[1], tracker_values[2], tracker_values[3],
                    final_values[0], final_values[1], final_values[2], final_values[3],
                    control_values[0], control_values[1], control_values[2], control_values[3], control_value_source,
                    det_values[0], det_values[1], det_values[2], det_values[3],
                    score, det_conf, det_count, best_detector_iou, best_center_distance_ratio, best_area_ratio,
                    state, track_elapsed, detect_elapsed, init_time_for_row, fps_value,
                    int(is_low_score), int(is_very_low_score), int(is_out_of_frame), center_delta, center_delta_ratio,
                    adjacent_area_ratio, int(is_large_jump), int(is_large_area_change), consecutive_suspicious,
                    int(ambiguous_detection), metrics["soft_reinitializations"], metrics["hard_reinitializations"], metrics["background_lock_events"],
                    identity.state, identity.last_similarity, int(identity.last_face_found), identity.last_candidate_index,
                    int(identity.target_ready), recognition_elapsed, identity.reinit_blocks,
                    metrics["eco_ready_warmed"], int(standby_warmed), metrics["eco_first_live_handoff_mode"],
                    wheel_control["enabled"], wheel_control["allowed"], wheel_control["center_error_norm"], wheel_control["distance_error_norm"],
                    wheel_control["linear_cmd"], wheel_control["angular_cmd"], wheel_control["left_cmd"], wheel_control["right_cmd"], wheel_control["reason"],
                ])
                csv_file.flush()

            metrics["frames_total"] += 1
            if output_dir is not None and args.metrics_interval > 0 and metrics["frames_total"] % int(args.metrics_interval) == 0:
                metrics["end_time"] = time.perf_counter()
                write_metrics_summary(output_dir, metrics)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                force_redetect = True
                tracking_active = False
                initializing = False
                current_box = None
                previous_box = None
                control_box = None
                control_box_source = "none"
                prev_output = {}
                consecutive_suspicious = 0
                consecutive_detector_missing = 0
                if recognizer_enabled:
                    identity.reset(keep_target=True)
                print("force_redetect_requested", flush=True)
            elif key in (ord("s"), ord("S")):
                if output_dir is None:
                    screenshot_path = Path("tracking_screenshot_%06d.jpg" % frame_index).resolve()
                else:
                    screenshot_dir = output_dir / "screenshots"
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    screenshot_path = screenshot_dir / ("frame_%06d_%s.jpg" % (frame_index, state.lower()))
                cv2.imwrite(str(screenshot_path), display)
                print("screenshot=%s" % screenshot_path, flush=True)
            else:
                force_redetect = False

            frame_index += 1
    finally:
        metrics["end_time"] = time.perf_counter()
        cap.release()
        detector.close()
        recognizer.close()
        if video_writer is not None:
            video_writer.release()
        if csv_file is not None:
            csv_file.close()
        write_metrics_summary(output_dir, metrics)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)
