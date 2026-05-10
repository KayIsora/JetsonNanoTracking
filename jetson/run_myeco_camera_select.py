#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2

from run_myeco_detector_camera import CppNcnnStdinDetector, HOGPersonDetector, PERSON_MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Run MyECO on a live camera and press I to select/reselect the initial bbox.")
    parser.add_argument("--camera", default="0", help="Camera index, video path, or GStreamer pipeline.")
    parser.add_argument("--gstreamer", action="store_true", help="Open --camera as a GStreamer pipeline.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width for index cameras.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height for index cameras.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS and output FPS fallback.")
    parser.add_argument("--tracker-name", default="eco", help="PyTracking tracker name.")
    parser.add_argument("--param", default="verified_otb936_run_update", help="PyTracking parameter name.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for CSV/video output.")
    parser.add_argument("--save-video", action="store_true", help="Save annotated camera output when --output-dir is set.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit. 0 means run until Q/Esc.")
    parser.add_argument("--window-name", default="MyECO Camera Tracking", help="OpenCV display window name.")
    parser.add_argument("--init-width-ratio", type=float, default=0.28, help="Center init bbox width as a ratio of frame width.")
    parser.add_argument("--init-height-ratio", type=float, default=0.70, help="Center init bbox height as a ratio of frame height.")
    parser.add_argument("--startup-detector-backend", choices=["off", "cpp_ncnn", "hog"], default="off", help="Use a person detector only during startup to initialize ECO from the centered target, then disable detection.")
    parser.add_argument("--startup-detector-bin", type=Path, default=Path("person_detection_update") / "pedestrian_detection" / "build" / "detect_person_stdin", help="C++ NCNN detector executable used only for startup initialization.")
    parser.add_argument("--startup-detector-model-dir", type=Path, default=PERSON_MODEL_DIR, help="Directory containing the NCNN person detector model used only for startup initialization.")
    parser.add_argument("--startup-detector-conf", type=float, default=0.70, help="Person detector confidence threshold for startup initialization.")
    parser.add_argument("--startup-detect-frames", type=int, default=30, help="Maximum READY frames to run startup detector before falling back to manual/center init.")
    parser.add_argument("--startup-detect-min-hits", type=int, default=2, help="Require this many centered detector hits before auto-initializing ECO.")
    parser.add_argument("--startup-center-threshold", type=float, default=0.45, help="Maximum detector center distance from frame center as a frame diagonal ratio for startup target selection.")
    parser.add_argument("--low-score-threshold", type=float, default=0.50, help="Tracker score threshold for LOW_CONFIDENCE state.")
    parser.add_argument("--very-low-score-threshold", type=float, default=0.25, help="Tracker score threshold for SUSPECTED_LOST/LOST state.")
    parser.add_argument("--lost-frames", type=int, default=20, help="Consecutive suspicious frames before LOST state.")
    parser.add_argument("--suspect-frames", type=int, default=10, help="Consecutive suspicious frames before SUSPECTED_LOST state.")
    parser.add_argument("--bbox-margin", type=float, default=0.15, help="Allowed bbox margin outside frame as a ratio of bbox size.")
    parser.add_argument("--jump-threshold", type=float, default=0.45, help="Center jump threshold as a ratio of frame diagonal.")
    parser.add_argument("--area-change-threshold", type=float, default=2.80, help="Large bbox area-ratio threshold between adjacent tracking frames.")
    parser.add_argument("--metrics-interval", type=int, default=30, help="Write camera_metrics.txt every N frames. 0 means only on exit.")
    return parser.parse_args()


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


def is_video_file_source(camera_arg, use_gstreamer):
    if use_gstreamer:
        return False
    try:
        int(camera_arg)
        return False
    except ValueError:
        pass
    return Path(camera_arg).expanduser().is_file()


def resolve_runtime_path(project_root, path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def load_startup_detector(project_root, args, output_dir):
    if args.startup_detector_backend == "off":
        return None
    if args.startup_detector_backend == "cpp_ncnn":
        detector_bin = resolve_runtime_path(project_root, args.startup_detector_bin)
        model_dir = resolve_runtime_path(project_root, args.startup_detector_model_dir)
        stderr_path = output_dir / "cpp_startup_detector_stderr.log" if output_dir is not None else None
        return CppNcnnStdinDetector(detector_bin, model_dir, stderr_path)
    if args.startup_detector_backend == "hog":
        return HOGPersonDetector(8, 1.05)
    raise RuntimeError("Unsupported startup detector backend: %s" % args.startup_detector_backend)


def draw_box(image, box, color):
    x, y, w, h = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


def get_center_init_box(frame_shape, width_ratio, height_ratio):
    frame_h, frame_w = frame_shape[:2]
    box_w = max(2.0, float(frame_w) * float(width_ratio))
    box_h = max(2.0, float(frame_h) * float(height_ratio))
    x = (float(frame_w) - box_w) / 2.0
    y = (float(frame_h) - box_h) / 2.0
    return [x, y, box_w, box_h]


def bbox_center(box):
    x, y, w, h = [float(v) for v in box]
    return x + w / 2.0, y + h / 2.0


def center_distance_to_frame(box, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    cx, cy = bbox_center(box)
    return math.hypot(cx - float(frame_w) / 2.0, cy - float(frame_h) / 2.0)


def bbox_area(box):
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def bbox_out_of_frame(box, frame_shape, margin_ratio):
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = [float(v) for v in box]
    margin_x = max(0.0, w * float(margin_ratio))
    margin_y = max(0.0, h * float(margin_ratio))
    return x + w < -margin_x or y + h < -margin_y or x > frame_w + margin_x or y > frame_h + margin_y


def finite_or_nan(value):
    if value is None:
        return float("nan")
    value = float(value)
    if math.isfinite(value):
        return value
    return float("nan")


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


def draw_status_label(image, text, color):
    cv2.putText(image, text, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_header(image, frame_index, fps_value, state, score):
    score_text = "nan" if score is None else "%.3f" % score
    text = "frame=%d fps=%.2f state=%s score=%s" % (frame_index, fps_value, state, score_text)
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "Startup detector selects initial target; press I for manual init   Q/Esc: quit", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)


def select_initial_bbox(frame, width_ratio, height_ratio):
    return get_center_init_box(frame.shape, width_ratio, height_ratio)


def select_startup_detection(detections, frame_shape, center_threshold):
    if not detections:
        return None, float("nan"), float("nan")
    frame_h, frame_w = frame_shape[:2]
    frame_diag = max(1.0, math.hypot(float(frame_w), float(frame_h)))
    frame_area = max(1.0, float(frame_w * frame_h))
    best_det = None
    best_rank = -1e9
    best_center_ratio = float("nan")
    for det in detections:
        box = det["box_xywh"]
        center_ratio = center_distance_to_frame(box, frame_shape) / frame_diag
        if center_ratio > float(center_threshold):
            continue
        area_score = min(1.0, bbox_area(box) / (frame_area * 0.60))
        center_score = max(0.0, 1.0 - center_ratio / max(float(center_threshold), 1e-9))
        rank = 1.4 * center_score + 0.7 * float(det.get("conf", 0.0)) + 0.2 * area_score
        if rank > best_rank:
            best_det = det
            best_rank = rank
            best_center_ratio = center_ratio
    return best_det, float(best_rank), best_center_ratio


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


def initialize_tracker_sync(tracker, frame_bgr, init_box):
    start = time.perf_counter()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = tracker.initialize(frame_rgb, {"init_bbox": list(map(float, init_box))}) or {}
    return out, time.perf_counter() - start


def write_metrics_summary(output_dir, metrics):
    if output_dir is None:
        return
    duration_s = max(metrics["end_time"] - metrics["start_time"], 1e-9)
    track_frames = max(metrics["track_frames"], 1)
    init_times = metrics["init_times"]
    scores = metrics["scores"]
    center_deltas = metrics["center_deltas"]
    area_ratios = metrics["area_ratios"]
    summary_path = output_dir / "camera_metrics.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("camera=%s\n" % metrics["camera"])
        f.write("tracker=%s\n" % metrics["tracker"])
        f.write("param=%s\n" % metrics["param"])
        f.write("frame_width=%d\n" % metrics["frame_width"])
        f.write("frame_height=%d\n" % metrics["frame_height"])
        f.write("low_score_threshold=%.6f\n" % metrics["low_score_threshold"])
        f.write("very_low_score_threshold=%.6f\n" % metrics["very_low_score_threshold"])
        f.write("suspect_frames_threshold=%d\n" % metrics["suspect_frames_threshold"])
        f.write("lost_frames_threshold=%d\n" % metrics["lost_frames_threshold"])
        f.write("jump_threshold=%.6f\n" % metrics["jump_threshold"])
        f.write("area_change_threshold=%.6f\n" % metrics["area_change_threshold"])
        f.write("startup_detector_backend=%s\n" % metrics["startup_detector_backend"])
        f.write("startup_detector_conf=%.6f\n" % metrics["startup_detector_conf"])
        f.write("startup_detect_frames=%d\n" % metrics["startup_detect_frames"])
        f.write("startup_detector_calls=%d\n" % metrics["startup_detector_calls"])
        f.write("startup_detector_hits=%d\n" % metrics["startup_detector_hits"])
        f.write("startup_detector_initializations=%d\n" % metrics["startup_detector_initializations"])
        f.write("startup_detector_selected_conf=%.6f\n" % metrics["startup_detector_selected_conf"])
        f.write("tracker_load_time_s=%.6f\n" % metrics["tracker_load_time_s"])
        f.write("duration_s=%.6f\n" % duration_s)
        f.write("frames_total=%d\n" % metrics["frames_total"])
        f.write("overall_loop_fps=%.6f\n" % (float(metrics["frames_total"]) / duration_s))
        f.write("track_frames=%d\n" % metrics["track_frames"])
        f.write("tracking_fps_excluding_idle=%.6f\n" % (float(track_frames) / max(metrics["track_time_total_s"], 1e-9)))
        f.write("track_time_total_s=%.6f\n" % metrics["track_time_total_s"])
        f.write("track_time_avg_s=%.6f\n" % (metrics["track_time_total_s"] / float(track_frames)))
        f.write("manual_reinitializations=%d\n" % metrics["manual_reinitializations"])
        f.write("initializations_completed=%d\n" % metrics["initializations_completed"])
        f.write("init_time_avg_s=%.6f\n" % (sum(init_times) / float(len(init_times)) if init_times else 0.0))
        f.write("init_time_max_s=%.6f\n" % (max(init_times) if init_times else 0.0))
        f.write("score_count=%d\n" % len(scores))
        f.write("score_min=%.6f\n" % (min(scores) if scores else float("nan")))
        f.write("score_p10=%.6f\n" % percentile(scores, 10))
        f.write("score_p25=%.6f\n" % percentile(scores, 25))
        f.write("score_median=%.6f\n" % percentile(scores, 50))
        f.write("score_p75=%.6f\n" % percentile(scores, 75))
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
        f.write("lost_events=%d\n" % metrics["lost_events"])
        f.write("max_consecutive_suspicious=%d\n" % metrics["max_consecutive_suspicious"])
        f.flush()
    print("metrics_summary=%s" % summary_path, flush=True)


def main():
    args = parse_args()
    output_dir = None
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print("output_dir=%s" % output_dir, flush=True)

    print("Resolving project root...", flush=True)
    project_root = resolve_project_root(Path(__file__).resolve().parent)
    print("project_root=%s" % project_root, flush=True)

    print("Creating tracker %s/%s..." % (args.tracker_name, args.param), flush=True)
    tracker_load_start = time.perf_counter()
    tracker = create_tracker(project_root, args.tracker_name, args.param)
    tracker_load_time = time.perf_counter() - tracker_load_start
    print("Tracker ready in %.3fs." % tracker_load_time, flush=True)

    startup_detector = load_startup_detector(project_root, args, output_dir)
    if startup_detector is not None:
        print("Startup detector ready backend=%s." % args.startup_detector_backend, flush=True)

    print("Opening camera source: %s" % args.camera, flush=True)
    video_file_source = is_video_file_source(args.camera, args.gstreamer)
    if video_file_source:
        print("Video file source detected; initialization will run synchronously to avoid skipping source frames.", flush=True)
    cap = open_capture(args.camera, args.gstreamer, args.width, args.height, args.fps)
    print("Camera opened.", flush=True)

    csv_file = None
    writer_csv = None
    video_writer = None
    if output_dir is not None:
        csv_file = (output_dir / "camera_predictions.csv").open("w", newline="", encoding="utf-8")
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow([
            "frame_index", "timestamp_s", "x", "y", "w", "h", "center_x", "center_y", "area",
            "score", "state", "track_time_s", "init_time_s", "fps", "is_low_score", "is_very_low_score",
            "is_out_of_frame", "center_delta_px", "center_delta_ratio", "area_ratio", "is_large_jump",
            "is_large_area_change", "consecutive_suspicious", "manual_reinitializations",
            "init_trigger", "startup_det_x", "startup_det_y", "startup_det_w", "startup_det_h",
            "startup_det_conf", "startup_detector_calls", "startup_detector_hits",
        ])

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    tracking_active = False
    initializing = False
    init_worker = None
    prev_output = {}
    current_box = None
    previous_box = None
    frame_index = 0
    last_time = time.perf_counter()
    run_start = time.perf_counter()
    consecutive_suspicious = 0
    was_lost = False
    last_init_time = 0.0
    startup_detector_calls = 0
    startup_detector_hits = 0
    startup_detector_used = False
    metrics = {
        "camera": args.camera,
        "tracker": args.tracker_name,
        "param": args.param,
        "frame_width": 0,
        "frame_height": 0,
        "low_score_threshold": float(args.low_score_threshold),
        "very_low_score_threshold": float(args.very_low_score_threshold),
        "suspect_frames_threshold": int(args.suspect_frames),
        "lost_frames_threshold": int(args.lost_frames),
        "jump_threshold": float(args.jump_threshold),
        "area_change_threshold": float(args.area_change_threshold),
        "startup_detector_backend": args.startup_detector_backend,
        "startup_detector_conf": float(args.startup_detector_conf),
        "startup_detect_frames": int(args.startup_detect_frames),
        "startup_detector_calls": 0,
        "startup_detector_hits": 0,
        "startup_detector_initializations": 0,
        "startup_detector_selected_conf": float("nan"),
        "tracker_load_time_s": tracker_load_time,
        "start_time": run_start,
        "end_time": run_start,
        "frames_total": 0,
        "track_frames": 0,
        "track_time_total_s": 0.0,
        "manual_reinitializations": 0,
        "initializations_completed": 0,
        "init_times": [],
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
        "lost_events": 0,
        "max_consecutive_suspicious": 0,
    }

    try:
        while True:
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("Camera frame read failed.", file=sys.stderr)
                break

            frame_height, frame_width = frame_bgr.shape[:2]
            frame_diag = math.hypot(float(frame_width), float(frame_height))
            metrics["frame_width"] = frame_width
            metrics["frame_height"] = frame_height
            display = frame_bgr.copy()
            track_elapsed = 0.0
            mode = "READY"
            score = None
            is_low_score = False
            is_very_low_score = False
            is_out_of_frame = False
            is_large_jump = False
            is_large_area_change = False
            center_delta = 0.0
            center_delta_ratio = 0.0
            area_ratio = 1.0
            init_time_for_row = 0.0
            init_trigger = "none"
            startup_selected_det = None
            startup_selected_conf = float("nan")
            init_box_preview = get_center_init_box(frame_bgr.shape, args.init_width_ratio, args.init_height_ratio)

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
                was_lost = False
                last_init_time = init_worker.elapsed
                init_time_for_row = last_init_time
                metrics["initializations_completed"] += 1
                metrics["init_times"].append(last_init_time)
                print("initialized_bbox=%.3f,%.3f,%.3f,%.3f init_time_s=%.3f" % tuple(current_box + [last_init_time]), flush=True)
                init_worker = None

            if tracking_active:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                start = time.perf_counter()
                out = tracker.track(frame_rgb, {"previous_output": prev_output}) or {}
                track_elapsed = time.perf_counter() - start
                prev_output = dict(out)
                previous_for_metrics = previous_box
                current_box = [float(v) for v in out["target_bbox"]]
                raw_score = getattr(tracker, "last_max_score", float("nan"))
                score = finite_or_nan(raw_score)
                is_low_score = math.isfinite(score) and score < float(args.low_score_threshold)
                is_very_low_score = math.isfinite(score) and score < float(args.very_low_score_threshold)
                is_out_of_frame = bbox_out_of_frame(current_box, frame_bgr.shape, args.bbox_margin)

                if previous_for_metrics is not None:
                    prev_center = bbox_center(previous_for_metrics)
                    curr_center = bbox_center(current_box)
                    center_delta = math.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
                    center_delta_ratio = center_delta / max(frame_diag, 1e-9)
                    prev_area = max(bbox_area(previous_for_metrics), 1e-9)
                    curr_area = max(bbox_area(current_box), 1e-9)
                    area_ratio = max(curr_area / prev_area, prev_area / curr_area)

                is_large_jump = center_delta_ratio > float(args.jump_threshold)
                is_large_area_change = area_ratio > float(args.area_change_threshold)
                suspicious = is_out_of_frame or is_very_low_score or (is_low_score and (is_large_jump or is_large_area_change))
                weak_confidence = is_low_score or is_large_jump or is_large_area_change

                if suspicious:
                    consecutive_suspicious += 1
                else:
                    consecutive_suspicious = 0
                metrics["max_consecutive_suspicious"] = max(metrics["max_consecutive_suspicious"], consecutive_suspicious)

                if is_out_of_frame or consecutive_suspicious >= int(args.lost_frames):
                    mode = "LOST"
                    color = (0, 0, 255)
                    if not was_lost:
                        metrics["lost_events"] += 1
                    was_lost = True
                elif consecutive_suspicious >= int(args.suspect_frames):
                    mode = "SUSPECTED_LOST"
                    color = (0, 128, 255)
                    was_lost = False
                elif weak_confidence:
                    mode = "LOW_CONFIDENCE"
                    color = (0, 165, 255)
                    was_lost = False
                else:
                    mode = "TRACKING"
                    color = (0, 255, 0)
                    was_lost = False

                draw_box(display, current_box, color)
                draw_status_label(display, mode, color)
                metrics["track_frames"] += 1
                metrics["track_time_total_s"] += track_elapsed
                if math.isfinite(score):
                    metrics["scores"].append(score)
                metrics["center_deltas"].append(center_delta)
                metrics["area_ratios"].append(area_ratio)
                if is_low_score:
                    metrics["low_score_frames"] += 1
                if is_very_low_score:
                    metrics["very_low_score_frames"] += 1
                if mode == "LOW_CONFIDENCE":
                    metrics["low_confidence_frames"] += 1
                if mode == "SUSPECTED_LOST":
                    metrics["suspected_lost_frames"] += 1
                if is_out_of_frame:
                    metrics["out_of_frame_frames"] += 1
                if is_large_jump:
                    metrics["large_jump_frames"] += 1
                if is_large_area_change:
                    metrics["large_area_change_frames"] += 1
                if mode == "LOST":
                    metrics["lost_frames"] += 1
                previous_box = current_box[:]
            elif initializing:
                mode = "INITIALIZING"
                initializing_box = init_worker.init_box if init_worker is not None else init_box_preview
                draw_box(display, initializing_box, (0, 165, 255))
                draw_status_label(display, "INITIALIZING", (0, 165, 255))
            else:
                if (
                    startup_detector is not None
                    and not startup_detector_used
                    and startup_detector_calls < max(0, int(args.startup_detect_frames))
                    and init_worker is None
                ):
                    startup_detector_calls += 1
                    metrics["startup_detector_calls"] = startup_detector_calls
                    detections = startup_detector.detect(frame_bgr, args.startup_detector_conf, 0.45)
                    startup_selected_det, _, startup_center_ratio = select_startup_detection(
                        detections,
                        frame_bgr.shape,
                        args.startup_center_threshold,
                    )
                    for det in detections:
                        draw_box(display, det["box_xywh"], (255, 180, 0))
                    if startup_selected_det is not None:
                        startup_detector_hits += 1
                        metrics["startup_detector_hits"] = startup_detector_hits
                        startup_selected_conf = float(startup_selected_det.get("conf", float("nan")))
                        draw_box(display, startup_selected_det["box_xywh"], (255, 0, 0))
                        cv2.putText(
                            display,
                            "STARTUP DET %.2f center=%.3f hits=%d/%d" % (
                                startup_selected_conf,
                                startup_center_ratio,
                                startup_detector_hits,
                                max(1, int(args.startup_detect_min_hits)),
                            ),
                            (12, 108),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        if startup_detector_hits >= max(1, int(args.startup_detect_min_hits)):
                            init_box = startup_selected_det["box_xywh"][:]
                            consecutive_suspicious = 0
                            previous_box = None
                            startup_detector_used = True
                            metrics["startup_detector_initializations"] += 1
                            metrics["startup_detector_selected_conf"] = startup_selected_conf
                            if video_file_source:
                                init_trigger = "startup_detector_sync"
                                out, last_init_time = initialize_tracker_sync(tracker, frame_bgr, init_box)
                                prev_output = dict(out)
                                current_box = [float(v) for v in out.get("target_bbox", init_box)]
                                previous_box = current_box[:]
                                tracking_active = True
                                initializing = False
                                init_worker = None
                                was_lost = False
                                init_time_for_row = last_init_time
                                metrics["initializations_completed"] += 1
                                metrics["init_times"].append(last_init_time)
                                print(
                                    "startup_detector_initialized_bbox=%.3f,%.3f,%.3f,%.3f det_conf=%.3f init_time_s=%.3f" % tuple(current_box + [startup_selected_conf, last_init_time]),
                                    flush=True,
                                )
                            else:
                                initializing = True
                                tracking_active = False
                                init_worker = TrackerInitializer(tracker, frame_bgr, init_box)
                                init_worker.start()
                                was_lost = False
                                init_trigger = "startup_detector_async"
                                print(
                                    "startup_detector_initializing_bbox=%.3f,%.3f,%.3f,%.3f det_conf=%.3f" % tuple(init_box + [startup_selected_conf]),
                                    flush=True,
                                )
                    if startup_detector_calls >= max(0, int(args.startup_detect_frames)) and not startup_detector_used:
                        cv2.putText(display, "Startup detector timeout. Press I for manual init.", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
                if tracking_active and current_box is not None:
                    mode = "DETECTED_INITIALIZED"
                    draw_box(display, current_box, (0, 255, 0))
                    draw_status_label(display, "DETECTED_INITIALIZED", (0, 255, 0))
                elif initializing and init_worker is not None:
                    mode = "DETECTED_INITIALIZING"
                    draw_box(display, init_worker.init_box, (0, 165, 255))
                    draw_status_label(display, "DETECTED_INITIALIZING", (0, 165, 255))
                else:
                    startup_detector_waiting = (
                        startup_detector is not None
                        and not startup_detector_used
                        and startup_detector_calls < max(0, int(args.startup_detect_frames))
                    )
                    if startup_detector_waiting:
                        draw_status_label(display, "READY: startup detector selecting centered person", (0, 255, 255))
                    else:
                        draw_box(display, init_box_preview, (0, 255, 255))
                        draw_status_label(display, "READY: stand in box and press I", (0, 255, 255))

            now = time.perf_counter()
            fps_value = 1.0 / max(now - last_time, 1e-9)
            last_time = now
            draw_header(display, frame_index, fps_value, mode, score)
            cv2.imshow(args.window_name, display)

            if args.save_video and output_dir is not None and video_writer is None:
                height, width = display.shape[:2]
                out_fps = float(cap.get(cv2.CAP_PROP_FPS))
                if out_fps <= 0:
                    out_fps = float(args.fps)
                video_writer = cv2.VideoWriter(str(output_dir / "camera_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
            if video_writer is not None:
                video_writer.write(display)

            if writer_csv is not None:
                if current_box is None:
                    row_box = [float("nan")] * 4
                    center_x = float("nan")
                    center_y = float("nan")
                    area = float("nan")
                else:
                    row_box = current_box
                    center_x, center_y = bbox_center(current_box)
                    area = bbox_area(current_box)
                if startup_selected_det is None:
                    startup_det_values = [float("nan")] * 4
                else:
                    startup_det_values = startup_selected_det["box_xywh"]
                writer_csv.writerow([
                    frame_index, now - run_start, row_box[0], row_box[1], row_box[2], row_box[3], center_x, center_y, area,
                    score, mode, track_elapsed, init_time_for_row, fps_value, int(is_low_score), int(is_very_low_score),
                    int(is_out_of_frame), center_delta, center_delta_ratio, area_ratio, int(is_large_jump),
                    int(is_large_area_change), consecutive_suspicious, metrics["manual_reinitializations"],
                    init_trigger, startup_det_values[0], startup_det_values[1], startup_det_values[2], startup_det_values[3],
                    startup_selected_conf, metrics["startup_detector_calls"], metrics["startup_detector_hits"],
                ])
                csv_file.flush()

            metrics["frames_total"] += 1
            if output_dir is not None and args.metrics_interval > 0 and metrics["frames_total"] % int(args.metrics_interval) == 0:
                metrics["end_time"] = time.perf_counter()
                write_metrics_summary(output_dir, metrics)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("i"), ord("I")) and init_worker is None:
                init_box = select_initial_bbox(frame_bgr, args.init_width_ratio, args.init_height_ratio)
                tracking_active = False
                consecutive_suspicious = 0
                previous_box = None
                metrics["manual_reinitializations"] += 1
                startup_detector_used = True
                if video_file_source:
                    out, last_init_time = initialize_tracker_sync(tracker, frame_bgr, init_box)
                    prev_output = dict(out)
                    current_box = [float(v) for v in out.get("target_bbox", init_box)]
                    previous_box = current_box[:]
                    tracking_active = True
                    initializing = False
                    init_worker = None
                    was_lost = False
                    metrics["initializations_completed"] += 1
                    metrics["init_times"].append(last_init_time)
                    print("manual_initialized_bbox=%.3f,%.3f,%.3f,%.3f init_time_s=%.3f" % tuple(current_box + [last_init_time]), flush=True)
                else:
                    initializing = True
                    init_worker = TrackerInitializer(tracker, frame_bgr, init_box)
                    init_worker.start()
                    was_lost = False
                    print("initializing_bbox=%.3f,%.3f,%.3f,%.3f" % tuple(init_worker.init_box), flush=True)

            frame_index += 1
    finally:
        metrics["end_time"] = time.perf_counter()
        cap.release()
        if startup_detector is not None:
            startup_detector.close()
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
