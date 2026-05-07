#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import math
import os
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2


CSV_FIELDS = [
    "run_index",
    "phase",
    "init_time_s",
    "initializing_frames",
    "track_avg_s",
    "track_p50_s",
    "track_p90_s",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "success",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ECO load/init/reinit latency on a live camera without detector or recognizer.")
    parser.add_argument("--camera", default="0", help="Camera index, video path, or GStreamer pipeline.")
    parser.add_argument("--gstreamer", action="store_true", help="Open --camera as a GStreamer pipeline.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width for index cameras.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height for index cameras.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS.")
    parser.add_argument("--tracker-name", default="eco", help="PyTracking tracker name.")
    parser.add_argument("--param", default="verified_otb936_run_update", help="PyTracking parameter name.")
    parser.add_argument("--bbox", required=True, help="Initial bbox as x,y,w,h in camera coordinates.")
    parser.add_argument("--init-repeats", type=int, default=10, help="Number of reinitialize calls after the first initialize.")
    parser.add_argument("--track-frames", type=int, default=30, help="Number of track frames measured after each initialize.")
    parser.add_argument("--output-dir", type=Path, default=Path("jetson/benchmarks/eco_init"), help="Directory for benchmark metrics and CSV output.")
    parser.add_argument("--no-initialize-features", action="store_true", help="Skip explicit tracker.initialize_features() to compare lazy init behavior.")
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


def parse_bbox(text):
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be x,y,w,h")
    try:
        bbox = [float(v.strip()) for v in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("--bbox must contain numeric x,y,w,h")
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise argparse.ArgumentTypeError("--bbox width and height must be positive")
    return bbox


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


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read camera frame")
    return frame


def clip_bbox(box_xywh, frame_shape):
    height, width = frame_shape[:2]
    x, y, w, h = [float(v) for v in box_xywh]
    x = max(0.0, min(x, width - 1.0))
    y = max(0.0, min(y, height - 1.0))
    w = max(1.0, min(w, width - x))
    h = max(1.0, min(h, height - y))
    return [x, y, w, h]


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


def avg(values):
    return sum(values) / float(len(values)) if values else float("nan")


def rss_mb():
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return float("nan")
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def time_stage(callback):
    start = time.perf_counter()
    result = callback()
    return result, time.perf_counter() - start


def create_tracker_with_stage_times(project_root, tracker_name, param_name, run_initialize_features):
    stage_times = {}

    _, stage_times["setup_pytracking_time_s"] = time_stage(lambda: setup_pytracking(project_root))
    def import_tracker_class():
        from pytracking.evaluation.tracker import Tracker
        return Tracker

    Tracker, stage_times["import_tracker_time_s"] = time_stage(import_tracker_class)

    wrapper, wrapper_time = time_stage(lambda: Tracker(tracker_name, param_name))
    stage_times["tracker_wrapper_time_s"] = wrapper_time
    params, stage_times["get_parameters_time_s"] = time_stage(lambda: wrapper.get_parameters())
    params.debug = 0
    params.visualization = False
    tracker, stage_times["create_tracker_time_s"] = time_stage(lambda: wrapper.create_tracker(params))

    if run_initialize_features and hasattr(tracker, "initialize_features"):
        _, stage_times["initialize_features_time_s"] = time_stage(lambda: tracker.initialize_features())
    else:
        stage_times["initialize_features_time_s"] = 0.0

    stage_times["tracker_load_time_s"] = (
        stage_times["setup_pytracking_time_s"]
        + stage_times["import_tracker_time_s"]
        + stage_times["tracker_wrapper_time_s"]
        + stage_times["get_parameters_time_s"]
        + stage_times["create_tracker_time_s"]
        + stage_times["initialize_features_time_s"]
    )
    return tracker, stage_times


class InitializeWorker(threading.Thread):
    def __init__(self, tracker, frame_bgr, bbox):
        threading.Thread.__init__(self)
        self.daemon = True
        self.tracker = tracker
        self.frame_bgr = frame_bgr.copy()
        self.bbox = list(map(float, bbox))
        self.output = None
        self.error = None
        self.elapsed = 0.0

    def run(self):
        start = time.perf_counter()
        try:
            frame_rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
            self.output = self.tracker.initialize(frame_rgb, {"init_bbox": self.bbox}) or {}
        except Exception as exc:
            self.error = exc
        finally:
            self.elapsed = time.perf_counter() - start


def initialize_and_count_wait_frames(cap, tracker, frame_bgr, bbox):
    worker = InitializeWorker(tracker, frame_bgr, bbox)
    worker.start()
    initializing_frames = 0
    while worker.is_alive():
        ok, _ = cap.read()
        if ok:
            initializing_frames += 1
        else:
            time.sleep(0.001)
    worker.join()
    if worker.error is not None:
        raise worker.error
    return worker.output or {}, worker.elapsed, initializing_frames


def measure_track_frames(cap, tracker, prev_output, count):
    times = []
    output = dict(prev_output or {})
    bbox = None
    for _ in range(max(0, int(count))):
        frame_bgr = read_frame(cap)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        start = time.perf_counter()
        output = tracker.track(frame_rgb, {"previous_output": output}) or {}
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if "target_bbox" in output:
            bbox = output["target_bbox"]
    return times, output, bbox


def write_metrics(path, args, stage_times, rows, all_init_times, all_track_times, first_init_time, rss_start, rss_after_create, rss_end):
    with path.open("w", encoding="utf-8") as f:
        f.write("camera=%s\n" % args.camera)
        f.write("tracker=%s\n" % args.tracker_name)
        f.write("param=%s\n" % args.param)
        f.write("frame_width=%d\n" % args.width)
        f.write("frame_height=%d\n" % args.height)
        f.write("bbox=%s\n" % args.bbox)
        f.write("init_repeats=%d\n" % args.init_repeats)
        f.write("track_frames=%d\n" % args.track_frames)
        for key in [
            "setup_pytracking_time_s",
            "import_tracker_time_s",
            "tracker_wrapper_time_s",
            "get_parameters_time_s",
            "create_tracker_time_s",
            "initialize_features_time_s",
            "tracker_load_time_s",
        ]:
            f.write("%s=%.6f\n" % (key, stage_times.get(key, float("nan"))))
        f.write("first_initialize_time_s=%.6f\n" % first_init_time)
        f.write("first_init_time_s=%.6f\n" % first_init_time)
        f.write("reinitialize_count=%d\n" % max(0, len(all_init_times) - 1))
        reinit_times = [
            float(row["init_time_s"])
            for row in rows
            if row["phase"] == "reinitialize" and math.isfinite(float(row["init_time_s"]))
        ]
        f.write("reinit_avg_s=%.6f\n" % avg(reinit_times))
        f.write("reinit_p50_s=%.6f\n" % percentile(reinit_times, 50))
        f.write("reinit_p90_s=%.6f\n" % percentile(reinit_times, 90))
        f.write("reinit_max_s=%.6f\n" % (max(reinit_times) if reinit_times else float("nan")))
        f.write("init_avg_s=%.6f\n" % avg(all_init_times))
        f.write("init_p50_s=%.6f\n" % percentile(all_init_times, 50))
        f.write("init_p90_s=%.6f\n" % percentile(all_init_times, 90))
        f.write("init_max_s=%.6f\n" % (max(all_init_times) if all_init_times else float("nan")))
        f.write("track_avg_s=%.6f\n" % avg(all_track_times))
        f.write("track_p50_s=%.6f\n" % percentile(all_track_times, 50))
        f.write("track_p90_s=%.6f\n" % percentile(all_track_times, 90))
        f.write("initializing_frames_total=%d\n" % sum(int(row["initializing_frames"]) for row in rows))
        f.write("initializing_frames_max=%d\n" % (max(int(row["initializing_frames"]) for row in rows) if rows else 0))
        f.write("rss_start_mb=%.3f\n" % rss_start)
        f.write("rss_after_create_mb=%.3f\n" % rss_after_create)
        f.write("rss_end_mb=%.3f\n" % rss_end)
        f.write("runs_csv=%s\n" % (path.parent / "eco_benchmark_runs.csv"))


def main():
    args = parse_args()
    bbox = parse_bbox(args.bbox)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("output_dir=%s" % output_dir, flush=True)
    project_root = resolve_project_root(Path(__file__).resolve().parent)
    print("project_root=%s" % project_root, flush=True)

    rss_start = rss_mb()
    print("Creating ECO tracker %s/%s..." % (args.tracker_name, args.param), flush=True)
    tracker, stage_times = create_tracker_with_stage_times(project_root, args.tracker_name, args.param, not args.no_initialize_features)
    rss_after_create = rss_mb()
    print("Tracker load stages: setup=%.3fs import_tracker=%.3fs get_parameters=%.3fs create_tracker=%.3fs initialize_features=%.3fs total=%.3fs" % (
        stage_times["setup_pytracking_time_s"],
        stage_times["import_tracker_time_s"],
        stage_times["get_parameters_time_s"],
        stage_times["create_tracker_time_s"],
        stage_times["initialize_features_time_s"],
        stage_times["tracker_load_time_s"],
    ), flush=True)

    cap = open_capture(args.camera, args.gstreamer, args.width, args.height, args.fps)
    rows = []
    all_init_times = []
    all_track_times = []
    first_init_time = float("nan")

    try:
        for run_index in range(0, max(0, int(args.init_repeats)) + 1):
            frame_bgr = read_frame(cap)
            clipped_bbox = clip_bbox(bbox, frame_bgr.shape)
            phase = "first_initialize" if run_index == 0 else "reinitialize"
            print("%s run_index=%d bbox=%.1f,%.1f,%.1f,%.1f" % tuple([phase, run_index] + clipped_bbox), flush=True)

            try:
                output, init_time, initializing_frames = initialize_and_count_wait_frames(cap, tracker, frame_bgr, clipped_bbox)
                track_times, output, tracked_bbox = measure_track_frames(cap, tracker, output, args.track_frames)
                success = 1
            except Exception:
                print(traceback.format_exc(), file=sys.stderr, flush=True)
                output = {}
                init_time = float("nan")
                initializing_frames = 0
                track_times = []
                tracked_bbox = None
                success = 0

            if run_index == 0:
                first_init_time = init_time
            if math.isfinite(init_time):
                all_init_times.append(init_time)
            all_track_times.extend(track_times)
            if tracked_bbox is not None and len(tracked_bbox) == 4:
                row_bbox = [float(v) for v in tracked_bbox]
            else:
                row_bbox = clipped_bbox

            row = {
                "run_index": run_index,
                "phase": phase,
                "init_time_s": init_time,
                "initializing_frames": initializing_frames,
                "track_avg_s": avg(track_times),
                "track_p50_s": percentile(track_times, 50),
                "track_p90_s": percentile(track_times, 90),
                "bbox_x": row_bbox[0],
                "bbox_y": row_bbox[1],
                "bbox_w": row_bbox[2],
                "bbox_h": row_bbox[3],
                "success": success,
            }
            rows.append(row)
            print("%s done init=%.3fs initializing_frames=%d track_avg=%.3fs success=%d" % (
                phase, init_time, initializing_frames, row["track_avg_s"], success
            ), flush=True)
    finally:
        cap.release()

    csv_path = output_dir / "eco_benchmark_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    metrics_path = output_dir / "eco_benchmark_metrics.txt"
    write_metrics(metrics_path, args, stage_times, rows, all_init_times, all_track_times, first_init_time, rss_start, rss_after_create, rss_mb())
    print("runs_csv=%s" % csv_path, flush=True)
    print("metrics_summary=%s" % metrics_path, flush=True)


if __name__ == "__main__":
    main()
