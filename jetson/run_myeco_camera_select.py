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
    parser.add_argument("--low-score-threshold", type=float, default=0.72, help="Tracker score threshold for LOW_SCORE state.")
    parser.add_argument("--lost-frames", type=int, default=10, help="Consecutive low-score/out-of-frame frames before LOST state.")
    parser.add_argument("--bbox-margin", type=float, default=0.15, help="Allowed bbox margin outside frame as a ratio of bbox size.")
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


def draw_status_label(image, text, color):
    cv2.putText(image, text, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_header(image, frame_index, fps_value, state, score):
    score_text = "nan" if score is None else "%.3f" % score
    text = "frame=%d fps=%.2f state=%s score=%s" % (frame_index, fps_value, state, score_text)
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "Stand in yellow box, press I to initialize   Q/Esc: quit", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)


def select_initial_bbox(frame, width_ratio, height_ratio):
    return get_center_init_box(frame.shape, width_ratio, height_ratio)


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


def write_metrics_summary(output_dir, metrics):
    if output_dir is None:
        return
    duration_s = max(metrics["end_time"] - metrics["start_time"], 1e-9)
    track_frames = max(metrics["track_frames"], 1)
    init_times = metrics["init_times"]
    summary_path = output_dir / "camera_metrics.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("camera=%s\n" % metrics["camera"])
        f.write("tracker=%s\n" % metrics["tracker"])
        f.write("param=%s\n" % metrics["param"])
        f.write("frame_width=%d\n" % metrics["frame_width"])
        f.write("frame_height=%d\n" % metrics["frame_height"])
        f.write("low_score_threshold=%.6f\n" % metrics["low_score_threshold"])
        f.write("lost_frames_threshold=%d\n" % metrics["lost_frames_threshold"])
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
        f.write("low_score_frames=%d\n" % metrics["low_score_frames"])
        f.write("lost_frames=%d\n" % metrics["lost_frames"])
        f.write("out_of_frame_frames=%d\n" % metrics["out_of_frame_frames"])
        f.write("lost_events=%d\n" % metrics["lost_events"])
        f.write("max_consecutive_low_score=%d\n" % metrics["max_consecutive_low_score"])
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

    print("Opening camera source: %s" % args.camera, flush=True)
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
            "score", "state", "track_time_s", "init_time_s", "fps", "is_low_score", "is_out_of_frame",
            "consecutive_low_score", "manual_reinitializations",
        ])

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    tracking_active = False
    initializing = False
    init_worker = None
    prev_output = {}
    current_box = None
    frame_index = 0
    last_time = time.perf_counter()
    run_start = time.perf_counter()
    consecutive_low_score = 0
    was_lost = False
    last_init_time = 0.0
    frame_width = 0
    frame_height = 0
    metrics = {
        "camera": args.camera,
        "tracker": args.tracker_name,
        "param": args.param,
        "frame_width": 0,
        "frame_height": 0,
        "low_score_threshold": float(args.low_score_threshold),
        "lost_frames_threshold": int(args.lost_frames),
        "tracker_load_time_s": tracker_load_time,
        "start_time": run_start,
        "end_time": run_start,
        "frames_total": 0,
        "track_frames": 0,
        "track_time_total_s": 0.0,
        "manual_reinitializations": 0,
        "initializations_completed": 0,
        "init_times": [],
        "low_score_frames": 0,
        "lost_frames": 0,
        "out_of_frame_frames": 0,
        "lost_events": 0,
        "max_consecutive_low_score": 0,
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
            metrics["frame_width"] = frame_width
            metrics["frame_height"] = frame_height
            display = frame_bgr.copy()
            track_elapsed = 0.0
            mode = "READY"
            score = None
            is_low_score = False
            is_out_of_frame = False
            init_time_for_row = 0.0
            init_box_preview = get_center_init_box(frame_bgr.shape, args.init_width_ratio, args.init_height_ratio)

            if init_worker is not None and not init_worker.is_alive():
                if init_worker.error is not None:
                    raise init_worker.error
                out = init_worker.output or {}
                prev_output = dict(out)
                current_box = [float(v) for v in out.get("target_bbox", init_worker.init_box)]
                tracking_active = True
                initializing = False
                consecutive_low_score = 0
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
                current_box = [float(v) for v in out["target_bbox"]]
                raw_score = getattr(tracker, "last_max_score", float("nan"))
                score = finite_or_nan(raw_score)
                is_low_score = math.isfinite(score) and score < float(args.low_score_threshold)
                is_out_of_frame = bbox_out_of_frame(current_box, frame_bgr.shape, args.bbox_margin)
                if is_low_score or is_out_of_frame:
                    consecutive_low_score += 1
                else:
                    consecutive_low_score = 0
                metrics["max_consecutive_low_score"] = max(metrics["max_consecutive_low_score"], consecutive_low_score)

                if consecutive_low_score >= int(args.lost_frames):
                    mode = "LOST"
                    color = (0, 0, 255)
                    if not was_lost:
                        metrics["lost_events"] += 1
                    was_lost = True
                elif is_low_score or is_out_of_frame:
                    mode = "LOW_SCORE"
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
                if is_low_score:
                    metrics["low_score_frames"] += 1
                if is_out_of_frame:
                    metrics["out_of_frame_frames"] += 1
                if mode == "LOST":
                    metrics["lost_frames"] += 1
            elif initializing:
                mode = "INITIALIZING"
                draw_box(display, init_box_preview, (0, 165, 255))
                draw_status_label(display, "INITIALIZING", (0, 165, 255))
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
                writer_csv.writerow([
                    frame_index, now - run_start, row_box[0], row_box[1], row_box[2], row_box[3], center_x, center_y, area,
                    score, mode, track_elapsed, init_time_for_row, fps_value, int(is_low_score), int(is_out_of_frame),
                    consecutive_low_score, metrics["manual_reinitializations"],
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
                initializing = True
                tracking_active = False
                consecutive_low_score = 0
                init_worker = TrackerInitializer(tracker, frame_bgr, init_box)
                init_worker.start()
                metrics["manual_reinitializations"] += 1
                print("initializing_bbox=%.3f,%.3f,%.3f,%.3f" % tuple(init_worker.init_box), flush=True)

            frame_index += 1
    finally:
        metrics["end_time"] = time.perf_counter()
        cap.release()
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
