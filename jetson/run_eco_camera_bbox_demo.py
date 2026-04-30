#!/usr/bin/env python3
"""Repo-2-only ECO camera/video bbox tracking demo."""

import argparse
import csv
import math
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2


STATE_WAIT_INIT = "WAIT_INIT"
STATE_INIT_PENDING = "INIT_PENDING"
STATE_TRACKING = "TRACKING"
STATE_LOST = "LOST"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECO tracking from a camera or video with a manual bbox.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index.")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera capture FPS.")
    parser.add_argument("--video-path", type=Path, default=None, help="Optional video file path instead of camera.")

    parser.add_argument("--init-xywh", nargs=4, type=float, metavar=("X", "Y", "W", "H"), help="Initial bbox.")
    parser.add_argument("--init-mode", choices=["center"], default="center", help="Initial bbox mode.")
    parser.add_argument("--center-w-ratio", type=float, default=0.35, help="Center bbox width ratio.")
    parser.add_argument("--center-h-ratio", type=float, default=0.75, help="Center bbox height ratio.")
    parser.add_argument("--center-y-ratio", type=float, default=0.50, help="Center bbox vertical center ratio.")

    parser.add_argument("--tracker-name", default="eco", help="PyTracking tracker name.")
    parser.add_argument("--param", default="verified_otb936_run_update", help="PyTracking parameter name.")
    parser.add_argument("--fast-mode", action="store_true", default=True, help="Use reduced ECO settings.")
    parser.add_argument("--max-sample", type=int, default=96, help="Fast mode max sample side length.")
    parser.add_argument("--min-sample", type=int, default=80, help="Fast mode min sample side length.")
    parser.add_argument("--init-cg-iter", type=int, default=1, help="Fast mode initial CG iterations.")
    parser.add_argument("--cg-iter", type=int, default=1, help="Fast mode per-update CG iterations.")
    parser.add_argument("--sample-memory", type=int, default=4, help="Fast mode sample memory size.")
    parser.add_argument("--train-skipping", type=int, default=80, help="Fast mode train skipping interval.")

    parser.add_argument("--lost-score-threshold", type=float, default=0.20, help="Score threshold for LOST.")
    parser.add_argument("--lost-patience", type=int, default=10, help="Consecutive bad frames before LOST.")
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.95, help="Max bbox/frame area ratio.")
    parser.add_argument("--min-bbox-area", type=int, default=100, help="Minimum bbox area.")

    parser.add_argument("--output-dir", type=Path, default=Path("jetson/output/eco_camera_bbox_demo"))
    parser.add_argument("--save-video", action="store_true", help="Save annotated MP4.")
    parser.add_argument("--save-csv", action="store_true", help="Save tracking CSV.")
    parser.add_argument("--prewarm", action="store_true", help="Run one blocking ECO init before showing the UI.")
    parser.add_argument("--prewarm-only", action="store_true", help="Run prewarm and exit.")
    parser.add_argument("--verbose", action="store_true", help="Print timing logs.")
    return parser.parse_args()


def resolve_project_root(script_dir):
    for candidate in [script_dir] + list(script_dir.parents):
        if (candidate / "pytracking").is_dir():
            return candidate
        if (candidate / "MyECOTracker" / "pytracking").is_dir():
            return candidate / "MyECOTracker"
    raise RuntimeError("Could not locate pytracking project root from %s" % script_dir)


def setup_pytracking(project_root):
    pytracking_dir = project_root / "pytracking"
    pytracking_str = str(pytracking_dir)
    if pytracking_str not in sys.path:
        sys.path.insert(0, pytracking_str)


def set_param_if_present(params, name, value):
    if hasattr(params, name):
        setattr(params, name, value)
        print("[eco_camera_demo] override %s=%s" % (name, value), flush=True)


def create_tracker(project_root, args):
    setup_pytracking(project_root)
    from pytracking.evaluation.tracker import Tracker

    start = time.perf_counter()
    if args.verbose:
        print("[eco_camera_demo] create_tracker start", flush=True)

    wrapper = Tracker(args.tracker_name, args.param)
    params = wrapper.get_parameters()

    params.debug = 0
    params.visualization = False
    params.preload_features_on_create = False
    params.warmup_on_create = False
    params.warmup_image_sz = 96
    print("[eco_camera_demo] override debug=%s" % params.debug, flush=True)
    print("[eco_camera_demo] override visualization=%s" % params.visualization, flush=True)
    print("[eco_camera_demo] override preload_features_on_create=%s" % params.preload_features_on_create, flush=True)
    print("[eco_camera_demo] override warmup_on_create=%s" % params.warmup_on_create, flush=True)
    print("[eco_camera_demo] override warmup_image_sz=%s" % params.warmup_image_sz, flush=True)

    if args.fast_mode:
        set_param_if_present(params, "max_image_sample_size", args.max_sample ** 2)
        set_param_if_present(params, "min_image_sample_size", args.min_sample ** 2)
        set_param_if_present(params, "init_CG_iter", args.init_cg_iter)
        set_param_if_present(params, "CG_iter", args.cg_iter)
        set_param_if_present(params, "sample_memory_size", args.sample_memory)
        set_param_if_present(params, "train_skipping", args.train_skipping)

    tracker = wrapper.create_tracker(params)
    if args.verbose:
        elapsed = time.perf_counter() - start
        print("[eco_camera_demo] create_tracker done %.3fs" % elapsed, flush=True)
        print("[eco_camera_demo] skipping explicit tracker.initialize_features(); lazy init in tracker.initialize", flush=True)
    return tracker


def initialize_tracker(project_root, args, frame_bgr, init_bbox, label):
    total_start = time.perf_counter()
    print("[eco_camera_demo] %s init worker starts" % label, flush=True)

    create_start = time.perf_counter()
    tracker = create_tracker(project_root, args)
    create_elapsed = time.perf_counter() - create_start
    print("[eco_camera_demo] %s create_tracker time %.3fs" % (label, create_elapsed), flush=True)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    init_start = time.perf_counter()
    out = tracker.initialize(frame_rgb, {"init_bbox": init_bbox})
    init_elapsed = time.perf_counter() - init_start
    print("[eco_camera_demo] %s tracker.initialize time %.3fs" % (label, init_elapsed), flush=True)

    prev_output = {} if out is None else dict(out)
    frame_height, frame_width = frame_bgr.shape[:2]
    if isinstance(out, dict) and valid_bbox(out.get("target_bbox"), frame_width, frame_height, args):
        bbox = out["target_bbox"]
    else:
        bbox = init_bbox

    score = score_from_tracker(tracker)
    if not math.isfinite(score):
        score = 1.0

    total_elapsed = time.perf_counter() - total_start
    print("[eco_camera_demo] %s total init time %.3fs" % (label, total_elapsed), flush=True)
    return {
        "ok": True,
        "tracker": tracker,
        "prev_output": prev_output,
        "bbox": bbox,
        "score": score,
        "elapsed": total_elapsed,
    }


def init_worker(project_root, args, frame_bgr, init_bbox, result_queue, generation):
    try:
        result = initialize_tracker(project_root, args, frame_bgr, init_bbox, "async")
        result["generation"] = generation
        result_queue.put(result)
    except Exception:
        error_text = traceback.format_exc()
        print("[eco_camera_demo] async init failed\n%s" % error_text, flush=True)
        result_queue.put({"ok": False, "generation": generation, "error": error_text})


def center_bbox(frame_width, frame_height, args):
    w = frame_width * args.center_w_ratio
    h = frame_height * args.center_h_ratio
    cx = frame_width / 2.0
    cy = frame_height * args.center_y_ratio
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def initial_bbox(frame_width, frame_height, args):
    if args.init_xywh is not None:
        return [float(v) for v in args.init_xywh]
    return center_bbox(frame_width, frame_height, args)


def coerce_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    try:
        return [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None


def valid_bbox(bbox, frame_width, frame_height, args):
    values = coerce_bbox(bbox)
    if values is None:
        return False

    x, y, w, h = values
    if not all(math.isfinite(v) for v in values):
        return False
    if w <= 0.0 or h <= 0.0:
        return False

    if x + w <= 0.0 or y + h <= 0.0 or x >= frame_width or y >= frame_height:
        return False
    if x + w < -frame_width or y + h < -frame_height:
        return False
    if x > frame_width * 2.0 or y > frame_height * 2.0:
        return False

    area = w * h
    if area < float(args.min_bbox_area):
        return False
    if area > float(frame_width * frame_height) * args.max_bbox_area_ratio:
        return False
    return True


def score_from_tracker(tracker):
    try:
        return float(getattr(tracker, "last_max_score", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def draw_bbox(frame, bbox, color, thickness=2):
    values = coerce_bbox(bbox)
    if values is None:
        return
    x, y, w, h = [int(round(v)) for v in values]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def draw_text(frame, lines):
    x, y = 12, 24
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def bbox_text(bbox):
    values = coerce_bbox(bbox)
    if values is None:
        return "bbox=None"
    x, y, w, h = values
    return "bbox=[%.1f, %.1f, %.1f, %.1f]" % (x, y, w, h)


def open_capture(args):
    if args.video_path is not None:
        cap = cv2.VideoCapture(str(args.video_path.expanduser()))
    else:
        cap = cv2.VideoCapture(args.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap


def write_csv_row(writer, frame_index, state, bbox, score, track_time_s, fps, event):
    values = coerce_bbox(bbox) or [float("nan")] * 4
    writer.writerow([
        frame_index,
        state,
        values[0],
        values[1],
        values[2],
        values[3],
        score,
        track_time_s,
        fps,
        event,
    ])


def main():
    args = parse_args()
    project_root = resolve_project_root(Path(__file__).resolve().parent)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if args.save_video or args.save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_capture(args)
    if not cap.isOpened():
        source = args.video_path if args.video_path is not None else "camera index %d" % args.camera_index
        print("Failed to open %s" % source, flush=True)
        return 1

    ok, frame = cap.read()
    if not ok or frame is None:
        print("Failed to read first frame.", flush=True)
        cap.release()
        return 1

    frame_height, frame_width = frame.shape[:2]
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = float(args.camera_fps)

    first_init_bbox = initial_bbox(frame_width, frame_height, args)
    if args.prewarm or args.prewarm_only:
        print("[eco_camera_demo] prewarm start", flush=True)
        try:
            prewarm_start = time.perf_counter()
            initialize_tracker(project_root, args, frame.copy(), list(first_init_bbox), "prewarm")
            print("[eco_camera_demo] prewarm elapsed %.3fs" % (time.perf_counter() - prewarm_start), flush=True)
        except Exception:
            print("[eco_camera_demo] prewarm failed\n%s" % traceback.format_exc(), flush=True)
            cap.release()
            return 1

    if args.prewarm_only:
        cap.release()
        return 0

    writer_video = None
    csv_file = None
    csv_writer = None
    window_name = "Repo2 ECO Camera BBox Demo"

    try:
        if args.save_video:
            video_path = output_dir / "tracked_camera_demo.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_video = cv2.VideoWriter(str(video_path), fourcc, source_fps, (frame_width, frame_height))
            if not writer_video.isOpened():
                raise RuntimeError("Failed to create output video: %s" % video_path)

        if args.save_csv:
            csv_path = output_dir / "tracked_camera_demo.csv"
            csv_file = csv_path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame_index", "state", "x", "y", "w", "h", "score", "track_time_s", "fps", "event"])

        state = STATE_WAIT_INIT
        tracker = None
        prev_output = {}
        current_bbox = None
        last_score = float("nan")
        last_track_time_s = 0.0
        track_fps = 0.0
        lost_count = 0
        frame_index = 0
        pending_frame = frame
        init_queue = queue.Queue()
        init_thread = None
        init_generation = 0
        init_started_at = None
        init_pending_bbox = None
        return_after_failed_init = STATE_WAIT_INIT

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            if pending_frame is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("End of input or failed frame read.", flush=True)
                    break
            else:
                frame = pending_frame
                pending_frame = None

            event = ""
            frame_height, frame_width = frame.shape[:2]
            proposed_bbox = initial_bbox(frame_width, frame_height, args)

            if state == STATE_INIT_PENDING:
                try:
                    init_result = init_queue.get_nowait()
                except queue.Empty:
                    init_result = None

                if init_result is not None:
                    if init_result.get("generation") == init_generation:
                        init_thread = None
                        init_started_at = None
                        if init_result.get("ok"):
                            tracker = init_result["tracker"]
                            prev_output = init_result["prev_output"]
                            current_bbox = init_result["bbox"]
                            last_score = init_result["score"]
                            last_track_time_s = init_result["elapsed"]
                            track_fps = 1.0 / last_track_time_s if last_track_time_s > 0.0 else 0.0
                            lost_count = 0
                            state = STATE_TRACKING
                            event = "init"
                        else:
                            print("[eco_camera_demo] init worker failed\n%s" % init_result.get("error", ""), flush=True)
                            state = return_after_failed_init
                            event = "init_failed"
                    else:
                        print("[eco_camera_demo] ignored stale init result", flush=True)
                        if init_thread is not None and not init_thread.is_alive():
                            init_thread = None

            if state == STATE_TRACKING:
                start = time.perf_counter()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if args.verbose:
                    print("[eco_camera_demo] TRACK frame=%d start" % frame_index, flush=True)
                out = tracker.track(frame_rgb, {"previous_output": prev_output}) or {}
                last_track_time_s = time.perf_counter() - start
                if args.verbose:
                    print("[eco_camera_demo] TRACK frame=%d done %.3fs" % (frame_index, last_track_time_s), flush=True)
                prev_output = dict(out)
                candidate_bbox = out.get("target_bbox")
                last_score = score_from_tracker(tracker)
                track_fps = 1.0 / last_track_time_s if last_track_time_s > 0.0 else 0.0

                score_lost = math.isfinite(last_score) and last_score < args.lost_score_threshold
                bbox_lost = not valid_bbox(candidate_bbox, frame_width, frame_height, args)
                if bbox_lost or score_lost:
                    lost_count += 1
                    event = "bad_bbox" if bbox_lost else "low_score"
                else:
                    lost_count = 0
                    current_bbox = candidate_bbox

                if lost_count >= args.lost_patience:
                    state = STATE_LOST
                    event = "lost"

            annotated = frame.copy()
            if state == STATE_WAIT_INIT:
                draw_bbox(annotated, proposed_bbox, (0, 255, 255), 2)
                cv2.putText(annotated, "Stand inside yellow box, press i to init", (12, frame_height - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            elif state == STATE_INIT_PENDING:
                draw_bbox(annotated, init_pending_bbox, (0, 165, 255), 2)
                init_elapsed = time.perf_counter() - init_started_at if init_started_at is not None else 0.0
                cv2.putText(annotated, "INITIALIZING... UI is alive, wait %.1fs" % init_elapsed,
                            (12, frame_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
            elif state == STATE_TRACKING:
                draw_bbox(annotated, current_bbox, (0, 255, 0), 2)
            elif state == STATE_LOST:
                draw_bbox(annotated, current_bbox, (0, 0, 255), 2)
                cv2.putText(annotated, "LOST - press i to reinitialize", (12, frame_height - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            if state == STATE_WAIT_INIT:
                overlay_bbox = proposed_bbox
            elif state == STATE_INIT_PENDING:
                overlay_bbox = init_pending_bbox
            else:
                overlay_bbox = current_bbox
            score_text = "nan" if not math.isfinite(last_score) else "%.3f" % last_score
            draw_text(annotated, [
                "state=%s frame=%d" % (state, frame_index),
                "track_fps=%.2f track_ms=%.1f score=%s" % (track_fps, last_track_time_s * 1000.0, score_text),
                bbox_text(overlay_bbox),
                "keys: i=init r=reset q=quit",
            ])

            if csv_writer is not None:
                write_csv_row(csv_writer, frame_index, state, overlay_bbox, last_score, last_track_time_s, track_fps, event)

            if writer_video is not None:
                writer_video.write(annotated)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                init_generation += 1
                state = STATE_WAIT_INIT
                tracker = None
                prev_output = {}
                lost_count = 0
                init_started_at = None
                init_pending_bbox = None
                event = "reset"
            elif key == ord("i"):
                init_running = init_thread is not None and init_thread.is_alive()
                if state in (STATE_WAIT_INIT, STATE_LOST) and not init_running:
                    init_generation += 1
                    init_bbox = list(proposed_bbox)
                    init_frame = frame.copy()
                    init_pending_bbox = init_bbox
                    init_started_at = time.perf_counter()
                    return_after_failed_init = state
                    tracker = None
                    prev_output = {}
                    lost_count = 0
                    print("[eco_camera_demo] async init worker starts", flush=True)
                    init_thread = threading.Thread(
                        target=init_worker,
                        args=(project_root, args, init_frame, init_bbox, init_queue, init_generation),
                    )
                    init_thread.daemon = True
                    init_thread.start()
                    state = STATE_INIT_PENDING
                    event = "init_requested"
                elif state == STATE_INIT_PENDING:
                    print("[eco_camera_demo] init already pending", flush=True)

            frame_index += 1

    finally:
        cap.release()
        if writer_video is not None:
            writer_video.release()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
