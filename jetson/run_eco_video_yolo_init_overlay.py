#!/usr/bin/env python3
"""Offline ECO video runner initialized from a YOLO txt bbox."""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECO on a video initialized from YOLO txt labels.")
    parser.add_argument("--video-path", type=Path, required=True, help="Input video path.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Directory containing frame_%06d.txt labels.")
    parser.add_argument("--output-path", type=Path, required=True, help="Annotated output MP4 path.")
    parser.add_argument("--init-frame", type=int, default=0, help="Frame index used for ECO initialization.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process, 0 for all.")
    parser.add_argument("--tracker-name", default="eco", help="PyTracking tracker name.")
    parser.add_argument("--param", default="verified_otb936_run_update", help="PyTracking parameter name.")
    parser.add_argument("--draw-labels", action="store_true", default=True, help="Draw YOLO label boxes when present.")
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--verbose", action="store_true", help="Print timing logs.")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none", help="Rotate frames before tracking.")

    parser.add_argument("--fast-mode", action="store_true", default=True, help="Use reduced ECO settings.")
    parser.add_argument("--max-sample", type=int, default=96, help="Fast mode max sample side length.")
    parser.add_argument("--min-sample", type=int, default=80, help="Fast mode min sample side length.")
    parser.add_argument("--init-cg-iter", type=int, default=1, help="Fast mode initial CG iterations.")
    parser.add_argument("--cg-iter", type=int, default=1, help="Fast mode per-update CG iterations.")
    parser.add_argument("--sample-memory", type=int, default=4, help="Fast mode sample memory size.")
    parser.add_argument("--train-skipping", type=int, default=80, help="Fast mode train skipping interval.")
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


def rotate_frame(frame, mode):
    if mode == "none":
        return frame
    if mode == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError("Unknown rotate mode: %s" % mode)


def set_param_if_present(params, name, value):
    if hasattr(params, name):
        setattr(params, name, value)
        print("[eco_video_yolo] override %s=%s" % (name, value), flush=True)


def create_tracker(project_root, args):
    setup_pytracking(project_root)
    from pytracking.evaluation.tracker import Tracker

    start = time.perf_counter()
    if args.verbose:
        print("[eco_video_yolo] create_tracker start", flush=True)

    wrapper = Tracker(args.tracker_name, args.param)
    params = wrapper.get_parameters()
    params.debug = 0
    params.visualization = False
    params.preload_features_on_create = False
    params.warmup_on_create = False
    params.warmup_image_sz = 96
    print("[eco_video_yolo] override debug=%s" % params.debug, flush=True)
    print("[eco_video_yolo] override visualization=%s" % params.visualization, flush=True)
    print("[eco_video_yolo] override preload_features_on_create=%s" % params.preload_features_on_create, flush=True)
    print("[eco_video_yolo] override warmup_on_create=%s" % params.warmup_on_create, flush=True)
    print("[eco_video_yolo] override warmup_image_sz=%s" % params.warmup_image_sz, flush=True)

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
        print("[eco_video_yolo] create_tracker done %.3fs" % elapsed, flush=True)
        print("[eco_video_yolo] skipping explicit tracker.initialize_features(); lazy init in tracker.initialize", flush=True)
    return tracker


def label_path(label_dir, frame_index):
    return label_dir / ("frame_%06d.txt" % frame_index)


def parse_yolo_label(path):
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Label file is empty: %s" % path)
    line = text.splitlines()[0].strip()
    parts = line.split()
    if len(parts) != 5:
        raise ValueError("Invalid YOLO label format in %s: expected 5 fields, got %d" % (path, len(parts)))
    try:
        class_id = int(float(parts[0]))
        xc, yc, w, h = [float(v) for v in parts[1:]]
    except ValueError as exc:
        raise ValueError("Invalid numeric YOLO label in %s: %s" % (path, exc))
    return class_id, xc, yc, w, h


def yolo_to_xywh(label, frame_width, frame_height):
    _, xc, yc, w_norm, h_norm = label
    w = w_norm * frame_width
    h = h_norm * frame_height
    x = (xc - w_norm / 2.0) * frame_width
    y = (yc - h_norm / 2.0) * frame_height
    return [x, y, w, h]


def clip_bbox_xywh(bbox, frame_width, frame_height):
    x, y, w, h = [float(v) for v in bbox]
    x1 = max(0.0, min(float(frame_width - 1), x))
    y1 = max(0.0, min(float(frame_height - 1), y))
    x2 = max(0.0, min(float(frame_width), x + w))
    y2 = max(0.0, min(float(frame_height), y + h))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def read_label_bbox(label_dir, frame_index, frame_width, frame_height, required=False):
    path = label_path(label_dir, frame_index)
    label = parse_yolo_label(path)
    if label is None:
        if required:
            raise FileNotFoundError("Missing init label txt: %s" % path)
        return None
    return clip_bbox_xywh(yolo_to_xywh(label, frame_width, frame_height), frame_width, frame_height)


def coerce_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    try:
        return [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None


def valid_bbox(bbox, frame_width, frame_height):
    values = coerce_bbox(bbox)
    if values is None:
        return False
    x, y, w, h = values
    if not all(math.isfinite(v) for v in values):
        return False
    if w <= 0.0 or h <= 0.0:
        return False
    return x + w > 0.0 and y + h > 0.0 and x < frame_width and y < frame_height


def score_from_tracker(tracker):
    try:
        return float(getattr(tracker, "last_max_score", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def draw_bbox(frame, bbox, color, label=None, thickness=2):
    values = coerce_bbox(bbox)
    if values is None:
        return
    x, y, w, h = [int(round(v)) for v in values]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(frame, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


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
    return "bbox=[%.1f, %.1f, %.1f, %.1f]" % tuple(values)


def write_csv_row(writer, frame_index, tracker_bbox, label_bbox, score, track_time_s, fps, event):
    tracker_values = coerce_bbox(tracker_bbox) or [float("nan")] * 4
    label_values = coerce_bbox(label_bbox) or [float("nan")] * 4
    writer.writerow([
        frame_index,
        tracker_values[0],
        tracker_values[1],
        tracker_values[2],
        tracker_values[3],
        label_values[0],
        label_values[1],
        label_values[2],
        label_values[3],
        score,
        track_time_s,
        fps,
        event,
    ])


def main():
    args = parse_args()
    project_root = resolve_project_root(Path(__file__).resolve().parent)
    video_path = args.video_path.expanduser().resolve()
    label_dir = args.label_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = None
    csv_writer = None
    cap = cv2.VideoCapture(str(video_path))
    writer = None

    try:
        if not cap.isOpened():
            raise RuntimeError("Failed to open video: %s" % video_path)

        raw_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        raw_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0.0:
            fps = 30.0
        if raw_width <= 0 or raw_height <= 0:
            raise RuntimeError("Invalid video dimensions from %s" % video_path)

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise RuntimeError("Failed to read first frame from %s" % video_path)
        first_frame = rotate_frame(first_frame, args.rotate)
        frame_height, frame_width = first_frame.shape[:2]
        print("[eco_video_yolo] rotate=%s" % args.rotate, flush=True)
        print(
            "[eco_video_yolo] raw_size=%dx%d rotated_size=%dx%d"
            % (raw_width, raw_height, frame_width, frame_height),
            flush=True,
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        init_bbox = read_label_bbox(label_dir, args.init_frame, frame_width, frame_height, required=True)
        if not valid_bbox(init_bbox, frame_width, frame_height):
            raise ValueError("Invalid clipped init bbox from frame %d: %s" % (args.init_frame, init_bbox))

        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
        if not writer.isOpened():
            raise RuntimeError("Failed to create output video: %s" % output_path)

        if args.save_csv is not None:
            csv_path = args.save_csv.expanduser().resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame_index",
                "tracker_x",
                "tracker_y",
                "tracker_w",
                "tracker_h",
                "label_x",
                "label_y",
                "label_w",
                "label_h",
                "score",
                "track_time_s",
                "fps",
                "event",
            ])

        tracker = create_tracker(project_root, args)
        prev_output = {}
        tracker_bbox = None
        score = float("nan")
        init_time_s = 0.0
        track_times = []
        frames_processed = 0

        frame_index = 0
        while True:
            if args.max_frames > 0 and frames_processed >= args.max_frames:
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_bgr = rotate_frame(frame_bgr, args.rotate)
            frame_height, frame_width = frame_bgr.shape[:2]

            event = ""
            label_bbox = read_label_bbox(label_dir, frame_index, frame_width, frame_height, required=False)
            track_time_s = 0.0
            inst_fps = 0.0

            if frame_index < args.init_frame:
                event = "before_init"
            elif frame_index == args.init_frame:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                start = time.perf_counter()
                out = tracker.initialize(frame_rgb, {"init_bbox": init_bbox})
                init_time_s = time.perf_counter() - start
                prev_output = {} if out is None else dict(out)
                if isinstance(out, dict) and valid_bbox(out.get("target_bbox"), frame_width, frame_height):
                    tracker_bbox = out["target_bbox"]
                else:
                    tracker_bbox = init_bbox
                score = score_from_tracker(tracker)
                if not math.isfinite(score):
                    score = 1.0
                track_time_s = init_time_s
                inst_fps = 1.0 / track_time_s if track_time_s > 0.0 else 0.0
                event = "init"
                if args.verbose:
                    print("[eco_video_yolo] init frame=%d time=%.3fs bbox=%s" % (frame_index, init_time_s, tracker_bbox), flush=True)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                start = time.perf_counter()
                out = tracker.track(frame_rgb, {"previous_output": prev_output}) or {}
                track_time_s = time.perf_counter() - start
                track_times.append(track_time_s)
                prev_output = dict(out)
                tracker_bbox = out.get("target_bbox")
                score = score_from_tracker(tracker)
                inst_fps = 1.0 / track_time_s if track_time_s > 0.0 else 0.0
                event = "track"
                if args.verbose:
                    print("[eco_video_yolo] track frame=%d time=%.3fs score=%s bbox=%s" % (frame_index, track_time_s, score, tracker_bbox), flush=True)

            annotated = frame_bgr.copy()
            if args.draw_labels and label_bbox is not None:
                draw_bbox(annotated, label_bbox, (255, 0, 0), "label", 2)
            if tracker_bbox is not None:
                draw_bbox(annotated, tracker_bbox, (0, 0, 255), "tracker", 2)

            score_text = "nan" if not math.isfinite(score) else "%.3f" % score
            draw_text(annotated, [
                "frame=%d event=%s" % (frame_index, event),
                "score=%s track_ms=%.1f fps=%.2f" % (score_text, track_time_s * 1000.0, inst_fps),
                bbox_text(tracker_bbox),
                "init_bbox=%s" % bbox_text(init_bbox),
            ])

            writer.write(annotated)
            if csv_writer is not None:
                write_csv_row(csv_writer, frame_index, tracker_bbox, label_bbox, score, track_time_s, inst_fps, event)

            frames_processed += 1
            frame_index += 1

        mean_track_time_s = sum(track_times) / len(track_times) if track_times else 0.0
        fps_excluding_init = 1.0 / mean_track_time_s if mean_track_time_s > 0.0 else 0.0
        print("frames_processed=%d" % frames_processed, flush=True)
        print("init_time_s=%.6f" % init_time_s, flush=True)
        print("mean_track_time_s=%.6f" % mean_track_time_s, flush=True)
        print("fps_excluding_init=%.6f" % fps_excluding_init, flush=True)
        print("output_path=%s" % output_path, flush=True)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
