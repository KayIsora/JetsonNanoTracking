#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


COUNT_FIELDS = {
    "frame_index",
    "source_frame_index",
    "source_is_video",
    "det_count",
    "is_low_score",
    "is_very_low_score",
    "is_out_of_frame",
    "is_large_jump",
    "is_large_area_change",
    "consecutive_suspicious",
    "ambiguous_detection",
    "soft_reinitializations",
    "hard_reinitializations",
    "background_lock_events",
    "identity_face_found",
    "identity_candidate_index",
    "identity_target_ready",
    "identity_reinit_blocks",
    "eco_ready_warmed",
    "eco_standby_warmed",
    "wheel_control_enabled",
    "wheel_control_allowed",
}

FLOAT_FIELDS = {
    "timestamp_s",
    "source_timestamp_s",
    "source_fps",
    "tracker_x",
    "tracker_y",
    "tracker_w",
    "tracker_h",
    "final_x",
    "final_y",
    "final_w",
    "final_h",
    "control_x",
    "control_y",
    "control_w",
    "control_h",
    "det_x",
    "det_y",
    "det_w",
    "det_h",
    "tracker_score",
    "det_conf",
    "best_detector_iou",
    "best_center_distance_ratio",
    "best_area_ratio",
    "track_time_s",
    "detect_time_s",
    "init_time_s",
    "fps",
    "center_delta_px",
    "center_delta_ratio",
    "area_ratio",
    "identity_score",
    "recognition_time_s",
    "wheel_center_error_norm",
    "wheel_distance_error_norm",
    "wheel_linear_cmd",
    "wheel_angular_cmd",
    "wheel_left_cmd",
    "wheel_right_cmd",
}

STAT_FIELDS = [
    "wheel_center_error_norm",
    "wheel_distance_error_norm",
    "wheel_linear_cmd",
    "wheel_angular_cmd",
    "wheel_left_cmd",
    "wheel_right_cmd",
    "control_h_ratio",
    "tracker_score",
]

DELTA_FIELDS = [
    "wheel_linear_cmd",
    "wheel_angular_cmd",
    "wheel_left_cmd",
    "wheel_right_cmd",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze wheel log-only replay output by source-video phases.")
    parser.add_argument("output_dir", type=Path, help="Run output directory containing camera_predictions.csv and camera_metrics.txt.")
    parser.add_argument("--phase-plan", required=True, help="Phase plan as start,end,label;start,end,label;...")
    return parser.parse_args()


def parse_phase_plan(value):
    phases = []
    for chunk in str(value).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError("Each phase must be start,end,label")
        start_s = float(parts[0])
        end_s = float(parts[1])
        label = parts[2]
        if end_s <= start_s:
            raise ValueError("Phase end must be greater than start for %s" % label)
        phases.append({"start_s": start_s, "end_s": end_s, "label": label})
    if not phases:
        raise ValueError("--phase-plan must contain at least one phase")
    return phases


def read_metrics(path):
    metrics = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            metrics[key] = value
    return metrics


def parse_int(value):
    if value is None:
        return 0
    text = str(value).strip()
    if text == "":
        return 0
    return int(float(text))


def parse_float(value):
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    lowered = text.lower()
    if lowered in ("nan", "none"):
        return float("nan")
    return float(text)


def parse_value(field_name, value):
    if field_name in COUNT_FIELDS:
        return parse_int(value)
    if field_name in FLOAT_FIELDS:
        return parse_float(value)
    return value


def read_rows(path, metrics):
    rows = []
    default_frame_height = parse_float(metrics.get("frame_height", "nan"))
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {}
            for key, value in raw_row.items():
                row[key] = parse_value(key, value)
            frame_height = row.get("frame_height", float("nan"))
            if not math.isfinite(frame_height):
                frame_height = default_frame_height
            control_h = row.get("control_h", float("nan"))
            if math.isfinite(control_h) and math.isfinite(frame_height) and frame_height > 0.0:
                row["control_h_ratio"] = float(control_h) / float(frame_height)
            else:
                row["control_h_ratio"] = float("nan")
            rows.append(row)
    return rows


def percentile(values, percent):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(percent) / 100.0
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def finite_values(rows, field_name):
    values = []
    for row in rows:
        value = row.get(field_name, float("nan"))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def format_float(value):
    if not math.isfinite(value):
        return "nan"
    return "%.6f" % float(value)


def format_counter(counter):
    if not counter:
        return "none"
    parts = []
    for key in sorted(counter):
        parts.append("%s=%d" % (key, counter[key]))
    return ", ".join(parts)


def rows_for_phase(rows, phase):
    selected = []
    start_s = phase["start_s"]
    end_s = phase["end_s"]
    for row in rows:
        timestamp_s = row.get("source_timestamp_s", float("nan"))
        if not math.isfinite(timestamp_s):
            continue
        if timestamp_s >= start_s and timestamp_s < end_s:
            selected.append(row)
    return selected


def summarize_numeric(rows, field_name):
    values = finite_values(rows, field_name)
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "avg": sum(values) / float(len(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": max(values),
    }


def summarize_deltas(rows, field_name):
    deltas = []
    previous = None
    for row in rows:
        value = row.get(field_name, float("nan"))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            value = float(value)
            if previous is not None:
                deltas.append(value - previous)
            previous = value
    if not deltas:
        return None
    return {
        "count": len(deltas),
        "min": min(deltas),
        "avg": sum(deltas) / float(len(deltas)),
        "p50": percentile(deltas, 50),
        "p90": percentile(deltas, 90),
        "max": max(deltas),
    }


def print_stat_block(prefix, summary):
    if summary is None:
        print("%s count=0" % prefix)
        return
    print(
        "%s count=%d min=%s avg=%s p50=%s p90=%s max=%s" % (
            prefix,
            summary["count"],
            format_float(summary["min"]),
            format_float(summary["avg"]),
            format_float(summary["p50"]),
            format_float(summary["p90"]),
            format_float(summary["max"]),
        )
    )


def median_for_phase(rows, label, field_name):
    values = []
    for row in rows:
        if row.get("phase_label") != label:
            continue
        value = row.get(field_name, float("nan"))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return percentile(values, 50)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    csv_path = output_dir / "camera_predictions.csv"
    metrics_path = output_dir / "camera_metrics.txt"
    phases = parse_phase_plan(args.phase_plan)
    metrics = read_metrics(metrics_path)
    rows = read_rows(csv_path, metrics)

    print("output_dir=%s" % output_dir)
    print("rows_total=%d" % len(rows))
    print("phase_plan=%s" % args.phase_plan)
    print("source_label=%s" % metrics.get("source_label", ""))
    print("source_is_video=%s" % metrics.get("source_is_video", ""))
    print("source_fps=%s" % metrics.get("source_fps", ""))
    print("input_rotate=%s" % metrics.get("input_rotate", ""))
    print("input_resize=%s" % metrics.get("input_resize", ""))
    print()

    phase_rows_all = []
    for phase in phases:
        phase_rows = rows_for_phase(rows, phase)
        for row in phase_rows:
            row["phase_label"] = phase["label"]
        phase_rows_all.extend(phase_rows)

        print("phase=%s start_s=%s end_s=%s" % (phase["label"], format_float(phase["start_s"]), format_float(phase["end_s"])))
        print("rows=%d" % len(phase_rows))
        if phase_rows:
            first_ts = phase_rows[0].get("source_timestamp_s", float("nan"))
            last_ts = phase_rows[-1].get("source_timestamp_s", float("nan"))
        else:
            first_ts = float("nan")
            last_ts = float("nan")
        print("source_time_range_s=%s..%s" % (format_float(first_ts), format_float(last_ts)))
        print("state_counts=%s" % format_counter(Counter(row.get("state", "") for row in phase_rows)))
        print("identity_state_counts=%s" % format_counter(Counter(row.get("identity_state", "") for row in phase_rows)))
        print("wheel_control_reason_counts=%s" % format_counter(Counter(row.get("wheel_control_reason", "") for row in phase_rows)))
        allowed_count = sum(1 for row in phase_rows if parse_int(row.get("wheel_control_allowed", 0)) == 1)
        blocked_count = sum(1 for row in phase_rows if parse_int(row.get("wheel_control_allowed", 0)) == 0)
        print("wheel_control_allowed=%d blocked=%d" % (allowed_count, blocked_count))
        for field_name in STAT_FIELDS:
            print_stat_block("%s_stats" % field_name, summarize_numeric(phase_rows, field_name))
        for field_name in DELTA_FIELDS:
            print_stat_block("%s_delta_stats" % field_name, summarize_deltas(phase_rows, field_name))
        print()

    print("suggested_wheel_target_height_ratio=%s" % format_float(median_for_phase(phase_rows_all, "normal", "control_h_ratio")))
    print("suggested_center_offset_norm=%s" % format_float(median_for_phase(phase_rows_all, "normal", "wheel_center_error_norm")))
    print()
    print("Checklist:")
    print("- Compare wheel_linear_cmd sign against intended forward/backward motion in each phase.")
    print("- Compare wheel_angular_cmd sign against intended left/right turning direction in each phase.")
    print("- Check left_cmd/right_cmd symmetry during normal phases before changing gains.")
    print("- Use source_timestamp_s-aligned phases to judge semantics; do not infer sign correctness automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
