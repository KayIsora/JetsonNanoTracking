#!/usr/bin/env python3
"""Unix socket service exposing PyTracking ECO to the robot runtime."""

import argparse
import math
import os
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


INIT_MSG = 0x01
UPDATE_MSG = 0x02
INIT_META = struct.Struct("<iiiiii")
UPDATE_META = struct.Struct("<ii")
RESPONSE = struct.Struct("<Biiiif")


def parse_args():
    parser = argparse.ArgumentParser(description="Expose ECO tracking over a Unix domain socket.")
    parser.add_argument("--socket-path", default="/tmp/eco_tracker.sock", help="Unix socket path.")
    parser.add_argument("--backend", choices=["eco", "echo"], default="eco", help="Tracking backend.")
    parser.add_argument("--tracker-name", default="eco", help="PyTracking tracker name.")
    parser.add_argument("--param", default="verified_otb936_run_update", help="PyTracking parameter name.")
    parser.add_argument(
        "--lost-score-threshold",
        type=float,
        default=0.0,
        help="Mark tracker lost when last_max_score is below this value. 0.0 disables score-based loss.",
    )
    parser.add_argument("--fast-service-mode", action="store_true", help="Use reduced ECO settings for service latency.")
    parser.add_argument("--service-max-sample", type=int, default=128, help="Fast service max sample side length.")
    parser.add_argument("--service-min-sample", type=int, default=96, help="Fast service min sample side length.")
    parser.add_argument("--service-init-cg-iter", type=int, default=2, help="Fast service initial CG iterations.")
    parser.add_argument("--service-cg-iter", type=int, default=1, help="Fast service per-update CG iterations.")
    parser.add_argument("--service-sample-memory", type=int, default=8, help="Fast service sample memory size.")
    parser.add_argument("--service-train-skipping", type=int, default=50, help="Fast service training skip interval.")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame tracking details.")
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
        return True
    return False


def print_param_override(name, value):
    print("[eco_service] override %s=%s" % (name, value), flush=True)


def create_tracker(project_root, tracker_name, param_name, service_options, log_stage=None):
    setup_pytracking(project_root)
    from pytracking.evaluation.tracker import Tracker

    if log_stage is not None:
        log_stage("before Tracker(%s, %s)" % (tracker_name, param_name))
    wrapper = Tracker(tracker_name, param_name)
    if log_stage is not None:
        log_stage("after Tracker(...)")
        log_stage("before wrapper.get_parameters()")
    params = wrapper.get_parameters()
    if log_stage is not None:
        log_stage("after wrapper.get_parameters()")

    params.debug = 0
    params.visualization = False
    params.preload_features_on_create = False
    params.warmup_on_create = False
    params.warmup_image_sz = 96
    print_param_override("debug", params.debug)
    print_param_override("visualization", params.visualization)
    print_param_override("preload_features_on_create", params.preload_features_on_create)
    print_param_override("warmup_on_create", params.warmup_on_create)
    print_param_override("warmup_image_sz", params.warmup_image_sz)

    if service_options["fast_service_mode"]:
        fast_overrides = [
            ("max_image_sample_size", service_options["service_max_sample"] ** 2),
            ("min_image_sample_size", service_options["service_min_sample"] ** 2),
            ("init_CG_iter", service_options["service_init_cg_iter"]),
            ("CG_iter", service_options["service_cg_iter"]),
            ("sample_memory_size", service_options["service_sample_memory"]),
            ("train_skipping", service_options["service_train_skipping"]),
        ]
        for name, value in fast_overrides:
            if set_param_if_present(params, name, value):
                print_param_override(name, value)

    if log_stage is not None:
        log_stage("after overriding params")
        log_stage("before wrapper.create_tracker(params)")
    tracker = wrapper.create_tracker(params)
    if log_stage is not None:
        log_stage("after wrapper.create_tracker(params)")
        log_stage("skipping explicit tracker.initialize_features(); lazy init in tracker.initialize")
    return tracker


def recv_exact(conn, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("client disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame_bgr(conn, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions: %dx%d" % (width, height))
    byte_count = width * height * 3
    data = recv_exact(conn, byte_count)
    frame = np.frombuffer(data, dtype=np.uint8)
    return frame.reshape((height, width, 3))


def coerce_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    try:
        return [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None


def is_valid_bbox(bbox, width, height):
    values = coerce_bbox(bbox)
    if values is None:
        return False

    x, y, w, h = values
    if not all(math.isfinite(v) for v in values):
        return False
    if w <= 0.0 or h <= 0.0:
        return False

    max_wild_w = float(width) * 3.0
    max_wild_h = float(height) * 3.0
    if w > max_wild_w or h > max_wild_h:
        return False

    if x + w < -float(width) or y + h < -float(height):
        return False
    if x > float(width) * 2.0 or y > float(height) * 2.0:
        return False

    return True


def bbox_to_ints(bbox):
    x, y, w, h = coerce_bbox(bbox)
    return [int(round(v)) for v in (x, y, w, h)]


def current_score(tracker):
    try:
        return float(getattr(tracker, "last_max_score", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def score_or_default(tracker, default=1.0):
    score = current_score(tracker)
    if not math.isfinite(score):
        return float(default)
    return score


def is_lost_by_score(score, threshold):
    return threshold > 0.0 and math.isfinite(score) and score < threshold


def send_response(conn, ok, bbox=None, score=float("nan")):
    if ok and bbox is not None:
        x, y, w, h = bbox_to_ints(bbox)
    else:
        x, y, w, h = 0, 0, 0, 0
    conn.sendall(RESPONSE.pack(1 if ok else 0, x, y, w, h, float(score)))


class EcoService:
    def __init__(
        self,
        project_root,
        backend,
        tracker_name,
        param_name,
        lost_score_threshold,
        service_options,
        verbose,
    ):
        self.project_root = project_root
        self.backend = backend
        self.tracker_name = tracker_name
        self.param_name = param_name
        self.lost_score_threshold = lost_score_threshold
        self.service_options = service_options
        self.verbose = verbose
        self.tracker = None
        self.prev_output = {}
        self.initialized = False
        self.last_bbox = None
        self.start_time = time.perf_counter()

    def log_stage(self, name):
        if self.verbose and self.backend == "eco":
            print("[eco_service] %.3fs %s" % (time.perf_counter() - self.start_time, name), flush=True)

    def handle_init(self, conn):
        meta = recv_exact(conn, INIT_META.size)
        self.log_stage("before parse INIT metadata")
        width, height, x, y, w, h = INIT_META.unpack(meta)
        self.log_stage(
            "after metadata parsed frame=%dx%d bbox=[%d, %d, %d, %d]" % (width, height, x, y, w, h)
        )
        self.log_stage("before read frame bytes")
        frame_bgr = read_frame_bgr(conn, width, height)
        self.log_stage("after frame bytes read")

        init_bbox = [x, y, w, h]
        if self.backend == "eco":
            self.log_stage("before BGR to RGB")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.log_stage("after BGR to RGB done")

        self.log_stage("before bbox validation")
        if not is_valid_bbox(init_bbox, width, height):
            if self.verbose:
                print("Invalid init bbox: %s frame=%dx%d" % (init_bbox, width, height), flush=True)
            self.tracker = None
            self.prev_output = {}
            self.initialized = False
            self.last_bbox = None
            send_response(conn, False)
            return True
        self.log_stage("after bbox validation done")

        if self.backend == "echo":
            self.initialized = True
            self.last_bbox = init_bbox
            self.tracker = None
            self.prev_output = {}
            print("ECHO INIT bbox=%s" % init_bbox, flush=True)
            send_response(conn, True, init_bbox, 1.0)
            return True

        self.tracker = create_tracker(
            self.project_root,
            self.tracker_name,
            self.param_name,
            self.service_options,
            self.log_stage,
        )
        self.log_stage("before tracker.initialize(frame_rgb, {'init_bbox': ...})")
        out = self.tracker.initialize(frame_rgb, {"init_bbox": init_bbox})
        self.log_stage("after tracker.initialize(...)")

        if out is None:
            self.prev_output = {}
            response_bbox = init_bbox
            score = 1.0
        elif isinstance(out, dict):
            self.prev_output = dict(out)
            output_bbox = out.get("target_bbox")
            response_bbox = output_bbox if is_valid_bbox(output_bbox, width, height) else init_bbox
            score = score_or_default(self.tracker, 1.0)
        else:
            raise TypeError("tracker.initialize returned unsupported type: %s" % type(out).__name__)

        self.initialized = True
        self.last_bbox = init_bbox
        print("INIT bbox=%s score=%s" % (response_bbox, score), flush=True)
        self.log_stage("before send response")
        send_response(conn, True, response_bbox, score)
        self.log_stage("after send response")
        return True

    def handle_update(self, conn):
        meta = recv_exact(conn, UPDATE_META.size)
        width, height = UPDATE_META.unpack(meta)
        self.log_stage("received UPDATE metadata frame=%dx%d" % (width, height))
        frame_bgr = read_frame_bgr(conn, width, height)
        self.log_stage("read UPDATE frame bytes complete")

        if self.backend == "echo":
            if not self.initialized or self.last_bbox is None:
                if self.verbose:
                    print("ECHO UPDATE before INIT; returning lost.", flush=True)
                send_response(conn, False)
                return True
            if self.verbose:
                print("ECHO UPDATE bbox=%s" % self.last_bbox, flush=True)
            send_response(conn, True, self.last_bbox, 1.0)
            return True

        if self.tracker is None:
            if self.verbose:
                print("UPDATE before INIT; returning lost.", flush=True)
            send_response(conn, False)
            return True

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.log_stage("converted UPDATE frame")
        self.log_stage("before tracker.track(...)")
        out = self.tracker.track(frame_rgb, {"previous_output": self.prev_output}) or {}
        self.log_stage("after tracker.track(...)")
        self.prev_output = dict(out)
        bbox = out.get("target_bbox")
        score = current_score(self.tracker)

        if not is_valid_bbox(bbox, width, height):
            if self.verbose:
                print("Invalid update bbox: %s score=%s frame=%dx%d" % (bbox, score, width, height), flush=True)
            self.log_stage("before sending invalid-bbox response")
            send_response(conn, False, score=score)
            self.log_stage("after sending invalid-bbox response")
            return True

        if is_lost_by_score(score, self.lost_score_threshold):
            if self.verbose:
                print("Lost by score: bbox=%s score=%s threshold=%s" % (bbox, score, self.lost_score_threshold), flush=True)
            self.log_stage("before sending lost-score response")
            send_response(conn, False, bbox, score)
            self.log_stage("after sending lost-score response")
            return True

        if self.verbose:
            print("UPDATE bbox=%s score=%s" % (bbox, score), flush=True)
        self.log_stage("before sending response")
        send_response(conn, True, bbox, score)
        self.log_stage("after sending response")
        return True

    def handle_client(self, conn):
        while True:
            try:
                msg_type = recv_exact(conn, 1)
            except ConnectionError:
                print("Client disconnected before request.", flush=True)
                return

            msg_type = msg_type[0]
            print("Request type received: 0x%02x" % msg_type, flush=True)
            try:
                if msg_type == INIT_MSG:
                    self.log_stage("request type INIT received")
                    keep_open = self.handle_init(conn)
                elif msg_type == UPDATE_MSG:
                    keep_open = self.handle_update(conn)
                else:
                    if self.verbose:
                        print("Unknown message type: 0x%02x" % msg_type, flush=True)
                    send_response(conn, False)
                    keep_open = True
            except ConnectionError as exc:
                print("Client disconnected during request: %s" % exc, flush=True)
                return
            except Exception as exc:
                print("Request error: %s" % exc, flush=True)
                print(traceback.format_exc(), flush=True)
                try:
                    send_response(conn, False)
                except OSError:
                    pass
                return

            if not keep_open:
                return


def remove_stale_socket(socket_path):
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        return


def serve(args):
    project_root = resolve_project_root(Path(__file__).resolve().parent)
    socket_path = args.socket_path
    parent = os.path.dirname(socket_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    service_options = {
        "fast_service_mode": args.fast_service_mode,
        "service_max_sample": args.service_max_sample,
        "service_min_sample": args.service_min_sample,
        "service_init_cg_iter": args.service_init_cg_iter,
        "service_cg_iter": args.service_cg_iter,
        "service_sample_memory": args.service_sample_memory,
        "service_train_skipping": args.service_train_skipping,
    }

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    remove_stale_socket(socket_path)
    try:
        server.bind(socket_path)
        server.listen(1)
        print("ECO service listening on %s" % socket_path, flush=True)

        while True:
            conn, _ = server.accept()
            print("Client connected.", flush=True)
            service = EcoService(
                project_root,
                args.backend,
                args.tracker_name,
                args.param,
                args.lost_score_threshold,
                service_options,
                args.verbose,
            )
            try:
                service.handle_client(conn)
            finally:
                conn.close()
                print("Client disconnected.", flush=True)
    finally:
        server.close()
        remove_stale_socket(socket_path)


def main():
    args = parse_args()
    try:
        serve(args)
    except KeyboardInterrupt:
        print("Stopping ECO service.", flush=True)


if __name__ == "__main__":
    main()
