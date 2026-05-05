from __future__ import print_function

import math
import struct
import subprocess
from pathlib import Path

import numpy as np


COMMAND_ENROLL = 1
COMMAND_VERIFY = 2
COMMAND_CLEAR = 3
RESULT_CHUNK_SIZE = 9


class CppFaceRecognizer(object):
    def __init__(
        self,
        recognizer_bin,
        model_dir,
        face_conf=0.80,
        face_nms=0.40,
        face_size=300,
        facenet_size=112,
        use_gpu=True,
        stderr_path=None,
    ):
        self.recognizer_bin = Path(recognizer_bin).expanduser().resolve()
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.recognizer_bin.exists():
            raise RuntimeError("Missing face recognizer executable: %s" % self.recognizer_bin)
        required = ["retinaface.param", "retinaface.bin", "mbv2facenet.param", "mbv2facenet.bin"]
        missing = [name for name in required if not (self.model_dir / name).exists()]
        if missing:
            raise RuntimeError("Missing face recognition model files in %s: %s" % (self.model_dir, ", ".join(missing)))

        self.stderr_file = None
        stderr_target = None
        if stderr_path is not None:
            self.stderr_file = Path(stderr_path).open("w", encoding="utf-8")
            stderr_target = self.stderr_file

        command = [
            str(self.recognizer_bin),
            "--model-dir",
            str(self.model_dir),
            "--face-conf",
            str(float(face_conf)),
            "--face-nms",
            str(float(face_nms)),
            "--face-size",
            str(int(face_size)),
            "--facenet-size",
            str(int(facenet_size)),
        ]
        if not use_gpu:
            command.append("--cpu")

        self.process = subprocess.Popen(
            command,
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

    def clear(self):
        self._request(COMMAND_CLEAR, np.zeros((1, 1, 3), dtype=np.uint8), [])

    def enroll(self, frame_bgr, boxes_xywh):
        return self._request(COMMAND_ENROLL, frame_bgr, boxes_xywh)

    def verify(self, frame_bgr, boxes_xywh):
        return self._request(COMMAND_VERIFY, frame_bgr, boxes_xywh)

    def _request(self, command, frame_bgr, boxes_xywh):
        if self.process.poll() is not None:
            raise RuntimeError("Face recognizer exited with code %s" % self.process.returncode)

        if command == COMMAND_CLEAR:
            header = struct.pack("<iiii", COMMAND_CLEAR, 0, 0, 0)
            self.process.stdin.write(header)
            self.process.stdin.flush()
            return self._read_response()

        if frame_bgr.dtype != np.uint8 or len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be uint8 HxWx3 BGR")

        frame = np.ascontiguousarray(frame_bgr)
        height, width = frame.shape[:2]
        boxes = []
        for box in boxes_xywh:
            x, y, w, h = [float(v) for v in box]
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                continue
            if w <= 1.0 or h <= 1.0:
                continue
            boxes.append([x, y, w, h])

        roi_array = np.asarray(boxes, dtype="<f4")
        if roi_array.size == 0:
            roi_bytes = b""
            roi_count = 0
        else:
            roi_array = np.ascontiguousarray(roi_array.reshape((-1, 4)))
            roi_bytes = roi_array.tobytes()
            roi_count = roi_array.shape[0]

        header = struct.pack("<iiii", int(command), int(width), int(height), int(roi_count))
        self.process.stdin.write(header)
        if roi_bytes:
            self.process.stdin.write(roi_bytes)
        self.process.stdin.write(frame.tobytes())
        self.process.stdin.flush()
        return self._read_response()

    def _read_response(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Face recognizer returned no response")
        text = line.decode("utf-8", errors="replace").strip()
        parts = text.split()
        if not parts:
            raise RuntimeError("Empty face recognizer response")
        if parts[0] != "OK":
            raise RuntimeError("Face recognizer error: %s" % text)
        if len(parts) < 4:
            raise RuntimeError("Malformed face recognizer response: %s" % text)

        target_ready = bool(int(parts[1]))
        enrolled_index = int(parts[2])
        count = int(parts[3])
        expected = 4 + count * RESULT_CHUNK_SIZE
        if len(parts) != expected:
            raise RuntimeError("Malformed face recognizer result count: %s" % text)

        results = []
        offset = 4
        for _ in range(count):
            chunk = parts[offset:offset + RESULT_CHUNK_SIZE]
            offset += RESULT_CHUNK_SIZE
            index = int(chunk[0])
            face_found = bool(int(chunk[1]))
            face_box = [float(chunk[2]), float(chunk[3]), float(chunk[4]), float(chunk[5])]
            face_score = float(chunk[6])
            face_quality = float(chunk[7])
            similarity = float(chunk[8])
            results.append({
                "index": index,
                "face_found": face_found,
                "face_box_xywh": face_box,
                "face_score": face_score,
                "face_quality": face_quality,
                "similarity": similarity,
            })

        return {
            "target_ready": target_ready,
            "enrolled_index": enrolled_index,
            "results": results,
        }


class DisabledFaceRecognizer(object):
    def close(self):
        pass

    def clear(self):
        return {"target_ready": False, "enrolled_index": -1, "results": []}

    def enroll(self, frame_bgr, boxes_xywh):
        del frame_bgr, boxes_xywh
        return {"target_ready": False, "enrolled_index": -1, "results": []}

    def verify(self, frame_bgr, boxes_xywh):
        del frame_bgr, boxes_xywh
        return {"target_ready": False, "enrolled_index": -1, "results": []}
