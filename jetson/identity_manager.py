from __future__ import print_function

import math


class IdentityState(object):
    UNKNOWN = "UNKNOWN"
    WAITING_FACE = "WAITING_FACE"
    VERIFIED = "VERIFIED"
    STALE = "STALE"
    PROBATION = "PROBATION"
    REJECTED = "REJECTED"
    DISABLED = "DISABLED"


class IdentityManager(object):
    def __init__(
        self,
        enabled=True,
        match_threshold=0.35,
        reject_threshold=0.12,
        reject_confirm_frames=2,
        stale_frames=90,
        probation_frames=45,
    ):
        self.enabled = bool(enabled)
        self.match_threshold = float(match_threshold)
        self.reject_threshold = float(reject_threshold)
        self.reject_confirm_frames = int(reject_confirm_frames)
        self.stale_frames = int(stale_frames)
        self.probation_frames = int(probation_frames)
        self.reset()

    def reset(self, keep_target=True):
        target_ready = getattr(self, "target_ready", False) if keep_target else False
        self.target_ready = target_ready
        self.state = IdentityState.UNKNOWN if self.enabled else IdentityState.DISABLED
        if self.target_ready:
            self.state = IdentityState.STALE
        self.last_verified_frame = -10**9
        self.last_checked_frame = -10**9
        self.probation_started_frame = -10**9
        self.consecutive_rejects = 0
        self.last_similarity = float("nan")
        self.last_face_found = False
        self.last_candidate_index = -1
        self.last_face_box = None
        self.enrollments = 0
        self.reinit_blocks = 0

    def mark_target_cleared(self):
        self.target_ready = False
        self.state = IdentityState.UNKNOWN if self.enabled else IdentityState.DISABLED
        self.consecutive_rejects = 0
        self.last_similarity = float("nan")
        self.last_face_found = False
        self.last_candidate_index = -1
        self.last_face_box = None

    def age(self, frame_index):
        if self.last_verified_frame < 0:
            return 10**9
        return int(frame_index) - int(self.last_verified_frame)

    def hard_rejected(self):
        return self.state == IdentityState.REJECTED

    def is_verified_or_short_stale(self, frame_index):
        if self.state == IdentityState.VERIFIED:
            return True
        return self.state == IdentityState.STALE and self.age(frame_index) <= self.stale_frames

    def refresh_staleness(self, frame_index, danger=False):
        if not self.enabled:
            self.state = IdentityState.DISABLED
            return self.state
        if not self.target_ready:
            self.state = IdentityState.UNKNOWN
            return self.state
        if danger and self.state in (IdentityState.VERIFIED, IdentityState.STALE):
            self.state = IdentityState.PROBATION
            if self.probation_started_frame < 0:
                self.probation_started_frame = int(frame_index)
            return self.state
        if self.state == IdentityState.VERIFIED and self.age(frame_index) > self.stale_frames:
            self.state = IdentityState.STALE
        if self.state == IdentityState.PROBATION:
            if int(frame_index) - self.probation_started_frame > self.probation_frames:
                self.state = IdentityState.STALE
        return self.state

    def should_check(self, frame_index, interval, fast_interval, danger=False):
        if not self.enabled:
            return False
        if danger:
            return int(frame_index) - self.last_checked_frame >= max(1, int(fast_interval))
        if not self.target_ready:
            return True
        return int(frame_index) - self.last_checked_frame >= max(1, int(interval))

    def observe_enroll(self, frame_index, response):
        self.last_checked_frame = int(frame_index)
        results = response.get("results", [])
        enrolled_index = int(response.get("enrolled_index", -1))
        target_ready = bool(response.get("target_ready", False))
        selected = self._result_by_index(results, enrolled_index)
        if target_ready and selected is not None and selected.get("face_found", False):
            self.target_ready = True
            self.state = IdentityState.VERIFIED
            self.last_verified_frame = int(frame_index)
            self.last_similarity = self._finite_or_nan(selected.get("similarity"))
            self.last_face_found = True
            self.last_candidate_index = enrolled_index
            self.last_face_box = selected.get("face_box_xywh")
            self.consecutive_rejects = 0
            self.probation_started_frame = -10**9
            self.enrollments += 1
            return enrolled_index

        self.state = IdentityState.WAITING_FACE
        self.last_similarity = float("nan")
        self.last_face_found = any(item.get("face_found", False) for item in results)
        self.last_candidate_index = -1
        self.last_face_box = None
        return -1

    def observe_verify(self, frame_index, response, danger=False):
        self.last_checked_frame = int(frame_index)
        results = response.get("results", [])
        if not bool(response.get("target_ready", False)):
            self.target_ready = False
            self.state = IdentityState.UNKNOWN
            self.last_similarity = float("nan")
            self.last_face_found = False
            self.last_candidate_index = -1
            self.last_face_box = None
            return -1

        self.target_ready = True
        face_results = [item for item in results if item.get("face_found", False)]
        if not face_results:
            self.last_face_found = False
            self.last_similarity = float("nan")
            self.last_candidate_index = -1
            self.last_face_box = None
            self.consecutive_rejects = 0
            if danger:
                self.state = IdentityState.PROBATION
                self.probation_started_frame = int(frame_index)
            else:
                self.refresh_staleness(frame_index)
            return -1

        best = max(face_results, key=lambda item: self._similarity_rank(item.get("similarity")))
        similarity = self._finite_or_nan(best.get("similarity"))
        self.last_similarity = similarity
        self.last_face_found = True
        self.last_candidate_index = int(best.get("index", -1))
        self.last_face_box = best.get("face_box_xywh")

        if math.isfinite(similarity) and similarity >= self.match_threshold:
            self.state = IdentityState.VERIFIED
            self.last_verified_frame = int(frame_index)
            self.consecutive_rejects = 0
            self.probation_started_frame = -10**9
            return self.last_candidate_index

        if math.isfinite(similarity) and similarity <= self.reject_threshold:
            self.consecutive_rejects += 1
            if self.consecutive_rejects >= self.reject_confirm_frames:
                self.state = IdentityState.REJECTED
            else:
                self.state = IdentityState.PROBATION
                if self.probation_started_frame < 0:
                    self.probation_started_frame = int(frame_index)
            return -1

        self.consecutive_rejects = 0
        self.state = IdentityState.PROBATION
        if self.probation_started_frame < 0:
            self.probation_started_frame = int(frame_index)
        return -1

    def block_reinit(self):
        self.reinit_blocks += 1

    def _result_by_index(self, results, index):
        for item in results:
            if int(item.get("index", -1)) == int(index):
                return item
        return None

    def _finite_or_nan(self, value):
        if value is None:
            return float("nan")
        value = float(value)
        if math.isfinite(value):
            return value
        return float("nan")

    def _similarity_rank(self, value):
        value = self._finite_or_nan(value)
        if math.isfinite(value):
            return value
        return -2.0
