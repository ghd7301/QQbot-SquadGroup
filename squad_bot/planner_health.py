from __future__ import annotations

import threading
import time


class SemanticPlannerHealth:
    def __init__(self, lanes: tuple[str, ...] = ("unsolicited", "addressed")) -> None:
        self._lock = threading.Lock()
        self._lanes = lanes
        self._consecutive_failures: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}
        self._probe_in_flight: dict[str, bool] = {}
        self._attempts: dict[str, int] = {}
        self._unavailable: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        with self._lock:
            for lane in self._lanes:
                self._consecutive_failures[lane] = 0
                self._circuit_open_until[lane] = 0.0
                self._probe_in_flight[lane] = False
                self._attempts[lane] = 0
                self._unavailable[lane] = 0

    def circuit_is_open(self, lane: str, *, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        with self._lock:
            return self._circuit_open_until.get(lane, 0.0) > current_time

    def reserve_request(
        self,
        lane: str,
        *,
        failure_threshold: int,
        now: float | None = None,
    ) -> bool:
        current_time = time.monotonic() if now is None else now
        with self._lock:
            if self._circuit_open_until.get(lane, 0.0) > current_time:
                return False
            threshold = max(1, int(failure_threshold))
            if self._consecutive_failures.get(lane, 0) < threshold:
                return True
            if self._probe_in_flight.get(lane, False):
                return False
            self._probe_in_flight[lane] = True
            return True

    def snapshot(self, *, now: float | None = None) -> dict[str, dict]:
        current_time = time.monotonic() if now is None else now
        with self._lock:
            return {
                lane: self._lane_snapshot(lane, current_time)
                for lane in self._lanes
            }

    def _lane_snapshot(self, lane: str, current_time: float) -> dict:
        attempts = self._attempts.get(lane, 0)
        unavailable = self._unavailable.get(lane, 0)
        return {
            "attempts": attempts,
            "unavailable": unavailable,
            "availability_rate": (
                round(1.0 - (unavailable / attempts), 4) if attempts else None
            ),
            "consecutive_failures": self._consecutive_failures.get(lane, 0),
            "circuit_open": self._circuit_open_until.get(lane, 0.0) > current_time,
            "retry_after_seconds": max(
                0,
                int(self._circuit_open_until.get(lane, 0.0) - current_time),
            ),
            "half_open_probe_in_flight": self._probe_in_flight.get(lane, False),
        }

    def record(
        self,
        available: bool,
        lane: str,
        *,
        failure_threshold: int,
        circuit_seconds: int,
        now: float | None = None,
    ) -> None:
        current_time = time.monotonic() if now is None else now
        with self._lock:
            self._attempts[lane] = self._attempts.get(lane, 0) + 1
            if available:
                self._consecutive_failures[lane] = 0
                self._circuit_open_until[lane] = 0.0
                self._probe_in_flight[lane] = False
                return
            self._unavailable[lane] = self._unavailable.get(lane, 0) + 1
            self._probe_in_flight[lane] = False
            self._consecutive_failures[lane] = (
                self._consecutive_failures.get(lane, 0) + 1
            )
            if self._consecutive_failures[lane] >= max(1, int(failure_threshold)):
                self._circuit_open_until[lane] = current_time + max(
                    1,
                    int(circuit_seconds),
                )
