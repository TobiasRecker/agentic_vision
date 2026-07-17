from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clip_pose_pipeline.clip_object_capture_session import ClipObjectCaptureSession  # noqa: E402


def parameter(value):
    return SimpleNamespace(value=value)


def auto_state():
    values = {
        "auto_move_settle_sec": 0.5,
        "auto_rgbd_settle_sec": 0.25,
        "auto_stage_timeout_sec": 20.0,
    }
    state = SimpleNamespace(
        auto_sequence_active=True,
        auto_sequence_stage="capturing",
        auto_sequence_total=3,
        auto_sequence_completed=0,
        auto_capture_start_index=4,
        sample_index=5,
        auto_stage_ready_at=0.0,
        auto_stage_timeout_at=0.0,
        auto_wait_image_after=0.0,
        last_image_received_at=0.0,
        last_camera_info_received_at=0.0,
        last_cloud_received_at=0.0,
        get_parameter=lambda name: parameter(values[name]),
        write_metadata=lambda: None,
        report_status=lambda *_args: None,
    )
    state.stopped = []
    state.stop_auto_sequence = lambda reason, level="warn": state.stopped.append((reason, level))
    state.finished = False
    state.finish_auto_sequence = lambda: setattr(state, "finished", True)
    state.prepared = 0
    state.prepare_next_auto_target = lambda: setattr(state, "prepared", state.prepared + 1)
    return state


def test_successful_capture_waits_for_complete_fresh_rgbd_set() -> None:
    state = auto_state()

    ClipObjectCaptureSession.on_auto_capture_finished(state, True, "saved")

    assert state.auto_sequence_completed == 1
    assert state.auto_sequence_stage == "waiting for RGBD"
    assert not state.stopped
    state.auto_stage_ready_at = 0.0
    state.auto_stage_timeout_at = float("inf")
    state.last_image_received_at = state.auto_wait_image_after + 1.0
    state.last_camera_info_received_at = state.auto_wait_image_after + 1.0
    state.last_cloud_received_at = state.auto_wait_image_after

    ClipObjectCaptureSession.update_auto_sequence(state)
    assert state.prepared == 0

    state.last_cloud_received_at = state.auto_wait_image_after + 1.0
    ClipObjectCaptureSession.update_auto_sequence(state)
    assert state.prepared == 1


def test_capture_failure_stops_sequence_without_incrementing() -> None:
    state = auto_state()

    ClipObjectCaptureSession.on_auto_capture_finished(state, False, "camera failed")

    assert state.auto_sequence_completed == 0
    assert state.stopped == [("camera failed", "error")]


def test_successful_auto_move_enters_settle_stage() -> None:
    state = auto_state()

    ClipObjectCaptureSession.on_auto_move_finished(
        state,
        {"auto_sequence": True},
        success=True,
        canceled=False,
        message="done",
    )

    assert state.auto_sequence_stage == "settling after move"
    assert state.auto_stage_ready_at > 0.0
    assert not state.stopped
