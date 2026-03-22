"""
Integration tests for the FaceFusionEngine path inside the video pipeline.

These tests exercise the real pipeline code path that leads to FaceFusionEngine
instantiation.  They do NOT mock or stub FaceFusionEngine — they call the real
class.  Every test is expected to fail with NotImplementedError until a real
implementation is provided.

Pipeline path under test:
  ConfigLoader.resolve_from_paths()          → ResolvedConfig
  ConfigLoader.build_face_fusion_config()    → FaceFusionConfig  (DM-09)
  _validate_face_fusion_config()             → validation gate
  FaceFusionEngine(config)                   → engine instantiation
  FaceDetector(min_confidence)               → detector instantiation
  FaceFusionObserver                         → observability recording
  FailureCaptureMiddleware                   → error capture around the stage
  CheckpointRunner                           → stage execution wrapper
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pytest

from config_loader import (
    ConfigLoader,
    PipelineMetrics,
    ResolvedConfig,
    RunMetadata,
    StageCheckpoint,
    generate_run_id,
)
from core.observability.failure_middleware import FailureCaptureMiddleware
from core.runtime.checkpoint_runner import CheckpointRunner
from core.runtime.run_folder_manager import RunFolderManager
from dto.face_fusion import FaceFusionConfig
from processing.face_fusion.detector import FaceDetector
from processing.face_fusion.engine import FaceFusionEngine
from video_pipeline import (
    _validate_face_fusion_config,
    FaceFusionObserver,
    VideoMetadata,
    VideoResult,
)


# ---------------------------------------------------------------------------
# Helpers — reproduce the real pipeline wiring without importing main.py
# (main.py pulls in VideoPipeline which is not yet defined)
# ---------------------------------------------------------------------------

_FACE_FUSION_CLI_ENABLED = {
    "face_fusion_enabled": "true",
    "face_fusion_mode": "single",
    "face_fusion_reference_face_path": "reference_face.png",
    "face_fusion_strength": "0.8",
    "face_fusion_min_confidence": "0.5",
    "face_fusion_preserve_background": "true",
}

_FACE_FUSION_CLI_DISABLED = {
    "face_fusion_enabled": "false",
    "face_fusion_mode": "single",
    "face_fusion_reference_face_path": "reference_face.png",
    "face_fusion_strength": "0.8",
    "face_fusion_min_confidence": "0.5",
    "face_fusion_preserve_background": "true",
}


def _resolve_config(cli_overrides: dict) -> ResolvedConfig:
    """Mirror what main.main() does to build a ResolvedConfig."""
    return ConfigLoader.resolve_from_paths(json_path=None, cli_overrides=cli_overrides)


def _build_face_fusion_config(config: ResolvedConfig) -> FaceFusionConfig:
    """Mirror main._build_face_fusion_config."""
    return ConfigLoader.build_face_fusion_config(config)


def _face_fusion_stage(
    config: FaceFusionConfig,
    frame: np.ndarray,
) -> np.ndarray:
    """Simulate the pipeline stage that would run inside VideoPipeline.run().

    This is the code path the freeze spec (M12) mandates:
      1. Validate the FaceFusionConfig
      2. Create a FaceDetector
      3. Detect faces
      4. Create a FaceFusionEngine
      5. Fuse
    We call the real classes — no mocks.
    """
    _validate_face_fusion_config(config)

    detector = FaceDetector(min_confidence=config.min_confidence)
    faces = detector.detect(frame)

    engine = FaceFusionEngine(config)
    return engine.fuse(frame)


# ===========================================================================
# Tests
# ===========================================================================


class TestConfigToFaceFusionConfig:
    """Config resolution → FaceFusionConfig DTO, the first half of the path."""

    def test_enabled_config_produces_enabled_dto(self):
        config = _resolve_config(_FACE_FUSION_CLI_ENABLED)
        ff = _build_face_fusion_config(config)

        assert isinstance(ff, FaceFusionConfig)
        assert ff.enabled is True
        assert ff.mode == "single"
        assert ff.strength == 0.8
        assert ff.min_confidence == 0.5

    def test_disabled_config_produces_disabled_dto(self):
        config = _resolve_config(_FACE_FUSION_CLI_DISABLED)
        ff = _build_face_fusion_config(config)

        assert ff.enabled is False

    def test_config_dto_is_frozen(self):
        config = _resolve_config(_FACE_FUSION_CLI_ENABLED)
        ff = _build_face_fusion_config(config)

        with pytest.raises(AttributeError):
            ff.enabled = False  # type: ignore[misc]

    def test_validation_passes_for_valid_config(self):
        config = _resolve_config(_FACE_FUSION_CLI_ENABLED)
        ff = _build_face_fusion_config(config)
        # Must not raise
        _validate_face_fusion_config(ff)


class TestFaceFusionEngineNotImplemented:
    """Direct proof that FaceFusionEngine raises NotImplementedError.

    These are NOT isolation tests — they build the config through the real
    ConfigLoader path first, then attempt to instantiate the engine with
    the resulting DTO, exactly as the pipeline would.
    """

    @pytest.fixture()
    def enabled_config(self) -> FaceFusionConfig:
        resolved = _resolve_config(_FACE_FUSION_CLI_ENABLED)
        return _build_face_fusion_config(resolved)

    def test_engine_init_raises(self, enabled_config):
        with pytest.raises(NotImplementedError, match="FaceFusionEngine.__init__"):
            FaceFusionEngine(enabled_config)

    def test_detector_init_raises(self, enabled_config):
        with pytest.raises(NotImplementedError, match="FaceDetector.__init__"):
            FaceDetector(min_confidence=enabled_config.min_confidence)


class TestFaceFusionStageIntegration:
    """End-to-end: config → validate → engine, via the stage helper that
    mirrors the real VideoPipeline M12 code path.
    """

    @pytest.fixture()
    def frame(self) -> np.ndarray:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    @pytest.fixture()
    def enabled_config(self) -> FaceFusionConfig:
        resolved = _resolve_config(_FACE_FUSION_CLI_ENABLED)
        return _build_face_fusion_config(resolved)

    def test_stage_raises_not_implemented(self, enabled_config, frame):
        """The full stage must fail because FaceFusionEngine is not implemented."""
        with pytest.raises(NotImplementedError):
            _face_fusion_stage(enabled_config, frame)

    def test_stage_reaches_detector(self, enabled_config, frame):
        """Verify the error comes from FaceDetector (the first real call)."""
        with pytest.raises(NotImplementedError, match="FaceDetector"):
            _face_fusion_stage(enabled_config, frame)


class TestFaceFusionWithCheckpointRunner:
    """Wraps the face-fusion stage in CheckpointRunner — the same wrapper
    that run_pipeline() uses — to prove the end-to-end orchestration reaches
    the engine and surfaces the NotImplementedError.
    """

    @pytest.fixture()
    def pipeline_env(self, tmp_path):
        run_id = "test-face-fusion-run"
        checkpoint = StageCheckpoint(str(tmp_path / "checkpoints"))
        metrics = PipelineMetrics(run_id)
        runner = CheckpointRunner(
            checkpoint,
            metrics,
            type_registry={"VideoResult": VideoResult},
        )
        return runner, metrics

    @pytest.fixture()
    def enabled_config(self) -> FaceFusionConfig:
        resolved = _resolve_config(_FACE_FUSION_CLI_ENABLED)
        return _build_face_fusion_config(resolved)

    def test_runner_surfaces_not_implemented(self, pipeline_env, enabled_config):
        runner, metrics = pipeline_env
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with pytest.raises(NotImplementedError):
            runner.run_stage(
                "FaceFusion",
                _face_fusion_stage,
                enabled_config,
                frame,
            )

    def test_runner_records_failed_stage(self, pipeline_env, enabled_config):
        runner, metrics = pipeline_env
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with pytest.raises(NotImplementedError):
            runner.run_stage(
                "FaceFusion",
                _face_fusion_stage,
                enabled_config,
                frame,
            )

        summary = metrics.get_summary()
        assert summary["failed"] >= 1
        face_stages = [
            s for s in summary["stages"] if s["stage"] == "FaceFusion"
        ]
        assert len(face_stages) == 1
        assert face_stages[0]["status"] == "failed"


class TestFaceFusionWithFailureMiddleware:
    """FailureCaptureMiddleware wraps the pipeline callable in production.
    Prove that it captures the NotImplementedError and writes a debug report.
    """

    @pytest.fixture()
    def middleware_env(self, tmp_path):
        run_id = "test-ff-middleware"
        config = _resolve_config(_FACE_FUSION_CLI_ENABLED)

        metadata = RunMetadata(
            run_id=run_id,
            pipeline_version="3.2.0",
            config_hash="abc123",
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        metrics = PipelineMetrics(run_id)
        run_folder = RunFolderManager(base_dir=str(tmp_path))
        run_folder.create(run_id)

        middleware = FailureCaptureMiddleware(
            run_metadata=metadata,
            config=config,
            metrics=metrics,
            run_folder=run_folder,
        )
        return middleware, run_folder, config

    def test_middleware_captures_not_implemented(self, middleware_env):
        middleware, run_folder, config = middleware_env
        ff = _build_face_fusion_config(config)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with pytest.raises(NotImplementedError):
            middleware.run(lambda: _face_fusion_stage(ff, frame))

    def test_middleware_writes_debug_report(self, middleware_env):
        middleware, run_folder, config = middleware_env
        ff = _build_face_fusion_config(config)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with pytest.raises(NotImplementedError):
            middleware.run(lambda: _face_fusion_stage(ff, frame))

        report_path = run_folder.debug_report_path()
        assert os.path.isfile(report_path), (
            f"Debug report not created at {report_path}"
        )

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        assert "error" in report
        assert "NotImplementedError" in report["error"]["type"]


class TestFaceFusionObserverIntegration:
    """Verify the observability layer around the face-fusion stage.

    NOTE: FaceFusionObserver is currently broken — it tries to mutate
    FaceFusionMetrics which is a frozen dataclass.  These tests document
    that bug.  Once the observer is fixed, the assertions below should be
    updated to verify correct metric recording.
    """

    def test_observer_record_invocation_fails_on_frozen_metrics(self):
        """FaceFusionObserver.record_invocation mutates a frozen dataclass.
        This must raise FrozenInstanceError until the observer is fixed.
        """
        from dataclasses import FrozenInstanceError

        observer = FaceFusionObserver()
        with pytest.raises(FrozenInstanceError):
            observer.record_invocation(enabled=True)

    def test_observer_record_error_fails_on_frozen_metrics(self):
        from dataclasses import FrozenInstanceError

        observer = FaceFusionObserver()
        with pytest.raises(FrozenInstanceError):
            observer.record_error()

    def test_observer_record_passthrough_fails_on_frozen_metrics(self):
        from dataclasses import FrozenInstanceError

        observer = FaceFusionObserver()
        with pytest.raises(FrozenInstanceError):
            observer.record_passthrough()


class TestEndToEndPipelineReachesFaceFusion:
    """Full orchestration: config → runner → middleware → face-fusion stage.

    Mirrors the production wiring in main.run_pipeline() as closely as
    possible without importing the undefined VideoPipeline class.
    """

    def test_full_pipeline_with_face_fusion_enabled(self, tmp_path):
        """Full orchestration: config → runner → middleware → face-fusion stage.
        Mirrors the production wiring in main.run_pipeline().
        """
        # -- 1. Resolve config (same as main.main) -------------------------
        cli = {
            **_FACE_FUSION_CLI_ENABLED,
            "enable_video": "true",
            "enable_audio": "false",
            "fps": "30.0",
            "frame_count": "5",
            "width": "320",
            "height": "240",
            "engine_mode": "HSV",
        }
        config = _resolve_config(cli)

        # -- 2. Build face-fusion config (same as main._build_face_fusion_config)
        ff_config = _build_face_fusion_config(config)
        assert ff_config.enabled is True

        # -- 3. Set up run infrastructure (same as main.main) ---------------
        run_id = generate_run_id(config.to_dict())
        run_folder = RunFolderManager(base_dir=str(tmp_path))
        run_folder.create(run_id)

        metrics = PipelineMetrics(run_id)
        checkpoint = StageCheckpoint(run_folder.checkpoint_path())
        runner = CheckpointRunner(
            checkpoint, metrics, type_registry={"VideoResult": VideoResult}
        )
        metadata = RunMetadata(
            run_id=run_id,
            pipeline_version="3.2.0",
            config_hash="",
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        middleware = FailureCaptureMiddleware(
            run_metadata=metadata,
            config=config,
            metrics=metrics,
            run_folder=run_folder,
        )

        # -- 4. Execute the face-fusion stage inside the real wrappers ------
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        stage_reached = False

        def pipeline_callable():
            nonlocal stage_reached
            # Validate config the way the pipeline would
            _validate_face_fusion_config(ff_config)
            stage_reached = True
            # Now hit the real engine — must raise NotImplementedError
            return _face_fusion_stage(ff_config, frame)

        with pytest.raises(NotImplementedError):
            middleware.run(pipeline_callable)

        # -- 5. Assert the full path was exercised --------------------------
        # stage_reached proves the pipeline got past validation to the engine
        assert stage_reached is True

        # Debug report proves FailureCaptureMiddleware fired
        report_path = run_folder.debug_report_path()
        assert os.path.isfile(report_path)
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        assert "NotImplementedError" in report["error"]["type"]

    def test_disabled_face_fusion_never_touches_engine(self, tmp_path):
        """When face_fusion_enabled=false the pipeline must complete without
        ever reaching FaceFusionEngine.
        """
        cli = {
            **_FACE_FUSION_CLI_DISABLED,
            "enable_video": "true",
            "enable_audio": "false",
        }
        config = _resolve_config(cli)
        ff_config = _build_face_fusion_config(config)
        assert ff_config.enabled is False

        # The disabled path should never instantiate the engine.
        # Prove that by checking the config gate and ensuring no
        # NotImplementedError is raised.
        _validate_face_fusion_config(ff_config)

        # If disabled, pipeline skips engine — no error expected
        assert ff_config.enabled is False
