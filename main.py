"""
main.py — Pipeline Orchestrator & CLI Entry Point (Production Version).
Video Pipeline v3.2.0 / Audio Pipeline v4.2.1
"""
from __future__ import annotations

import argparse
import datetime
import os
import wave
from typing import Iterator, Optional

import numpy as np

from config_loader import (
    ConfigLoader,
    PIPELINE_VERSION as VIDEO_PIPELINE_VERSION,
    ResolvedConfig,
    RunMetadata,
    PipelineMetrics,
    StageCheckpoint,
    generate_run_id,
    logger,
    setup_logging,
    stage_timer,
)
from video_pipeline import (
    FPSNormalizer,
    MappingConfig,
    MappingEntry,
    OCRModule,
    RecolorEngineAdapter,
    ROIDetector,
    SubtitleRenderer,
    SubtitleTracker,
    TranslationAdapter,
    TranslationBufferManager,
    TranslationCache,
    VideoMetadata,
    VideoPipeline,
    VideoResult,
)

try:
    from dto.face_fusion import FaceFusionConfig
except ImportError:  # pragma: no cover
    from video_pipeline.dto.face_fusion import FaceFusionConfig

from audio_pipeline import AudioInputContract, AudioPipeline, AudioResult

from core.runtime.run_folder_manager import RunFolderManager
from core.runtime.checkpoint_runner import CheckpointRunner
from core.observability.failure_middleware import FailureCaptureMiddleware

CHECKPOINT_TYPE_REGISTRY = {
    "VideoResult": VideoResult,
    "AudioResult": AudioResult,
}


class AudioInputAdapter:
    @staticmethod
    def from_video_result(video_result: VideoResult) -> AudioInputContract:
        if not isinstance(video_result, VideoResult):
            raise TypeError("video_result must be VideoResult")

        return AudioInputContract(
            silent_video_path=video_result.output_path,
            translated_srt_path=video_result.subtitle_path or "",
            metadata=video_result.metadata.to_dict(),
        )


def _mock_translate(text: str) -> str:
    return f"translated_{text}"


def _generate_mock_frames(
    width: int = 320,
    height: int = 240,
    count: int = 10,
) -> Iterator[np.ndarray]:
    for i in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 25) % 256
        frame[:, :, 1] = (i * 50) % 256
        frame[:, :, 2] = (i * 75) % 256
        yield frame


def _ensure_mock_video_file(path: str) -> None:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        n_samples = 44100 * 3
        t = np.arange(n_samples, dtype=np.float64) / 44100
        signal = np.sin(2.0 * 3.14159265 * 440.0 * t) * 0.3
        samples_int16 = (signal * 32767).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(samples_int16.tobytes())


def _build_face_fusion_config(config: ResolvedConfig) -> FaceFusionConfig:
    return ConfigLoader.build_face_fusion_config(config)


def create_video_pipeline(
    fps: float = 30.0,
    face_fusion_config: Optional[FaceFusionConfig] = None,
) -> VideoPipeline:
    return VideoPipeline(
        roi_detector=ROIDetector(),
        ocr_module=OCRModule(),
        subtitle_tracker=SubtitleTracker(fps=fps),
        translation_buffer=TranslationBufferManager(threshold=20),
        translation_cache=TranslationCache(),
        translation_adapter=TranslationAdapter(translate_fn=_mock_translate),
        recolor_engine=RecolorEngineAdapter(),
        subtitle_renderer=SubtitleRenderer(),
        fps_normalizer=FPSNormalizer(input_fps=fps),
        face_fusion_config=face_fusion_config,
    )


def run_pipeline(
    config: ResolvedConfig,
    runner: CheckpointRunner,
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "video_result": None,
        "audio_result": None,
    }

    enable_video = config.values.get("enable_video", "true").lower() == "true"
    enable_audio = config.values.get("enable_audio", "true").lower() == "true"

    face_fusion_config = _build_face_fusion_config(config)
    video_result: Optional[VideoResult] = None

    if enable_video:
        with stage_timer("Pipeline.VideoPipeline"):
            fps = float(config.values.get("fps", "30.0"))
            frame_count = int(config.values.get("frame_count", "10"))
            width = int(config.values.get("width", "320"))
            height = int(config.values.get("height", "240"))

            video_pipeline = create_video_pipeline(
                fps=fps,
                face_fusion_config=face_fusion_config,
            )

            metadata = VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
            )

            mapping_config = MappingConfig(
                version="1.0.0",
                engine_mode=config.values.get("engine_mode", "HSV"),
                mappings=(
                    MappingEntry(
                        source_range=(0, 50),
                        target_hue_range=(100, 150),
                    ),
                ),
            )

            output_path = os.path.join(output_dir, "silent_video.mp4")

            frame_source = _generate_mock_frames(
                width=width,
                height=height,
                count=frame_count,
            )

            video_result = runner.run_stage(
                "VideoPipeline",
                video_pipeline.run,
                frame_source,
                metadata,
                mapping_config,
                output_path,
            )

            results["video_result"] = video_result
            logger.info("Video pipeline complete: %s", video_result.output_path)

    if enable_audio and video_result is not None:
        with stage_timer("Pipeline.AudioPipeline"):
            _ensure_mock_video_file(video_result.output_path)

            audio_input = AudioInputAdapter.from_video_result(video_result)
            audio_pipeline = AudioPipeline()

            audio_result = runner.run_stage(
                "AudioPipeline",
                audio_pipeline.run,
                audio_input,
            )

            results["audio_result"] = audio_result
            logger.info("Audio pipeline complete: %s", audio_result.audio_path)

    return results


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Video+Audio Pipeline")

    parser.add_argument("--input", default="", help="Input video path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--enable-video", default="true", help="Enable video pipeline")
    parser.add_argument("--enable-audio", default="true", help="Enable audio pipeline")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--fps", default="30.0", help="FPS")
    parser.add_argument("--frame-count", default="10", help="Frame count")

    parser.add_argument("--face-fusion-enabled", default="false", help="Enable Face Fusion")
    parser.add_argument("--face-fusion-mode", default="single", help="Face Fusion mode")
    parser.add_argument(
        "--face-fusion-reference-face-path",
        default="reference_face.png",
        help="Reference face image path",
    )
    parser.add_argument("--face-fusion-strength", default="1.0", help="Face Fusion strength")
    parser.add_argument(
        "--face-fusion-min-confidence",
        default="0.5",
        help="Face Fusion minimum confidence",
    )
    parser.add_argument(
        "--face-fusion-preserve-background",
        default="true",
        help="Preserve background in Face Fusion",
    )

    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)

    setup_logging(parsed.log_level)

    cli_overrides = {
        "input_path": parsed.input,
        "output_path": parsed.output,
        "enable_video": parsed.enable_video,
        "enable_audio": parsed.enable_audio,
        "fps": parsed.fps,
        "frame_count": parsed.frame_count,
        "face_fusion_enabled": parsed.face_fusion_enabled,
        "face_fusion_mode": parsed.face_fusion_mode,
        "face_fusion_reference_face_path": parsed.face_fusion_reference_face_path,
        "face_fusion_strength": parsed.face_fusion_strength,
        "face_fusion_min_confidence": parsed.face_fusion_min_confidence,
        "face_fusion_preserve_background": parsed.face_fusion_preserve_background,
    }

    config = ConfigLoader.resolve_from_paths(
        json_path=parsed.config,
        cli_overrides=cli_overrides,
    )

    run_id = generate_run_id(config.to_dict())

    metadata = RunMetadata(
        run_id=run_id,
        pipeline_version=VIDEO_PIPELINE_VERSION,
        config_hash="",
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
    )

    run_folder = RunFolderManager()
    run_folder.create(run_id)

    metrics = PipelineMetrics(run_id)
    checkpoint = StageCheckpoint(run_folder.checkpoint_path())

    runner = CheckpointRunner(
        checkpoint,
        metrics,
        type_registry=CHECKPOINT_TYPE_REGISTRY,
    )

    logger.info("Pipeline starting with config: %s", config.values)

    middleware = FailureCaptureMiddleware(
        run_metadata=metadata,
        config=config,
        metrics=metrics,
        run_folder=run_folder,
    )

    def pipeline_exec():
        with stage_timer("Pipeline.Total"):
            return run_pipeline(
                config=config,
                runner=runner,
                output_dir=run_folder.artifacts_dir,
            )

    return middleware.run(pipeline_exec)


if __name__ == "__main__":
    main()
