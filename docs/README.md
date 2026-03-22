# Technical Architecture Documentation

Video Pipeline v3.2.0 / Audio Pipeline v4.2.1

---

## Module Map

The system is organized into 21 primary modules plus 5 observability modules.

### Video Pipeline Modules

| ID | Module | File | Responsibility |
|----|--------|------|----------------|
| M1 | ConfigLoader | `config_loader.py` | Merge defaults, JSON, CLI args into `ResolvedConfig` |
| M2 | DTO Layer | `video_pipeline.py` | Frozen dataclasses: `ROIBox`, `MappingEntry`, `MappingConfig`, `VideoMetadata`, `VideoResult`, `VideoOutputContract` |
| M3 | PipelineContext | `config_loader.py` | Runtime context with color mapping, scene, and translation caches |
| M4 | ROIDetector | `video_pipeline.py` | Region-of-interest detection on video frames |
| M5 | OCRModule | `video_pipeline.py` | Optical character recognition on detected ROIs |
| M6 | SubtitleTracker | `video_pipeline.py` | FPS-aware subtitle tracking across frames |
| M7 | TranslationBufferManager | `video_pipeline.py` | Buffered text aggregation with configurable threshold |
| M8 | TranslationAdapter | `video_pipeline.py` | Pluggable translation function adapter |
| M9 | TranslationCache | `video_pipeline.py` | In-memory translation result cache |
| M10 | MappingConfig Loader | `video_pipeline.py` | Semver-validated color mapping configuration |
| M11 | RecolorEngineAdapter | `video_pipeline.py` | HSV/LAB color remapping engine |
| M12 | FaceFusionEngine | `video_pipeline.py` | Face detection + fusion with configurable modes |
| M13 | SubtitleRenderer | `video_pipeline.py` | Rendered subtitle overlay onto frames |
| M14 | FPSNormalizer | `video_pipeline.py` | Frame rate normalization |
| M15 | ReaderThread | `video_pipeline.py` | Threaded frame reading with sentinel EOF |
| M16 | ProcessorThread | `video_pipeline.py` | Per-frame processing chain (M4→M13) |
| M17 | EncoderThread | `video_pipeline.py` | Threaded frame encoding |
| M18 | BackboneExecutionCore | `video_pipeline.py` | Thread coordination: Reader→Processor→Encoder |
| M19 | VideoPipeline | `video_pipeline.py` | Top-level video orchestrator |
| M20 | AudioInputAdapter | `main.py` | Converts `VideoResult` → `AudioInputContract` |
| M21 | AudioPipeline | `audio_pipeline.py` | Top-level audio orchestrator |

### Audio Pipeline Stages

| ID | Stage | Responsibility |
|----|-------|----------------|
| M20.1 | AudioExtractor | Extract audio track from silent video, produce `ExtractedAudio` |
| M20.2 | SourceSeparator | Separate voice from background, produce `SeparatedAudio` |
| M20.3 | VoiceAnalyzer | ASR + prosody analysis, produce `VoiceAnalysisResult` with `ASRSegment` tuples |
| M20.4 | SubtitleAligner | Align translated subtitles to ASR segments, produce `AlignedSegment` tuples |
| M20.5 | TTSGenerator | Generate TTS audio per segment with pitch shifting, produce `TTSOutputSegment` tuples |
| M20.6 | DurationAligner | Time-stretch TTS to match original duration, produce `DurationAlignedSegment` tuples |
| M20.7 | RMSProcessor | RMS loudness normalization per segment, produce `RMSProcessedSegment` tuples |
| M20.8 | AudioMixer | Mix processed voice with background audio, produce `MixedAudio` then `AudioResult` |

### Observability Modules

| ID | Module | File | Responsibility |
|----|--------|------|----------------|
| O1 | CheckpointRunner | `core/runtime/checkpoint_runner.py` | Execute stages with checkpoint save/restore and type registry |
| O2 | MetricsInstrumentation | `core/observability/metrics_instrumentation.py` | `@instrument_stage` decorator for timing |
| O3 | FailureCaptureMiddleware | `core/observability/failure_middleware.py` | Wrap pipeline execution, generate debug reports on failure |
| O4 | RunFolderManager | `core/runtime/run_folder_manager.py` | Create and manage per-run directory structure |
| O5 | DebugReportGenerator | `core/observability/debug_report.py` | Structured JSON debug reports with environment, artifacts, stack traces |

---

## Data Flow

```
                         ┌─────────────────────────────┐
                         │        CLI (main.py)         │
                         │  parse_args → ConfigLoader   │
                         └──────────┬──────────────────┘
                                    │ ResolvedConfig
                                    ▼
                         ┌──────────────────────┐
                         │  RunFolderManager     │──→ run_<hash>/
                         │  CheckpointRunner     │      ├── artifacts/
                         │  PipelineMetrics      │      ├── checkpoints/
                         │  FailureMiddleware    │      └── logs/
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │  VideoPipeline    │           │  AudioPipeline    │
        │  (M19)            │           │  (M21)            │
        └───────┬───────────┘           └───────┬───────────┘
                │                               │
                ▼                               ▼
    ┌───────────────────────┐       ┌───────────────────────┐
    │ Per-Frame Chain:      │       │ Sequential Stages:    │
    │ ROI → OCR → Track →  │       │ Extract → Separate →  │
    │ Translate → Recolor → │       │ Analyze → Align →     │
    │ FaceFuse → Render     │       │ TTS → Duration →      │
    │                       │       │ RMS → Mix             │
    │ Threaded Backbone:    │       │                       │
    │ Reader → Processor →  │       │ (Single-threaded)     │
    │ Encoder               │       │                       │
    └───────────┬───────────┘       └───────────┬───────────┘
                │                               │
                ▼                               ▼
          VideoResult                     AudioResult
```

### DTO Flow (Video)

```
Iterator[np.ndarray] + VideoMetadata + MappingConfig
    → VideoPipeline.run()
    → VideoResult(output_path, subtitle_path, metadata)
    → AudioInputAdapter.from_video_result()
    → AudioInputContract(silent_video_path, translated_srt_path, metadata)
```

### DTO Flow (Audio)

```
AudioInputContract
    → ExtractedAudio
    → SeparatedAudio
    → VoiceAnalysisResult (Tuple[ASRSegment, ...])
        └── ASRSegment contains SegmentProsody
            └── SegmentProsody contains Tuple[RMSFrame, ...] + Tuple[PitchFrame, ...]
    → Tuple[AlignedSegment, ...]
    → Tuple[TTSOutputSegment, ...]
    → Tuple[DurationAlignedSegment, ...]
    → Tuple[RMSProcessedSegment, ...]
    → MixedAudio
    → AudioResult
```

---

## Configuration Resolution

Configuration follows a strict three-layer merge with last-writer-wins:

```
Defaults (ConfigLoader._DEFAULT_CONFIG)
    ← JSON file (--config path)
        ← CLI arguments
```

All values are stored as `Dict[str, str]` and wrapped in a frozen `ResolvedConfig`. Face fusion parameters undergo additional validation (mode allowlist, float range checks, boolean parsing).

---

## Design Decisions

### Frozen Dataclasses Everywhere

All DTOs use `@dataclass(frozen=True)` with `__post_init__` validation. This enforces immutability at runtime and catches invalid data at construction time rather than during processing.

### Typed Checkpoint Envelope

`CheckpointRunner` serializes stage results using an envelope format with a `__type__` field. A type registry maps string names to classes, enabling polymorphic deserialization without `pickle` or unsafe eval.

### Schema Registry (Audio)

The audio pipeline maintains a `_SCHEMA_MAP` registry mapping schema IDs to `(expected_type, validator_fn)` pairs. Each validator returns a list of human-readable error strings rather than raising exceptions, allowing batch validation.

### Threaded Video Backbone

The video pipeline uses a three-thread architecture (Reader → Processor → Encoder) connected via `queue.Queue` with sentinel-based EOF signaling. This decouples I/O from computation.

### Single-Threaded Audio

The audio pipeline is explicitly single-threaded (frozen in the v4.2.1 spec) to avoid complexity in sequential signal processing stages where each stage depends on the previous output.

### Non-Intrusive Observability

All observability modules (metrics, checkpoints, failure capture, debug reports) are layered on top of pipeline execution via middleware, decorators, and context managers. They never modify pipeline DTOs or execution logic.

### Error Hierarchy

The audio pipeline defines a dedicated exception hierarchy rooted at `AudioPipelineError` with specific subclasses for each stage (e.g., `AudioExtractionError`, `TTSGenerationError`). This enables precise error handling and failure-stage detection in debug reports.

### Configuration as Strings

All configuration values are stored as strings (`Dict[str, str]`) and parsed at point-of-use. This simplifies serialization, JSON round-tripping, and CLI argument handling while keeping the config layer type-agnostic.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | NDArray frame representation, audio signal math, WAV sample conversion |

All other imports are Python standard library (`dataclasses`, `typing`, `hashlib`, `json`, `logging`, `wave`, `struct`, `threading`, `queue`, `collections`, `math`, `re`, `os`, `time`, `pathlib`, `contextlib`, `functools`, `argparse`, `datetime`, `platform`, `traceback`, `types`).

---

## Related Documentation

- [Video Pipeline v3.2.0 Freeze Spec](architecture/video_pipeline.md)
- [Video Pipeline Validation & Architecture](freeze/video_freeze.md)
- [Audio Pipeline v4.2.1 Freeze Spec](freeze/audio_freeze.md)
