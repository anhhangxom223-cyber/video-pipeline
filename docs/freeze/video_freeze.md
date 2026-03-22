# VIDEO PIPELINE v3.2.0

## Face Fusion Extended Consolidated Architecture & Contract Reference
### Validation-Driven Freeze Specification

Status: FROZEN

Scope: Architecture, Module Map, Interfaces, Data Models, Runtime Rules, Dependency Rules, Change Policy

Baseline: VIDEO PIPELINE v3.1.1 — Consolidated Architecture & Contract Reference

Upgrade: Insert Face Fusion into ProcessorThread between Recolor and Subtitle Rendering

Compatibility target: Deterministic video-side extension; audio freeze remains unchanged

## 1. Validation & Dry-Run Summary

Validation result: PASS after consistency fixes.

Check | Result | Fix / Note
● Data flow mismatch | PASS | Face Fusion is internal to ProcessorThread; no change to audio handoff.
● Interface inconsistency | PASS | Added IF-10 for FaceFusion; existing IF-01..IF-09 remain unchanged.
● Threading issues | PASS | No new thread, no new queue, sentinel protocol unchanged.
● Determinism violation | PASS | Face Fusion is constrained to pure per-frame processing, no randomness/time-based logic.
● Dependency rule violation | PASS | No audio-side import of video modules; no circular imports.
● DTO completeness | PASS | All DTOs frozen=True and explicitly defined, including new Face Fusion DTOs.

Applied design fix: Face Fusion is treated as an internal video-stage module only. It is called after RecolorEngineAdapter and before SubtitleRenderer, while VideoResult and AudioInputContract remain unchanged.

## 2. Architecture Freeze

Frozen end-to-end execution flow:

FrameSource -> ReaderThread -> ProcessorThread -> EncoderThread -> BackboneExecutionCore -> VideoPipeline.run() -> VideoResult -> AudioInputAdapter -> AudioInputContract -> AudioPipeline.run() -> AudioResult

Frozen per-frame processing flow:

ROI detection -> OCR extraction -> Subtitle tracking -> Translation cache lookup -> Translation adapter -> Recolor engine -> Face Fusion -> Subtitle rendering -> Frame output

Architectural invariants:
● Face Fusion is inserted inside ProcessorThread only.
● Subtitle rendering remains the final overlay stage before encoding.
● AudioPipeline remains outside the video freeze boundary and starts only after video threads join.
● No new queues, threads, or asynchronous execution primitives are introduced.
● VideoResult stays the same contract boundary for downstream audio consumption.

## 3. Module Map

The existing module family M1-M20 remains compatible with v3.1.1. M21 is added for Face Fusion.

ID | Module | Responsibility | Placement / Notes
● M1 | ConfigLoader | Resolve CLI + JSON config into ResolvedConfig. | Inherited
● M2 | DTO Layer | Frozen data models and contract definitions. | Inherited
● M3 | PipelineContext | Global context, caches, and factory wiring. | Inherited
● M4 | ROI Detector | Detect subtitle region and return ROIBox. | Inherited
● M5 | OCR Module | Extract subtitle text from video frames. | Inherited
● M6 | SubtitleTracker | Track text blocks across frames. | Inherited
● M7 | TranslationBufferManager | Batch subtitle segments for translation. | Inherited
● M8 | TranslationAdapter | Translate batches with timeout and retry rules. | Inherited
● M9 | TranslationCache | SHA-256 keyed translation cache, single-thread owned. | Inherited
● M10 | MappingConfigLoader | Load MappingConfig JSON. | Inherited
● M11 | RecolorEngineAdapter | Apply hue/saturation/value recoloring. | Inherited
● M12 | FaceFusionEngine | Perform face swap / blend on processed frames. | New in v3.2.0; inserted after Recolor and before Rendering
● M13 | SubtitleRenderer | Render translated subtitles on the fused frame. | Inherited, execution follows M12
● M14 | FPSNormalizer | Normalize video frame rate to target FPS. | Inherited
● M15 | ReaderThread | Read frames and inject SENTINEL at EOF. | Inherited
● M16 | ProcessorThread | Run per-frame processing chain and forward SENTINEL. | Inherited
● M17 | EncoderThread | Encode frames, flush buffer, forward SENTINEL. | Inherited
● M18 | BackboneExecutionCore | Start/join threads, manage queues, handle errors. | Inherited
● M19 | VideoPipeline | Orchestrate the video pipeline and return VideoResult. | Inherited
● M20 | AudioInputAdapter | Convert VideoResult to AudioInputContract. | Inherited
● M21 | AudioPipeline | Run audio processing after video completion. | Outside video freeze boundary; unchanged

## 4. Interface Freeze

All frozen interfaces remain signature-locked. A new interface is added for Face Fusion without altering any inherited signatures.

ID | Interface | Frozen signature
● IF-01 | PipelineInterface | run(self) -> Any
● IF-02 | ConfigLoader | resolve(self, cli_args: dict, json_config: dict) -> dict[str, str]
● IF-03 | VideoPipeline | run(self) -> VideoResult
● IF-04 | ROI | detect_roi(self, frame: ndarray[uint8]) -> ROIBox
● IF-05 | OCR | perform_ocr(self, frame: ndarray[uint8]) -> list[str]
● IF-06 | Translation | translate_batch(self, texts: list[str]) -> list[str]
● IF-07 | Recolor | apply_hsv_recolor(self, frame: ndarray[uint8], mapping_config: MappingConfig) -> ndarray[uint8]
● IF-08 | FaceFusion | fuse_face(self, frame: ndarray[uint8], face_fusion_config: FaceFusionConfig) -> ndarray[uint8]
● IF-09 | Rendering | render_subtitle(self, frame: ndarray[uint8], subtitle_text: str) -> ndarray[uint8]
● IF-10 | Encoding | encode_frame(self, frame: ndarray[uint8]) -> None

Interface rules:
● No parameter may be added, removed, renamed, or retyped for any frozen interface.
● FaceFusion receives and returns only frame data plus its own frozen FaceFusionConfig.
● Audio interfaces are not part of the video freeze and remain governed by the audio freeze document.

## 5. Data Model (Contract) Freeze

All DTOs must be immutable (frozen=True). Existing video contracts remain unchanged; new Face Fusion contracts are added as frozen dataclasses.

ID | DTO | Fields | Constraints / Notes
● DM-01 | ResolvedConfig | configuration_parameters: dict[str, str] | Frozen; inherited
● DM-02 | PipelineContext | config, global_color_mapping, scene_mapping_cache, translation_cache | Frozen; caches remain read-only in runtime scope
● DM-03 | VideoResult | silent_video_path, translated_srt_path, mapping_config_path, metadata | Frozen; output boundary to audio stage
● DM-04 | VideoOutputContract | silent_video_path, translated_srt_path | Frozen; minimal downstream contract
● DM-05 | MappingConfig | mappings, engine_mode, scene_id | Frozen; MappingEntry nested and immutable
● DM-06 | MaskMap | dict[str, ndarray[bool]] | Frozen wrapper contract
● DM-07 | TranslationCache | dict[str, str] | Frozen ownership inside PipelineContext
● DM-08 | ROIBox | x, y, width, height | Frozen; integer coordinates only
● DM-09 | FaceFusionConfig | enabled, mode, reference_face_path, strength, min_confidence, preserve_background | range_0_1 on numeric thresholds; non_empty_string on path/mode
● DM-10 | FaceRegion | x, y, width, height, confidence, track_id | range_0_1 confidence; non-negative geometry
● DM-11 | FaceFusionMask | mask, blur_radius | mask is ndarray[bool]; blur_radius is non-negative
● DM-12 | FaceFusionResult | fused_frame, fused_face_count, metadata | Internal result DTO; frozen=True

Additional nested contract note: MappingEntry remains nested under MappingConfig and keeps the same field structure as v3.1.1.

## 6. Runtime Rules

Rule area | Specification
● Execution model | Three video threads only: ReaderThread, ProcessorThread, EncoderThread. AudioPipeline remains single-threaded after video completion.
● Determinism | Identical input MUST produce identical output. No randomness, no time-dependent logic, no hidden global mutable state.
● Immutability enforcement | All frozen DTOs use frozen=True. Face Fusion must not mutate upstream DTOs in place.
● Threading model | threading + queue only; no asyncio, no multiprocessing, no concurrent.futures.
● Sentinel protocol | SENTINEL is a unique singleton, injected exactly once by each upstream stage and never duplicated or reordered.
● Error handling | fatal_error_holder is first-failure-wins; stop_event always overrides sentinel flow.

Dry-run verification outcome: the video pipeline terminates cleanly, hands off a valid VideoResult, and the audio pipeline then executes on the caller thread without concurrency overlap.

## 7. Dependency Rules

Layering rule:

DTO Layer <- Core Modules <- Processing Modules <- Threading Backbone <- VideoPipeline <- AudioInputAdapter <- AudioPipeline

● AudioPipeline must not import VideoPipeline or any ReaderThread / ProcessorThread / EncoderThread symbols.
● DTO modules must not import runtime modules.
● Threading modules must not import pipeline orchestration modules.
● No circular imports are permitted across video, audio, or DTO layers.
● The FaceFusionEngine must only depend on the frozen video dependency set and internal DTOs; any new external package would require a separate dependency freeze.

Frozen dependency set: numpy, opencv-python, Pillow, paddleocr, paddlepaddle, google-generativeai, tqdm, protobuf, requests, plus standard library

## 8. Change Policy

Policy class | Rule
● Frozen | Architecture order, module map, interface signatures, DTO fields/types, runtime rules, dependency rules.
● Allowed | Implementation details inside FaceFusionEngine, internal optimization, config defaults that do not alter signatures or contracts.
● Prohibited | Field addition/removal/type change, interface signature change, new thread/queue, circular imports, audio coupling, nondeterministic behavior.
● Versioning | Any structural change requires a new versioned freeze document.

## 9. Final Freeze Statement

The validated v3.2.0 architecture is internally consistent after applying the Face Fusion placement fix. It preserves the existing video-to-audio boundary, keeps all DTOs immutable, uses explicit frozen signatures, and remains deterministic under the locked runtime and dependency rules.
