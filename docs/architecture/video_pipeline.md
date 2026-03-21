VIDEO PIPELINE v3.2.0 — Face Fusion Extended Freeze Specification (FINAL WITH OBSERVABILITY)
● Status: FROZEN
Observability
● Scope: Architecture, Module Map, Interfaces, Data Models, Runtime Rules, Dependency Rules, Change Policy,
● Compatibility target: Deterministic video-side extension; audio unchanged
⸻
1. Architecture Freeze
● Frozen execution flow
FrameSource → ReaderThread → ProcessorThread → EncoderThread
→ BackboneExecutionCore → VideoPipeline.run() → VideoResult
→ AudioInputAdapter → AudioInputContract → AudioPipeline.run() → AudioResult
● Frozen per-frame processing flow
ROI detection → OCR extraction → Subtitle tracking → Translation cache lookup
→ Translation adapter → Recolor engine → Face Fusion → Subtitle rendering → Frame output
● Architectural invariants
● Face Fusion is inside ProcessorThread only
● Subtitle rendering is final overlay stage
● AudioPipeline runs only after video threads join
● No new threads, queues, or async primitives
● VideoResult remains unchanged boundary
⸻
2. Module Map
● Core Modules (unchanged)
● M1–M20: unchanged
● M12 FaceFusionEngine
● Responsibility: face swap / blend
● Placement: after RecolorEngineAdapter, before SubtitleRenderer
● Observability Modules (NON-INTRUSIVE)
● O1 CheckpointRunner
● Stage execution with checkpoint + resume
● Typed serialization envelope enforced
● O2 MetricsInstrumentation
● Decorator-based stage metrics
● O3 FailureCaptureMiddleware
● Wrap pipeline, capture failure, trigger debug report
● O4 RunFolderManager
● Manage artifacts / checkpoints / logs / debug_report path
● O5 DebugReportGenerator
● Generate structured debug report JSON
● Module rules
● Observability modules are OUTSIDE pipeline graph
● No modification to M1–M20 responsibilities
● No new thread / queue
⸻
3. Interface Freeze
● IF-01 PipelineInterface
● IF-02 ConfigLoader
● IF-03 VideoPipeline
● IF-04 ROI
● IF-05 OCR
● IF-06 Translation
● IF-07 Recolor
● IF-08 FaceFusion
fuse_face(self, frame: ndarray[uint8], face_fusion_config: FaceFusionConfig) ->
ndarray[uint8]
● IF-09 Rendering
render_subtitle(self, frame: ndarray[uint8], subtitle_text: str) -> ndarray[uint8]
● IF-10 Encoding
encode_frame(self, frame: ndarray[uint8]) -> None
● Interface rules
● No parameter change
● No rename / retype
● FaceFusion only operates on frame + config
● Observability MUST NOT introduce new frozen interfaces
⸻
4. Data Model (Contract) Freeze
● DM-01 → DM-08: unchanged
● New Face Fusion DTOs
● DM-09 FaceFusionConfig
● enabled, mode, reference_
face
_path
● strength (range_
0
_1)
● min
_confidence (range_
0
_1)
● preserve_background
● DM-10 FaceRegion
● x, y, width, height
● confidence (range_
0
_1)
● track
id
_
● DM-11 FaceFusionMask
● mask: ndarray[bool]
● blur
radius ≥ 0
_
● DM-12 FaceFusionResult
● fused
frame
_
● fused
face
count
_
_
● metadata
● Data rules
● All DTOs frozen=True
● No mutation
● No schema change
● Observability constraint
● Debug data MUST NOT be added into DTO layer
● Debug uses external JSON only
⸻
5. Runtime Rules
● Execution model
● 3 threads only: Reader, Processor, Encoder
● Audio runs after video
● Determinism
● identical input → identical output
● no randomness
● no time dependency
● Threading
● threading + queue only
● no asyncio / multiprocessing
● Sentinel protocol
● injected once per stage
● never duplicated
● Error handling
● fatal
error
holder = first-failure-wins
_
_
● stop_
event overrides sentinel
⸻
6. Observability Runtime (Non-Intrusive)
● CheckpointRunner
● wraps stage execution, no logic change
● requires DTO to_dict/from_
dict
● MetricsInstrumentation
● decorator-based measurement
● FailureCaptureMiddleware
● wraps pipeline entrypoint
● triggers debug report on exception
● DebugReportGenerator
● writes JSON report only
● no effect on pipeline output
● RunFolderManager
● manages filesystem only
⸻
7. Dependency Rules
● Layering
DTO ← Core ← Processing ← Threading ← VideoPipeline
← AudioInputAdapter ← AudioPipeline
● Observability constraints
● Observability MUST NOT be imported by DTO layer
● No circular imports
● Observability → Pipeline (read-only) only
⸻
8. Change Policy
● Frozen
● architecture
● module map
● interfaces
● DTOs
● runtime rules
● dependency rules
● Allowed
● internal FaceFusion implementation
● observability (non-intrusive)
● Prohibited
● interface change
● DTO change
● new thread / queue
● video–audio coupling
● nondeterminism
⸻
9. Debug Pipeline Flow (External)
● RunFolderManager.create()
→ create run directory
● FailureCaptureMiddleware.run(pipeline_callable)
→ execute pipeline
● Optional inside pipeline:
● CheckpointRunner.run_stage()
● @instrument_stage(metrics, stage_name)
● On failure:
→ DebugReportGenerator.generate(…)
⸻
10. Debug Report Content
● pipeline metadata (run_id, version, timestamp)
● failed stage detection via metrics
● error + stack trace
● config snapshot
● environment info
● artifact status
● checkpoint snapshot
● log tail
⸻
11. Final Freeze Statement
● Architecture v3.2.0 is fully consistent and frozen
● Face Fusion integrated without breaking contracts
● Debug pipeline added as external observability layer
● No violation of:
● interface freeze
● DTO immutability
● threading model
● determinism
● dependency rules
