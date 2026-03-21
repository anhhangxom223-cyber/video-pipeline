Nội dung dưới đây đã được chuẩn hóa toàn bộ bullet thành ●.  ￼

PHASE 4 — FREEZE DOCUMENT
Audio Pipeline v4.2.1 — Enterprise Freeze Specification
Version: 1.0
Status: FROZEN
Effective: Immediately upon approval
Scope: Architecture, Service Interfaces, Contracts, Data Schemas, Dependencies, Runtime, Thread
Model

1. ARCHITECTURE FREEZE

1.1 Frozen Execution Flow
The audio pipeline executes exactly 10 steps in this fixed order. No step may be added, removed, or
reordered.
Step
Module
ID
Module Name Input Output
1 M20.1 AudioExtractor AudioInputContract ExtractedAudio
2 M20.2 SourceSeparator ExtractedAudio SeparatedAudio
3 M20.3 VoiceAnalyzer voice_audio_path: str VoiceAnalysisResult
4 M20 _build_segment_map Tuple[ASRSegment] MappingProxyType
5 M20.4 SubtitleAligner translated_srt_path,
asr_segments List[AlignedSegment]
6 M20.5 TTSGenerator
aligned_segments,
asr_segment_map,
voice_name,
speaking_rate_range
List[TTSOutputSegment]
7 M20.6 DurationAligner tts_segments List[DurationAlignedSegment]
8 M20.7 RMSProcessor
duration_aligned_segments,
asr_segment_map,
frame_ms
List[RMSProcessedSegment]
9 M20.8 AudioMixer rms_segments,
background_audio_path
MixedAudio
10 M20 return MixedAudio + metadata AudioResult

1.2 Frozen Module Count
Total modules: 9 (1 orchestrator + 8 sub-modules). No new modules permitted.

1.3 Frozen Dependency Direction
DTO Layer (audio_contracts.py)
 ↑
Sub-modules (M20.1–M20.8)
 ↑
Orchestrator (AudioPipeline M20)

Prohibited imports from audio pipeline:
● VideoPipeline
● ReaderThread, ProcessorThread, EncoderThread
● BackboneExecutionCore

1.4 Frozen Prosody Governance Rules
Rule
ID
Rule Enforcement
PG-01
Prosody exists ONLY inside
ASRSegment.prosody
Structural (SegmentProsody nested in
ASRSegment only)
PG-02 ASRSegment is immutable (frozen=True) Runtime (FrozenInstanceError)
PG-03 SegmentProsody is immutable (frozen=True) Runtime (FrozenInstanceError)
PG-04
Segment map built exactly once in
AudioPipeline.run() Code review + test
PG-05
Segment map is immutable
(MappingProxyType) Runtime (TypeError on write)
PG-06 SubtitleAligner must NOT access .prosody AST enforcement test
PG-07
DurationAligner must NOT access .prosody or
segment map
AST enforcement test
PG-08
TTSGenerator reads prosody ONLY via
asr_segment_map
Code review + test
PG-09
RMSProcessor reads prosody ONLY via
asr_segment_map
Code review + test
PG-10 No service may rebuild the segment map Code review

2. SERVICE INTERFACE FREEZE

2.1 Frozen Method Signatures
Every method signature below is locked. No parameter may be added, removed, renamed, or
retyped.

IF-A01: AudioExtractor.extract
extract(self, audio_input: AudioInputContract) -> ExtractedAudio

IF-A02: SourceSeparator.separate
separate(self, extracted_audio: ExtractedAudio) -> SeparatedAudio

IF-A03: VoiceAnalyzer.analyze
analyze(self, voice_audio_path: str) -> VoiceAnalysisResult

IF-A04: SubtitleAligner.align
align(self, translated_srt_path: str, asr_segments: Tuple[ASRSegment, ...]) -> List[AlignedSegment]

IF-A05: TTSGenerator.generate_batch
generate_batch(self, aligned_segments: List[AlignedSegment], asr_segment_map: MappingProxyType, voice_name: str, speaking_rate_range: Tuple[float, float]) -> List[TTSOutputSegment]

IF-A06: DurationAligner.align
align(self, tts_segments: List[TTSOutputSegment]) -> List[DurationAlignedSegment]

IF-A07: RMSProcessor.apply
apply(self, duration_aligned_segments: List[DurationAlignedSegment], asr_segment_map: MappingProxyType, frame_ms: int) -> List[RMSProcessedSegment]

IF-A08: AudioMixer.mix
mix(self, rms_segments: List[RMSProcessedSegment], background_audio_path: str) -> MixedAudio

IF-A09: AudioPipeline.run
run(self, audio_input: AudioInputContract) -> AudioResult

IF-A10: AudioPipeline._build_segment_map (static)
_build_segment_map(asr_segments) -> MappingProxyType

Total frozen interfaces: 10

3. CONTRACT FREEZE

3.1 Frozen DTO Registry
Contract ID DTO Name Field Count Frozen Nested
ADM-01 TimeRange 2 Yes —
ADM-02 PitchFrame 3 Yes —
ADM-03 RMSFrame 2 Yes —
ADM-04 SegmentProsody 3 Yes ADM-02, ADM-03
ADM-05 ASRSegment 6 Yes ADM-01, ADM-04
ADM-06 ExtractedAudio 4 Yes —
ADM-07 SeparatedAudio 2 Yes —
ADM-08 VoiceAnalysisResult 1 Yes ADM-05
ADM-09 AlignedSegment 4 Yes ADM-01
ADM-10 TTSOutputSegment 5 Yes ADM-01
ADM-11 DurationAlignedSegment 5 Yes ADM-01
ADM-12 RMSProcessedSegment 4 Yes ADM-01
ADM-13 MixedAudio 3 Yes —
ADM-14 AudioInputContract 3 Yes —
ADM-15 AudioResult 2 Yes —
Total frozen contracts: 15 (49 fields)

3.2 Contract Change Policy
Policy Rule Description
No field addition No new field may be added to any frozen DTO
No field removal No field may be removed
No type modification Field types are locked
No name change Field names are locked
No frozen flag change All DTOs must remain frozen=True
Versioned updates only Any change requires new version number

4. DATA SCHEMA FREEZE

4.1 Frozen Field Specifications
All field constraints defined in the schema registry (ADM-01 through ADM-15) are frozen. The
complete constraint set:
Constraint Fields Using It
non_empty_string audio_path (×5), voice_audio_path, background_audio_path, transcript,
silent_video_path, translated_srt_path
positive frequency_hz, speech_rate, sample_rate, channels, stretch_ratio
non_negative start, end, time_offset (×3), rms_value, segment_id (×5), duration_seconds
(×2), adjusted_duration
range_0_1 confidence, no_speech_prob
finite_float All float fields
valid_tuple rms_frames, pitch_frames, asr_segments

4.2 Schema Change Policy
Rule Description
No constraint addition No new constraint may be added to frozen fields
No constraint removal No constraint may be removed
No bound modification Constraint parameters (e.g., range limits) are locked
No schema ID change ADM-01 through ADM-15 IDs are permanent

5. DEPENDENCY FREEZE

5.1 Audio Pipeline Dependencies
The audio pipeline introduces no new external dependencies beyond those already frozen in
Video Pipeline v3.1.1.

Inherited frozen dependencies (from Video Pipeline Dependency Freeze v1.1):
numpy==1.26.4

Audio pipeline uses ONLY:
Python standard library (dataclasses, types, typing, hashlib, os, math)
numpy (inherited, for future audio array processing)

5.2 Prohibited New Dependencies
No additional pip packages may be introduced without:
1. New Dependency Freeze document
2. Hash-locked requirements.txt update
3. Compatibility verification with Video Pipeline v3.1.1

5.3 Internal Module Dependencies
Module Imports From
All sub-modules core.interfaces.audio_contracts
AudioPipeline core.audio.* (all 8 sub-modules)
No module core.video.*, threading, BackboneExecutionCore

6. RUNTIME FREEZE

6.1 Inherited Runtime Constraints
All runtime constraints from Video Pipeline Runtime Freeze v1.1 apply:
● Python 3.11.8 (CPython)
● PYTHONHASHSEED=0
● UTF-8 encoding
● No asyncio, no multiprocessing

6.2 Audio-Specific Runtime Rules
Rule Specification
Execution model Sequential within AudioPipeline.run()
Memory model No shared mutable state between modules
Determinism Identical input MUST produce identical output
No randomness No random module usage without fixed seed
No time-dependent logic No datetime/time-based decisions
No global mutable state All state passed via method arguments
Error handling AudioPipelineError hierarchy, fatal by default
Exit codes 5 = runtime failure, 6 = governance violation

6.3 Immutability Runtime Enforcement
Mechanism What It Protects
frozen=True dataclass All 15 DTOs — raises FrozenInstanceError
MappingProxyType Segment lookup map — raises TypeError
Tuple (not list) Prosody frame collections — structurally immutable

7. THREAD MODEL FREEZE

7.1 Audio Pipeline Thread Model
The audio pipeline is single-threaded. It executes entirely within the calling thread after
VideoPipeline.run() completes.
Property Value
Thread count 0 (runs in caller’s thread)
Concurrency model Sequential
Queue usage None
Sentinel protocol Not applicable
stop_event Not applicable

7.2 Interaction with Video Pipeline Threading
ReaderThread → ProcessorThread → EncoderThread
 │
 ▼
 BackboneExecutionCore
 │
 ▼
 VideoPipeline.run()
 │
 ▼
 VideoResult
 │
 ▼
 AudioInputAdapter (M19)
 │
 ▼
 AudioPipeline.run() (M20) ← SINGLE-THREADED
 │
 ▼
 AudioResult
The audio pipeline begins only after all video pipeline threads have joined. There is no concurrent
execution between video and audio pipelines.

7.3 Thread Model Prohibitions
Prohibition Reason
No threading.Thread inside audio pipeline Sequential execution model
No queue.Queue No inter-stage queuing
No asyncio Runtime freeze
No multiprocessing Runtime freeze
No concurrent.futures Sequential execution model

8. FREEZE ENFORCEMENT SUMMARY
Freeze Area Frozen Items Enforcement Method
Architecture 10 steps, 9 modules, dependency direction Automated test
Service Interface 10 method signatures Signature inspection test
Contract 15 DTOs, 54 fields Field completeness test
Data Schema 15 schemas, all constraints Schema validator test
Dependency 0 new external deps Import analysis test
Runtime Sequential, deterministic, immutable Determinism + immutability test
Thread Model Single-threaded, no concurrency Import prohibition test

9. CHANGE POLICY (LOCKED STATE)
Policy Description
No architecture modification Execution flow, module count, dependency direction locked
No interface modification Method signatures locked
No contract modification DTOs, fields, types locked
No schema modification Constraints, IDs locked
No dependency addition No new pip packages
No runtime modification Execution model, determinism rules locked
No threading introduction Single-threaded model locked
Versioned updates only Any change requires new freeze document version

10. COMPLETENESS CONFIRMATION
Metric Value
Architecture steps frozen 10
Modules frozen 9
Interfaces frozen 10
Contracts frozen 15
Total fields frozen 49
Schema IDs frozen ADM-01 → ADM-15 (continuous)
External dependencies added 0
Thread count 0 (single-threaded)
Prosody governance rules 10
Freeze Document v1.0 — Complete and ready for enforcement.
