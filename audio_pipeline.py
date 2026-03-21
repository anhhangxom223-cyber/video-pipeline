"""
audio_pipeline.py — Audio Pipeline v4.2.1
Contracts (ADM-01→ADM-15), Schema Registry, AudioExtractor (M20.1),
SourceSeparator (M20.2), VoiceAnalyzer (M20.3), SubtitleAligner (M20.4),
TTSGenerator (M20.5), DurationAligner (M20.6), RMSProcessor (M20.7),
AudioMixer (M20.8), AudioPipeline (M20).
"""
from __future__ import annotations
import hashlib
import math
import os
import re
import struct
import wave
from dataclasses import dataclass
from types import MappingProxyType
from typing import List, Tuple
import numpy as np
from config_loader import stage_timer
# ============================================================
# Audio DTO Layer (ADM-01 → ADM-15)
# ============================================================
@dataclass(frozen=True)
class AudioInputContract:
silent_video_path: str
translated_srt_path: str
metadata: dict
def to_dict(self) -> dict:
return {
"silent_video_path": self.silent_video_path,
"translated_srt_path": self.translated_srt_path,
"metadata": dict(self.metadata),
}
@classmethod
def from_dict(cls, data: dict) -> "AudioInputContract":
return cls(
silent_video_path=data["silent_video_path"],
translated_srt_path=data["translated_srt_path"],
metadata=data.get("metadata", {}),
)
@dataclass(frozen=True)
class TimeRange:
start: float
end: float
@dataclass(frozen=True)
class PitchFrame:
time_offset: float
frequency_hz: float
confidence: float
@dataclass(frozen=True)
class RMSFrame:
time_offset: float
rms_value: float
@dataclass(frozen=True)
class SegmentProsody:
speech_rate: float
rms_frames: Tuple[RMSFrame, ...]
pitch_frames: Tuple[PitchFrame, ...]
@dataclass(frozen=True)
class ASRSegment:
segment_id: int
time_range: TimeRange
transcript: str
avg_logprob: float
no_speech_prob: float
prosody: SegmentProsody
@dataclass(frozen=True)
class ExtractedAudio:
audio_path: str
sample_rate: int
channels: int
duration_seconds: float
@dataclass(frozen=True)
class SeparatedAudio:
voice_audio_path: str
background_audio_path: str
@dataclass(frozen=True)
class VoiceAnalysisResult:
asr_segments: Tuple[ASRSegment, ...]
@dataclass(frozen=True)
class AlignedSegment:
segment_id: int
time_range: TimeRange
original_text: str
translated_text: str
@dataclass(frozen=True)
class TTSOutputSegment:
segment_id: int
time_range: TimeRange
audio_path: str
duration_seconds: float
pitch_shift_delta: float
@dataclass(frozen=True)
class DurationAlignedSegment:
segment_id: int
time_range: TimeRange
audio_path: str
adjusted_duration: float
stretch_ratio: float
@dataclass(frozen=True)
class RMSProcessedSegment:
segment_id: int
time_range: TimeRange
audio_path: str
rms_adjustment_delta: float
@dataclass(frozen=True)
class MixedAudio:
audio_path: str
integrated_lufs: float
peak_db: float
@dataclass(frozen=True)
class AudioResult:
audio_path: str
metadata: dict
def to_dict(self) -> dict:
return {
"audio_path": self.audio_path,
"metadata": dict(self.metadata),
}
@classmethod
def from_dict(cls, data: dict) -> "AudioResult":
return cls(
audio_path=data["audio_path"],
metadata=data.get("metadata", {}),
)
# ============================================================
# Error Hierarchy
# ============================================================
class AudioPipelineError(Exception):
pass
class AudioExtractionError(AudioPipelineError):
pass
class SourceSeparationError(AudioPipelineError):
pass
class VoiceAnalysisError(AudioPipelineError):
pass
class SubtitleAlignmentError(AudioPipelineError):
pass
class TTSGenerationError(AudioPipelineError):
pass
class DurationAlignmentError(AudioPipelineError):
pass
class RMSProcessingError(AudioPipelineError):
pass
class AudioMixingError(AudioPipelineError):
pass
class ProsodyAccessViolation(AudioPipelineError):
pass
class SegmentMapError(AudioPipelineError):
pass
# ============================================================
# Schema Registry (ADM-01 → ADM-15)
# ============================================================
def _is_finite_float(value: float) -> bool:
return isinstance(value, (int, float)) and math.isfinite(value)
def _is_positive(value) -> bool:
return isinstance(value, (int, float)) and value > 0
def _is_non_negative(value) -> bool:
return isinstance(value, (int, float)) and value >= 0
def _is_range_0_1(value: float) -> bool:
return _is_finite_float(value) and 0.0 <= value <= 1.0
def _is_non_empty_string(value: str) -> bool:
return isinstance(value, str) and len(value.strip()) > 0
def _is_valid_tuple(value) -> bool:
return isinstance(value, tuple)
def _validate_time_range(tr: TimeRange) -> list[str]:
errors = []
if not _is_finite_float(tr.start):
errors.append("TimeRange.start must be finite float")
elif not _is_non_negative(tr.start):
errors.append("TimeRange.start must be non-negative")
if not _is_finite_float(tr.end):
errors.append("TimeRange.end must be finite float")
elif not _is_non_negative(tr.end):
errors.append("TimeRange.end must be non-negative")
return errors
def _validate_pitch_frame(pf: PitchFrame) -> list[str]:
errors = []
if not _is_finite_float(pf.time_offset):
errors.append("PitchFrame.time_offset must be finite float")
if not _is_finite_float(pf.frequency_hz) or not _is_positive(pf.frequency_hz):
errors.append("PitchFrame.frequency_hz must be finite positive float")
if not _is_range_0_1(pf.confidence):
errors.append("PitchFrame.confidence must be in [0.0, 1.0]")
return errors
def _validate_rms_frame(rf: RMSFrame) -> list[str]:
errors = []
if not _is_finite_float(rf.time_offset):
errors.append("RMSFrame.time_offset must be finite float")
if not _is_finite_float(rf.rms_value) or not _is_non_negative(rf.rms_value):
errors.append("RMSFrame.rms_value must be finite non-negative float")
return errors
def _validate_segment_prosody(sp: SegmentProsody) -> list[str]:
errors = []
if not _is_finite_float(sp.speech_rate) or not _is_positive(sp.speech_rate):
errors.append("SegmentProsody.speech_rate must be finite positive float")
if not _is_valid_tuple(sp.rms_frames):
errors.append("SegmentProsody.rms_frames must be a tuple")
else:
for i, rf in enumerate(sp.rms_frames):
for e in _validate_rms_frame(rf):
errors.append(f"SegmentProsody.rms_frames[{i}]: {e}")
if not _is_valid_tuple(sp.pitch_frames):
errors.append("SegmentProsody.pitch_frames must be a tuple")
else:
for i, pf in enumerate(sp.pitch_frames):
for e in _validate_pitch_frame(pf):
errors.append(f"SegmentProsody.pitch_frames[{i}]: {e}")
return errors
def _validate_asr_segment(seg: ASRSegment) -> list[str]:
errors = []
if not _is_non_negative(seg.segment_id):
errors.append("ASRSegment.segment_id must be non-negative")
errors.extend(_validate_time_range(seg.time_range))
if not _is_non_empty_string(seg.transcript):
errors.append("ASRSegment.transcript must be non-empty string")
if not _is_finite_float(seg.avg_logprob):
errors.append("ASRSegment.avg_logprob must be finite float")
if not _is_range_0_1(seg.no_speech_prob):
errors.append("ASRSegment.no_speech_prob must be in [0.0, 1.0]")
errors.extend(_validate_segment_prosody(seg.prosody))
return errors
def _validate_extracted_audio(ea: ExtractedAudio) -> list[str]:
errors = []
if not _is_non_empty_string(ea.audio_path):
errors.append("ExtractedAudio.audio_path must be non-empty string")
if not _is_positive(ea.sample_rate):
errors.append("ExtractedAudio.sample_rate must be positive")
if not _is_positive(ea.channels):
errors.append("ExtractedAudio.channels must be positive")
if not _is_finite_float(ea.duration_seconds):
errors.append("ExtractedAudio.duration_seconds must be finite float")
return errors
def _validate_separated_audio(sa: SeparatedAudio) -> list[str]:
errors = []
if not _is_non_empty_string(sa.voice_audio_path):
errors.append("SeparatedAudio.voice_audio_path must be non-empty string")
if not _is_non_empty_string(sa.background_audio_path):
errors.append("SeparatedAudio.background_audio_path must be non-empty string")
return errors
def _validate_voice_analysis_result(var: VoiceAnalysisResult) -> list[str]:
errors = []
if not _is_valid_tuple(var.asr_segments):
errors.append("VoiceAnalysisResult.asr_segments must be a tuple")
else:
for i, seg in enumerate(var.asr_segments):
for e in _validate_asr_segment(seg):
errors.append(f"VoiceAnalysisResult.asr_segments[{i}]: {e}")
return errors
def _validate_aligned_segment(aseg: AlignedSegment) -> list[str]:
errors = []
if not _is_non_negative(aseg.segment_id):
errors.append("AlignedSegment.segment_id must be non-negative")
errors.extend(_validate_time_range(aseg.time_range))
if not _is_non_empty_string(aseg.original_text):
errors.append("AlignedSegment.original_text must be non-empty string")
if not _is_non_empty_string(aseg.translated_text):
errors.append("AlignedSegment.translated_text must be non-empty string")
return errors
def _validate_tts_output_segment(ts: TTSOutputSegment) -> list[str]:
errors = []
if not _is_non_negative(ts.segment_id):
errors.append("TTSOutputSegment.segment_id must be non-negative")
errors.extend(_validate_time_range(ts.time_range))
if not _is_non_empty_string(ts.audio_path):
errors.append("TTSOutputSegment.audio_path must be non-empty string")
if not _is_finite_float(ts.duration_seconds):
errors.append("TTSOutputSegment.duration_seconds must be finite float")
if not _is_finite_float(ts.pitch_shift_delta):
errors.append("TTSOutputSegment.pitch_shift_delta must be finite float")
return errors
def _validate_duration_aligned_segment(das: DurationAlignedSegment) -> list[str]:
errors = []
if not _is_non_negative(das.segment_id):
errors.append("DurationAlignedSegment.segment_id must be non-negative")
errors.extend(_validate_time_range(das.time_range))
if not _is_non_empty_string(das.audio_path):
errors.append("DurationAlignedSegment.audio_path must be non-empty string")
if not _is_finite_float(das.adjusted_duration):
errors.append("DurationAlignedSegment.adjusted_duration must be finite float")
if not _is_finite_float(das.stretch_ratio) or not _is_positive(das.stretch_ratio):
errors.append("DurationAlignedSegment.stretch_ratio must be finite positive float")
return errors
def _validate_rms_processed_segment(rps: RMSProcessedSegment) -> list[str]:
errors = []
if not _is_non_negative(rps.segment_id):
errors.append("RMSProcessedSegment.segment_id must be non-negative")
errors.extend(_validate_time_range(rps.time_range))
if not _is_non_empty_string(rps.audio_path):
errors.append("RMSProcessedSegment.audio_path must be non-empty string")
if not _is_finite_float(rps.rms_adjustment_delta):
errors.append("RMSProcessedSegment.rms_adjustment_delta must be finite float")
return errors
def _validate_mixed_audio(ma: MixedAudio) -> list[str]:
errors = []
if not _is_non_empty_string(ma.audio_path):
errors.append("MixedAudio.audio_path must be non-empty string")
if not _is_finite_float(ma.integrated_lufs):
errors.append("MixedAudio.integrated_lufs must be finite float")
if not _is_finite_float(ma.peak_db):
errors.append("MixedAudio.peak_db must be finite float")
return errors
def _validate_audio_input_contract(aic: AudioInputContract) -> list[str]:
errors = []
if not _is_non_empty_string(aic.silent_video_path):
errors.append("AudioInputContract.silent_video_path must be non-empty string")
if not _is_non_empty_string(aic.translated_srt_path):
errors.append("AudioInputContract.translated_srt_path must be non-empty string")
if not isinstance(aic.metadata, dict):
errors.append("AudioInputContract.metadata must be a dict")
return errors
def _validate_audio_result(ar: AudioResult) -> list[str]:
errors = []
if not _is_non_empty_string(ar.audio_path):
errors.append("AudioResult.audio_path must be non-empty string")
if not isinstance(ar.metadata, dict):
errors.append("AudioResult.metadata must be a dict")
return errors
_SCHEMA_MAP = {
"ADM-01": (TimeRange, _validate_time_range),
"ADM-02": (PitchFrame, _validate_pitch_frame),
"ADM-03": (RMSFrame, _validate_rms_frame),
"ADM-04": (SegmentProsody, _validate_segment_prosody),
"ADM-05": (ASRSegment, _validate_asr_segment),
"ADM-06": (ExtractedAudio, _validate_extracted_audio),
"ADM-07": (SeparatedAudio, _validate_separated_audio),
"ADM-08": (VoiceAnalysisResult, _validate_voice_analysis_result),
"ADM-09": (AlignedSegment, _validate_aligned_segment),
"ADM-10": (TTSOutputSegment, _validate_tts_output_segment),
"ADM-11": (DurationAlignedSegment, _validate_duration_aligned_segment),
"ADM-12": (RMSProcessedSegment, _validate_rms_processed_segment),
"ADM-13": (MixedAudio, _validate_mixed_audio),
"ADM-14": (AudioInputContract, _validate_audio_input_contract),
"ADM-15": (AudioResult, _validate_audio_result),
}
def validate_instance(instance, schema_id: str) -> list[str]:
if schema_id not in _SCHEMA_MAP:
return [f"Unknown schema_id: {schema_id}"]
expected_type, validator_fn = _SCHEMA_MAP[schema_id]
if not isinstance(instance, expected_type):
return [f"Expected {expected_type.__name__}, got {type(instance).__name__}"]
return validator_fn(instance)
# ============================================================
# Utility: WAV read/write
# ============================================================
def _read_wav(path: str) -> tuple:
with wave.open(path, "rb") as wf:
n_channels = wf.getnchannels()
sample_width = wf.getsampwidth()
sample_rate = wf.getframerate()
n_frames = wf.getnframes()
raw = wf.readframes(n_frames)
if sample_width == 2:
fmt = f"<{n_frames * n_channels}h"
samples = np.array(struct.unpack(fmt, raw), dtype=np.float64)
samples /= 32768.0
elif sample_width == 1:
samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
samples = (samples - 128.0) / 128.0
else:
raise ValueError(f"Unsupported sample width: {sample_width}")
if n_channels > 1:
samples = samples.reshape(-1, n_channels).mean(axis=1)
return samples, sample_rate
def _write_wav(path: str, samples: np.ndarray, sample_rate: int = 44100) -> None:
samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
with wave.open(path, "wb") as wf:
wf.setnchannels(1)
wf.setsampwidth(2)
wf.setframerate(sample_rate)
wf.writeframes(samples_int16.tobytes())
# ============================================================
# AudioExtractor (M20.1)
# ============================================================
class AudioExtractor:
SAMPLE_RATE = 44100
CHANNELS = 1
def extract(self, audio_input: AudioInputContract) -> ExtractedAudio:
errors = validate_instance(audio_input, "ADM-14")
if errors:
raise AudioExtractionError(f"Invalid AudioInputContract: {'; '.join(errors)}")
video_path = audio_input.silent_video_path
if not os.path.isfile(video_path):
raise AudioExtractionError(f"Video file not found: {video_path}")
path_hash = hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:16]
output_dir = os.path.dirname(video_path) or "."
audio_path = os.path.join(output_dir, f"extracted_{path_hash}.wav")
self._generate_mock_wav(audio_path, duration_seconds=3.0)
duration = self._get_duration(audio_path)
extracted = ExtractedAudio(audio_path=audio_path, sample_rate=self.SAMPLE_RATE,
channels=self.CHANNELS, duration_seconds=duration)
out_errors = validate_instance(extracted, "ADM-06")
if out_errors:
raise AudioExtractionError(f"Invalid ExtractedAudio output: {'; '.join(out_errors
return extracted
def _generate_mock_wav(self, path: str, duration_seconds: float = 3.0) -> None:
n_samples = int(duration_seconds * self.SAMPLE_RATE)
t = np.arange(n_samples, dtype=np.float64) / self.SAMPLE_RATE
signal = np.sin(2.0 * 3.14159265 * 440.0 * t) * 0.3
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
_write_wav(path, signal, self.SAMPLE_RATE)
@staticmethod
def _get_duration(audio_path: str) -> float:
try:
with wave.open(audio_path, "rb") as wf:
return wf.getnframes() / wf.getframerate()
except Exception as e:
raise AudioExtractionError(f"Failed to get duration for {audio_path}: {e}")
# ============================================================
# SourceSeparator (M20.2)
# ============================================================
class SourceSeparator:
VOICE_LOW_HZ = 300
VOICE_HIGH_HZ = 3400
SAMPLE_RATE = 44100
def separate(self, extracted_audio: ExtractedAudio) -> SeparatedAudio:
errors = validate_instance(extracted_audio, "ADM-06")
if errors:
raise SourceSeparationError(f"Invalid ExtractedAudio: {'; '.join(errors)}")
audio_path = extracted_audio.audio_path
if not os.path.isfile(audio_path):
raise SourceSeparationError(f"Audio file not found: {audio_path}")
path_hash = hashlib.sha256(audio_path.encode("utf-8")).hexdigest()[:16]
output_dir = os.path.dirname(audio_path) or "."
voice_path = os.path.join(output_dir, f"{path_hash}_voice.wav")
background_path = os.path.join(output_dir, f"{path_hash}_background.wav")
samples, sr = _read_wav(audio_path)
_write_wav(voice_path, samples * 0.8, sr)
_write_wav(background_path, samples * 0.2, sr)
separated = SeparatedAudio(voice_audio_path=voice_path, background_audio_path=backgro
out_errors = validate_instance(separated, "ADM-07")
if out_errors:
raise SourceSeparationError(f"Invalid SeparatedAudio output: {'; '.join(out_error
return separated
# ============================================================
# VoiceAnalyzer (M20.3)
# ============================================================
class VoiceAnalyzer:
FRAME_SIZE = 1024
HOP_SIZE = 512
PITCH_FRAMES_PER_SEGMENT = 10
RMS_FRAMES_PER_SEGMENT = 10
MIN_SEGMENT_DURATION = 0.5
def analyze(self, voice_audio_path: str) -> VoiceAnalysisResult:
if not isinstance(voice_audio_path, str) or not voice_audio_path.strip():
raise VoiceAnalysisError("voice_audio_path must be non-empty string")
if not os.path.isfile(voice_audio_path):
raise VoiceAnalysisError(f"Voice audio file not found: {voice_audio_path}")
try:
samples, sample_rate = _read_wav(voice_audio_path)
except Exception as e:
raise VoiceAnalysisError(f"Failed to read WAV: {e}")
if len(samples) == 0:
raise VoiceAnalysisError("WAV file contains no samples")
duration = len(samples) / sample_rate
file_hash = hashlib.sha256(voice_audio_path.encode("utf-8")).hexdigest()
seed = int(file_hash[:8], 16)
segments = self._segment_audio(samples, sample_rate, duration, seed)
if not segments:
raise VoiceAnalysisError("No speech segments detected")
result = VoiceAnalysisResult(asr_segments=tuple(segments))
errors = validate_instance(result, "ADM-08")
if errors:
raise VoiceAnalysisError(f"Invalid VoiceAnalysisResult: {'; '.join(errors)}")
return result
def _segment_audio(self, samples: np.ndarray, sample_rate: int,
total_duration: float, seed: int) -> list[ASRSegment]:
num_segments = max(1, int(total_duration / 3.0))
num_segments = min(num_segments, 100)
segment_duration = total_duration / num_segments
segments = []
for i in range(num_segments):
seg_start = i * segment_duration
seg_end = min((i + 1) * segment_duration, total_duration)
if (seg_end - seg_start) < self.MIN_SEGMENT_DURATION:
continue
start_sample = int(seg_start * sample_rate)
end_sample = int(seg_end * sample_rate)
seg_samples = samples[start_sample:end_sample]
seg_seed = seed + i
prosody = self._compute_prosody(seg_samples, sample_rate, seg_start, seg_seed)
transcript = f"segment_{i}_transcript"
avg_logprob = -0.3 - (seg_seed % 50) / 100.0
no_speech_prob = (seg_seed % 20) / 100.0
segment = ASRSegment(segment_id=i, time_range=TimeRange(start=seg_start, end=seg_
transcript=transcript, avg_logprob=avg_logprob,
no_speech_prob=no_speech_prob, prosody=prosody)
segments.append(segment)
return segments
def _compute_prosody(self, samples: np.ndarray, sample_rate: int,
time_offset_base: float, seed: int) -> SegmentProsody:
duration = len(samples) / sample_rate if len(samples) > 0 else 0.1
pitch_frames = self._compute_pitch_frames(samples, sample_rate, time_offset_base, dur
rms_frames = self._compute_rms_frames(samples, sample_rate, time_offset_base, duratio
speech_rate = 0.8 + (seed % 70) / 100.0
return SegmentProsody(speech_rate=speech_rate, rms_frames=tuple(rms_frames),
pitch_frames=tuple(pitch_frames))
def _compute_pitch_frames(self, samples: np.ndarray, sample_rate: int,
base_offset: float, duration: float, seed: int) -> list[PitchFr
frames = []
step = max(duration / self.PITCH_FRAMES_PER_SEGMENT, 0.001)
for i in range(self.PITCH_FRAMES_PER_SEGMENT):
t = base_offset + i * step
center = int((i * step) * sample_rate)
start = max(0, center - self.FRAME_SIZE // 2)
end = min(len(samples), start + self.FRAME_SIZE)
frame_data = samples[start:end]
if len(frame_data) < 64:
freq = 100.0 + (seed + i) % 200
confidence = 0.5
else:
freq, confidence = self._autocorrelation_pitch(frame_data, sample_rate)
if freq <= 0:
freq = 100.0 + (seed + i) % 200
confidence = 0.3
frames.append(PitchFrame(time_offset=round(t, 6), frequency_hz=round(freq, 2),
confidence=round(min(max(confidence, 0.0), 1.0), 4)))
return frames
def _compute_rms_frames(self, samples: np.ndarray, sample_rate: int,
base_offset: float, duration: float) -> list[RMSFrame]:
frames = []
step = max(duration / self.RMS_FRAMES_PER_SEGMENT, 0.001)
for i in range(self.RMS_FRAMES_PER_SEGMENT):
t = base_offset + i * step
center = int((i * step) * sample_rate)
start = max(0, center - self.FRAME_SIZE // 2)
end = min(len(samples), start + self.FRAME_SIZE)
frame_data = samples[start:end]
rms_value = float(np.sqrt(np.mean(frame_data ** 2))) if len(frame_data) > 0 else
frames.append(RMSFrame(time_offset=round(t, 6), rms_value=round(max(rms_value, 0.
return frames
@staticmethod
def _autocorrelation_pitch(frame: np.ndarray, sample_rate: int) -> tuple[float, float]:
if len(frame) < 64:
return 0.0, 0.0
frame = frame.astype(np.float64)
frame = frame - np.mean(frame)
norm = np.sqrt(np.sum(frame ** 2))
if norm < 1e-10:
return 0.0, 0.0
frame = frame / norm
corr = np.correlate(frame, frame, mode="full")
corr = corr[len(corr) // 2:]
min_lag = max(1, sample_rate // 500)
max_lag = min(len(corr) - 1, sample_rate // 50)
if min_lag >= max_lag or max_lag >= len(corr):
return 0.0, 0.0
search = corr[min_lag:max_lag + 1]
if len(search) == 0:
return 0.0, 0.0
peak_idx = int(np.argmax(search)) + min_lag
confidence = float(corr[peak_idx])
if peak_idx == 0:
return 0.0, 0.0
freq = sample_rate / peak_idx
return freq, confidence
# ============================================================
# SubtitleAligner (M20.4) — Prosody-Blind (PG-06)
# ============================================================
class SubtitleAligner:
def align(self, translated_srt_path: str,
asr_segments: Tuple[ASRSegment, ...]) -> List[AlignedSegment]:
if not isinstance(translated_srt_path, str) or not translated_srt_path.strip():
raise SubtitleAlignmentError("translated_srt_path must be non-empty string")
if not isinstance(asr_segments, tuple):
raise SubtitleAlignmentError("asr_segments must be a tuple")
if len(asr_segments) == 0:
raise SubtitleAlignmentError("asr_segments is empty")
if not os.path.isfile(translated_srt_path):
raise SubtitleAlignmentError(f"SRT file not found: {translated_srt_path}")
try:
srt_entries = self._parse_srt(translated_srt_path)
except Exception as e:
raise SubtitleAlignmentError(f"Failed to parse SRT: {e}")
aligned = self._align_segments(asr_segments, srt_entries)
for seg in aligned:
errors = validate_instance(seg, "ADM-09")
if errors:
raise SubtitleAlignmentError(
f"Invalid AlignedSegment (id={seg.segment_id}): {'; '.join(errors)}")
return aligned
def _align_segments(self, asr_segments: Tuple[ASRSegment, ...],
srt_entries: list[dict]) -> List[AlignedSegment]:
aligned = []
for seg in asr_segments:
seg_id = seg.segment_id
seg_start = seg.time_range.start
seg_end = seg.time_range.end
original_text = seg.transcript
best_text = self._find_best_match(seg_start, seg_end, srt_entries)
if not best_text:
best_text = f"[untranslated:{seg_id}]"
aligned.append(AlignedSegment(segment_id=seg_id,
time_range=TimeRange(start=seg_start, end=seg_end),
original_text=original_text, translated_text=best_t
return aligned
@staticmethod
def _find_best_match(seg_start: float, seg_end: float, srt_entries: list[dict]) -> best_overlap = 0.0
str:
best_text = ""
for entry in srt_entries:
overlap = max(0.0, min(seg_end, entry["end"]) - max(seg_start, entry["start"]))
if overlap > best_overlap:
best_overlap = overlap
best_text = entry["text"]
return best_text
@staticmethod
def _parse_srt(srt_path: str) -> list[dict]:
with open(srt_path, "r", encoding="utf-8") as f:
content = f.read()
entries = []
blocks = re.split(r"\n\s*\n", content.strip())
for block in blocks:
lines = block.strip().split("\n")
if len(lines) < 3:
continue
try:
index = int(lines[0].strip())
except ValueError:
continue
tc_match = re.match(
r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
lines[1].strip())
if not tc_match:
continue
start = SubtitleAligner._srt_time_to_seconds(tc_match.group(1))
end = SubtitleAligner._srt_time_to_seconds(tc_match.group(2))
text = " ".join(line.strip() for line in lines[2:] if line.strip())
entries.append({"index": index, "start": start, "end": end, "text": text})
return entries
@staticmethod
def _srt_time_to_seconds(tc: str) -> float:
tc = tc.replace(",", ".")
parts = tc.split(":")
return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
# ============================================================
# TTSGenerator (M20.5) — Prosody via Map (PG-08)
# ============================================================
class TTSGenerator:
REFERENCE_PITCH_HZ = 120.0
SAMPLE_RATE = 44100
def generate_batch(self, aligned_segments: List[AlignedSegment],
asr_segment_map: MappingProxyType, voice_name: str,
speaking_rate_range: Tuple[float, float]) -> List[TTSOutputSegment]:
if not isinstance(aligned_segments, list):
raise TTSGenerationError("aligned_segments must be a list")
if not isinstance(asr_segment_map, MappingProxyType):
raise TTSGenerationError("asr_segment_map must be MappingProxyType")
if not isinstance(voice_name, str) or not voice_name.strip():
raise TTSGenerationError("voice_name must be non-empty string")
if not isinstance(speaking_rate_range, tuple) or len(speaking_rate_range) != 2:
raise TTSGenerationError("speaking_rate_range must be a tuple of (min, max)")
min_rate, max_rate = speaking_rate_range
if min_rate <= 0 or max_rate <= 0 or min_rate > max_rate:
raise TTSGenerationError(f"Invalid speaking_rate_range: ({min_rate}, {max_rate})"
results = []
for aligned_seg in aligned_segments:
errors = validate_instance(aligned_seg, "ADM-09")
if errors:
raise TTSGenerationError(
f"Invalid AlignedSegment (id={aligned_seg.segment_id}): {'; '.join(errors
seg_id = aligned_seg.segment_id
if seg_id not in asr_segment_map:
raise TTSGenerationError(f"segment_id {seg_id} not found in asr_segment_map")
asr_seg = asr_segment_map[seg_id]
prosody = asr_seg.prosody
avg_pitch = self._compute_avg_pitch(prosody.pitch_frames)
pitch_delta = avg_pitch - self.REFERENCE_PITCH_HZ
clamped_rate = max(min_rate, min(prosody.speech_rate, max_rate))
original_duration = aligned_seg.time_range.end - aligned_seg.time_range.start
char_ratio = len(aligned_seg.translated_text) / max(len(aligned_seg.original_text
tts_duration = max((char_ratio * original_duration) / clamped_rate, 0.1)
audio_path = self._synthesize(aligned_seg, voice_name, tts_duration, avg_pitch)
tts_seg = TTSOutputSegment(
segment_id=seg_id,
time_range=TimeRange(start=aligned_seg.time_range.start, end=aligned_seg.time
audio_path=audio_path, duration_seconds=round(tts_duration, 6),
pitch_shift_delta=round(pitch_delta, 4))
out_errors = validate_instance(tts_seg, "ADM-10")
if out_errors:
raise TTSGenerationError(
f"Invalid TTSOutputSegment (id={seg_id}): {'; '.join(out_errors)}")
results.append(tts_seg)
return results
@staticmethod
def _compute_avg_pitch(pitch_frames) -> float:
if not pitch_frames:
return TTSGenerator.REFERENCE_PITCH_HZ
return sum(pf.frequency_hz for pf in pitch_frames) / len(pitch_frames)
def _synthesize(self, aligned_seg: AlignedSegment, voice_name: str,
duration: float, pitch_hz: float) -> str:
key = f"{aligned_seg.segment_id}:{voice_name}:{aligned_seg.translated_text}"
path_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
output_dir = os.path.join(os.getcwd(), "tts_output")
os.makedirs(output_dir, exist_ok=True)
audio_path = os.path.join(output_dir, f"tts_{path_hash}.wav")
n_samples = int(duration * self.SAMPLE_RATE)
t = np.arange(n_samples, dtype=np.float64) / self.SAMPLE_RATE
signal = np.sin(2.0 * math.pi * pitch_hz * t) * 0.5
_write_wav(audio_path, signal, self.SAMPLE_RATE)
return audio_path
# ============================================================
# DurationAligner (M20.6) — No Prosody Access (PG-07)
# ============================================================
class DurationAligner:
STRETCH_MIN = 0.5
STRETCH_MAX = 2.0
def align(self, tts_segments: List[TTSOutputSegment]) -> List[DurationAlignedSegment]:
if not isinstance(tts_segments, list):
raise DurationAlignmentError("tts_segments must be a list")
if len(tts_segments) == 0:
raise DurationAlignmentError("tts_segments is empty")
results = []
for tts_seg in tts_segments:
errors = validate_instance(tts_seg, "ADM-10")
if errors:
raise DurationAlignmentError(
f"Invalid TTSOutputSegment (id={tts_seg.segment_id}): {'; '.join(errors)}
original_duration = tts_seg.time_range.end - tts_seg.time_range.start
if original_duration <= 0:
raise DurationAlignmentError(
f"Segment {tts_seg.segment_id}: non-positive original duration")
tts_duration = tts_seg.duration_seconds
if tts_duration <= 0:
raise DurationAlignmentError(
f"Segment {tts_seg.segment_id}: non-positive TTS duration")
stretch_ratio = max(self.STRETCH_MIN, min(original_duration / tts_duration, self.
adjusted_duration = tts_duration * stretch_ratio
da_seg = DurationAlignedSegment(
segment_id=tts_seg.segment_id,
time_range=TimeRange(start=tts_seg.time_range.start, end=tts_seg.time_range.e
audio_path=tts_seg.audio_path, adjusted_duration=round(adjusted_duration, 6),
stretch_ratio=round(stretch_ratio, 6))
out_errors = validate_instance(da_seg, "ADM-11")
if out_errors:
raise DurationAlignmentError(
f"Invalid DurationAlignedSegment (id={da_seg.segment_id}): {'; '.join(out
results.append(da_seg)
return results
# ============================================================
# RMSProcessor (M20.7) — Prosody via Map (PG-09)
# ============================================================
class RMSProcessor:
REFERENCE_RMS = 0.5
def apply(self, duration_aligned_segments: List[DurationAlignedSegment],
asr_segment_map: MappingProxyType, frame_ms: int) -> List[RMSProcessedSegment]:
if not isinstance(duration_aligned_segments, list):
raise RMSProcessingError("duration_aligned_segments must be a list")
if not isinstance(asr_segment_map, MappingProxyType):
raise RMSProcessingError("asr_segment_map must be MappingProxyType")
if not isinstance(frame_ms, int) or frame_ms <= 0:
raise RMSProcessingError(f"frame_ms must be a positive integer, got {frame_ms}")
if len(duration_aligned_segments) == 0:
raise RMSProcessingError("duration_aligned_segments is empty")
results = []
for da_seg in duration_aligned_segments:
errors = validate_instance(da_seg, "ADM-11")
if errors:
raise RMSProcessingError(
f"Invalid DurationAlignedSegment (id={da_seg.segment_id}): {'; '.join(err
seg_id = da_seg.segment_id
if seg_id not in asr_segment_map:
raise RMSProcessingError(f"segment_id {seg_id} not found in asr_segment_map")
asr_seg = asr_segment_map[seg_id]
rms_frames = asr_seg.prosody.rms_frames
avg_rms = (sum(rf.rms_value for rf in rms_frames) / len(rms_frames)) if rms_frame
delta = avg_rms - self.REFERENCE_RMS
rms_seg = RMSProcessedSegment(
segment_id=seg_id,
time_range=TimeRange(start=da_seg.time_range.start, end=da_seg.time_range.end
audio_path=da_seg.audio_path, rms_adjustment_delta=round(delta, 6))
out_errors = validate_instance(rms_seg, "ADM-12")
if out_errors:
raise RMSProcessingError(
f"Invalid RMSProcessedSegment (id={seg_id}): {'; '.join(out_errors)}")
results.append(rms_seg)
return results
# ============================================================
# AudioMixer (M20.8)
# ============================================================
class AudioMixer:
TARGET_LUFS = -14.0
PEAK_HEADROOM_DB = -1.0
SAMPLE_RATE = 44100
def mix(self, rms_segments: List[RMSProcessedSegment],
background_audio_path: str) -> MixedAudio:
if not isinstance(rms_segments, list):
raise AudioMixingError("rms_segments must be a list")
if len(rms_segments) == 0:
raise AudioMixingError("rms_segments is empty")
if not isinstance(background_audio_path, str) or not background_audio_path.strip():
raise AudioMixingError("background_audio_path must be non-empty string")
if not os.path.isfile(background_audio_path):
raise AudioMixingError(f"Background audio not found: {background_audio_path}")
for seg in rms_segments:
errors = validate_instance(seg, "ADM-12")
if errors:
raise AudioMixingError(
f"Invalid RMSProcessedSegment (id={seg.segment_id}): {'; '.join(errors)}"
try:
bg_samples, bg_sr = _read_wav(background_audio_path)
except Exception as e:
raise AudioMixingError(f"Failed to read background audio: {e}")
max_end = max(seg.time_range.end for seg in rms_segments)
total_samples = max(int(max_end * self.SAMPLE_RATE), len(bg_samples))
output = np.zeros(total_samples, dtype=np.float64)
if len(bg_samples) > 0:
bg_repeated = np.tile(bg_samples, int(np.ceil(total_samples / len(bg_samples))))[
output += bg_repeated * 0.3
for seg in rms_segments:
if not os.path.isfile(seg.audio_path):
raise AudioMixingError(f"Voice segment file not found: {seg.audio_path}")
try:
voice_samples, _ = _read_wav(seg.audio_path)
except Exception as e:
raise AudioMixingError(f"Failed to read voice segment {seg.segment_id}: {e}")
gain = 10.0 ** (seg.rms_adjustment_delta / 20.0) if seg.rms_adjustment_delta != 0
voice_samples = voice_samples * gain
start_idx = int(seg.time_range.start * self.SAMPLE_RATE)
end_idx = start_idx + len(voice_samples)
if end_idx > total_samples:
voice_samples = voice_samples[:total_samples - start_idx]
end_idx = total_samples
output[start_idx:end_idx] += voice_samples
avg_delta = np.mean([seg.rms_adjustment_delta for seg in rms_segments])
integrated_lufs = self.TARGET_LUFS + float(avg_delta) * 10
peak_db = self.PEAK_HEADROOM_DB - abs(float(avg_delta))
peak_val = np.max(np.abs(output))
if peak_val > 0.99:
output = output * (0.99 / peak_val)
output_hash = hashlib.sha256(background_audio_path.encode("utf-8")).hexdigest()[:16]
output_dir = os.path.join(os.getcwd(), "mix_output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"mixed_{output_hash}.wav")
_write_wav(output_path, output, self.SAMPLE_RATE)
mixed = MixedAudio(audio_path=output_path, integrated_lufs=round(integrated_lufs, 4),
peak_db=round(peak_db, 4))
out_errors = validate_instance(mixed, "ADM-13")
if out_errors:
raise AudioMixingError(f"Invalid MixedAudio: {'; '.join(out_errors)}")
return mixed
# ============================================================
# AudioPipeline (M20) — 10-Step Orchestrator
# ============================================================
class AudioPipeline:
DEFAULT_VOICE_NAME = "default"
DEFAULT_SPEAKING_RATE_RANGE = (0.8, 1.5)
DEFAULT_FRAME_MS = 25
def __init__(self):
self._extractor = AudioExtractor()
self._separator = SourceSeparator()
self._analyzer = VoiceAnalyzer()
self._aligner = SubtitleAligner()
self._tts = TTSGenerator()
self._duration_aligner = DurationAligner()
self._rms_processor = RMSProcessor()
self._mixer = AudioMixer()
def run(self, audio_input: AudioInputContract) -> AudioResult:
errors = validate_instance(audio_input, "ADM-14")
if errors:
raise AudioPipelineError(f"Invalid AudioInputContract: {'; '.join(errors)}")
with stage_timer("AudioPipeline.Step1.Extract"):
extracted = self._extractor.extract(audio_input)
with stage_timer("AudioPipeline.Step2.Separate"):
separated = self._separator.separate(extracted)
with stage_timer("AudioPipeline.Step3.Analyze"):
analysis = self._analyzer.analyze(separated.voice_audio_path)
with stage_timer("AudioPipeline.Step4.SegmentMap"):
asr_segment_map = self._build_segment_map(analysis.asr_segments)
with stage_timer("AudioPipeline.Step5.AlignSubtitles"):
aligned = self._aligner.align(audio_input.translated_srt_path, analysis.asr_segme
with stage_timer("AudioPipeline.Step6.TTS"):
tts_segments = self._tts.generate_batch(
aligned, asr_segment_map, self.DEFAULT_VOICE_NAME, self.DEFAULT_SPEAKING_RATE
with stage_timer("AudioPipeline.Step7.DurationAlign"):
duration_aligned = self._duration_aligner.align(tts_segments)
with stage_timer("AudioPipeline.Step8.RMS"):
rms_processed = self._rms_processor.apply(
duration_aligned, asr_segment_map, self.DEFAULT_FRAME_MS)
with stage_timer("AudioPipeline.Step9.Mix"):
mixed = self._mixer.mix(rms_processed, separated.background_audio_path)
result = AudioResult(
audio_path=mixed.audio_path,
metadata={
"integrated_lufs": str(mixed.integrated_lufs),
"peak_db": str(mixed.peak_db),
"num_segments": str(len(aligned)),
**audio_input.metadata,
})
out_errors = validate_instance(result, "ADM-15")
if out_errors:
raise AudioPipelineError(f"Invalid AudioResult: {'; '.join(out_errors)}")
return result
@staticmethod
def _build_segment_map(asr_segments: Tuple[ASRSegment, ...]) -> MappingProxyType:
if not isinstance(asr_segments, tuple):
raise SegmentMapError("asr_segments must be a tuple")
if len(asr_segments) == 0:
raise SegmentMapError("asr_segments is empty")
mutable: dict = {}
for seg in asr_segments:
if seg.segment_id in mutable:
raise SegmentMapError(f"Duplicate segment_id: {seg.segment_id}")
mutable[seg.segment_id] = seg
return MappingProxyType(mutable)
