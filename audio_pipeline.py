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
# Schema Registry
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
    if not _is_finite_float(pf.frequency_hz) or not _is_positive(
        pf.frequency_hz
    ):
        errors.append("PitchFrame.frequency_hz must be finite positive float")
    if not _is_range_0_1(pf.confidence):
        errors.append("PitchFrame.confidence must be in [0.0, 1.0]")
    return errors


def _validate_rms_frame(rf: RMSFrame) -> list[str]:
    errors = []
    if not _is_finite_float(rf.time_offset):
        errors.append("RMSFrame.time_offset must be finite float")
    if not _is_finite_float(rf.rms_value) or not _is_non_negative(
        rf.rms_value
    ):
        errors.append("RMSFrame.rms_value must be finite non-negative float")
    return errors


def _validate_segment_prosody(sp: SegmentProsody) -> list[str]:
    errors = []
    if not _is_finite_float(sp.speech_rate) or not _is_positive(
        sp.speech_rate
    ):
        errors.append(
            "SegmentProsody.speech_rate must be finite positive float"
        )
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


_SCHEMA_MAP = {}


def validate_instance(instance, schema_id: str) -> list[str]:
    if schema_id not in _SCHEMA_MAP:
        return [f"Unknown schema_id: {schema_id}"]
    expected_type, validator_fn = _SCHEMA_MAP[schema_id]
    if not isinstance(instance, expected_type):
        return [
            f"Expected {expected_type.__name__}, got {type(instance).__name__}"
        ]
    return validator_fn(instance)


# ============================================================
# Utility WAV
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


def _write_wav(path: str, samples: np.ndarray, sample_rate: int = 44100):
    samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())
