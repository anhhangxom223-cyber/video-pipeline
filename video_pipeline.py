"""
video_pipeline.py — Video Pipeline v3.2.0
DTOs (M2), ROI (M4), OCR (M5), SubtitleTracker (M6), TranslationBuffer (M7),
TranslationCache/Adapter (M8/M9), RecolorEngine (M11), FaceFusionEngine (M12),
SubtitleRenderer (M13), FPSNormalizer (M14), ReaderThread (M15),
ProcessorThread (M16), EncoderThread (M17), BackboneExecutionCore (M18),
VideoPipeline (M19).
"""
from __future__ import annotations
import hashlib
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import numpy as np
from config_loader import stage_timer
try:
from dto.face_fusion import FaceFusionConfig
from processing.face_fusion.detector import FaceDetector
from processing.face_fusion.engine import FaceFusionEngine
from processing.face_fusion.exceptions import FaceFusionInputError
except ImportError: # pragma: no cover
from video_pipeline.dto.face_fusion import FaceFusionConfig
from video_pipeline.processing.face_fusion.detector import FaceDetector
from video_pipeline.processing.face_fusion.engine import FaceFusionEngine
from video_pipeline.processing.face_fusion.exceptions import FaceFusionInputError
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
ENGINE_MODES = {"HSV", "LAB"}
FACE_FUSION_ALLOWED_MODES = {"single", "first", "top1", "one", "multi", "all"}
def _validate_tuple_range(value: Tuple[int, int], name: str) -> None:
if not isinstance(value, tuple) or len(value) != 2:
raise TypeError(f"{name} must be tuple[int, int]")
a, b = value
if not isinstance(a, int) or not isinstance(b, int):
raise TypeError(f"{name} values must be int")
if a < 0 or b < 0 or a > 255 or b > 255:
raise ValueError(f"{name} values must be within range 0-255")
if a > b:
raise ValueError(f"{name} start must be <= end")
def _validate_face_fusion_config(cfg: FaceFusionConfig) -> None:
if not isinstance(cfg, FaceFusionConfig):
raise FaceFusionInputError("Invalid FaceFusionConfig")
if not isinstance(cfg.enabled, bool):
raise FaceFusionInputError("enabled must be bool")
if not isinstance(cfg.mode, str) or not cfg.mode.strip():
raise FaceFusionInputError("mode must be non-empty string")
mode = cfg.mode.strip().lower()
if mode not in FACE_FUSION_ALLOWED_MODES:
raise FaceFusionInputError(f"invalid mode: {cfg.mode}")
if not isinstance(cfg.reference_face_path, str) or not cfg.reference_face_path.strip():
raise FaceFusionInputError("reference_face_path must be non-empty")
if not isinstance(cfg.strength, (int, float)) or not (0.0 <= float(cfg.strength) <= 1.0):
raise FaceFusionInputError("strength must be within [0,1]")
if not isinstance(cfg.min_confidence, (int, float)) or not (0.0 <= float(cfg.min_confidence) <= 1.0):
raise FaceFusionInputError("min_confidence must be within [0,1]")
if not isinstance(cfg.preserve_background, bool):
raise FaceFusionInputError("preserve_background must be bool")
@dataclass(frozen=True)
class FaceFusionMetrics:
stage_invocations: int = 0
enabled_invocations: int = 0
disabled_invocations: int = 0
passthrough_invocations: int = 0
has_face_invocations: int = 0
no_face_invocations: int = 0
error_count: int = 0
class FaceFusionObserver:
def __init__(self) -> None:
self.metrics = FaceFusionMetrics()
def record_invocation(self, enabled: bool) -> None:
self.metrics.stage_invocations += 1
if enabled:
self.metrics.enabled_invocations += 1
else:
self.metrics.disabled_invocations += 1
def record_passthrough(self) -> None:
self.metrics.passthrough_invocations += 1
def record_face_presence(self, has_face: bool) -> None:
if has_face:
self.metrics.has_face_invocations += 1
else:
self.metrics.no_face_invocations += 1
def record_error(self) -> None:
self.metrics.error_count += 1
@dataclass(frozen=True)
class ROIBox:
x: int
y: int
width: int
height: int
def __post_init__(self) -> None:
if not all(isinstance(v, int) for v in (self.x, self.y, self.width, self.height)):
raise TypeError("ROIBox coordinates must be integers")
if self.x < 0 or self.y < 0:
raise ValueError("ROIBox coordinates must be non-negative")
if self.width <= 0 or self.height <= 0:
raise ValueError("ROIBox width and height must be positive")
def to_dict(self) -> dict:
return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
@classmethod
def from_dict(cls, data: dict) -> "ROIBox":
return cls(x=int(data["x"]), y=int(data["y"]), width=int(data["width"]), height=int(data["height"]))
@dataclass(frozen=True)
class MappingEntry:
source_range: Tuple[int, int]
target_hue_range: Tuple[int, int]
def __post_init__(self) -> None:
_validate_tuple_range(self.source_range, "source_range")
_validate_tuple_range(self.target_hue_range, "target_hue_range")
def to_dict(self) -> dict:
return {
"source_range": list(self.source_range),
"target_hue_range": list(self.target_hue_range),
}
@classmethod
def from_dict(cls, data: dict) -> "MappingEntry":
return cls(
source_range=tuple(data["source_range"]),
target_hue_range=tuple(data["target_hue_range"]),
)
@dataclass(frozen=True)
class MappingConfig:
version: str
engine_mode: str
mappings: Tuple[MappingEntry, ...]
def __post_init__(self) -> None:
if not isinstance(self.version, str) or not SEMVER_PATTERN.match(self.version):
raise ValueError("version must follow semantic version format X.Y.Z")
if self.engine_mode not in ENGINE_MODES:
raise ValueError(f"engine_mode must be one of {ENGINE_MODES}")
if not isinstance(self.mappings, tuple) or len(self.mappings) == 0:
raise ValueError("mappings must be non-empty tuple")
for entry in self.mappings:
if not isinstance(entry, MappingEntry):
raise TypeError("mappings must contain MappingEntry objects")
def to_dict(self) -> dict:
return {"version": self.version, "engine_mode": self.engine_mode, "mappings": [m.to_dict() for m in
self.mappings]}
@classmethod
def from_dict(cls, data: dict) -> "MappingConfig":
mappings = tuple(MappingEntry.from_dict(m) for m in data["mappings"])
return cls(version=data["version"], engine_mode=data["engine_mode"], mappings=mappings)
@dataclass(frozen=True)
class VideoMetadata:
width: int
height: int
fps: float
frame_count: int
def __post_init__(self) -> None:
if not isinstance(self.width, int) or self.width <= 0:
raise ValueError("width must be positive int")
if not isinstance(self.height, int) or self.height <= 0:
raise ValueError("height must be positive int")
if not isinstance(self.fps, (int, float)) or self.fps <= 0:
raise ValueError("fps must be positive number")
if not isinstance(self.frame_count, int) or self.frame_count < 0:
raise ValueError("frame_count must be non-negative int")
def to_dict(self) -> dict:
return {"width": self.width, "height": self.height, "fps": self.fps, "frame_count": self.frame_count}
@classmethod
def from_dict(cls, data: dict) -> "VideoMetadata":
return cls(width=int(data["width"]), height=int(data["height"]), fps=float(data["fps"]),
frame_count=int(data["frame_count"]))
@dataclass(frozen=True)
class VideoResult:
output_path: str
subtitle_path: Optional[str]
metadata: VideoMetadata
def __post_init__(self) -> None:
if not isinstance(self.output_path, str) or not self.output_path:
raise ValueError("output_path must be non-empty string")
if self.subtitle_path is not None and not isinstance(self.subtitle_path, str):
raise TypeError("subtitle_path must be string or None")
if not isinstance(self.metadata, VideoMetadata):
raise TypeError("metadata must be VideoMetadata")
def to_dict(self) -> dict:
return {"output_path": self.output_path, "subtitle_path": self.subtitle_path, "metadata":
self.metadata.to_dict()}
@classmethod
def from_dict(cls, data: dict) -> "VideoResult":
return cls(output_path=data["output_path"], subtitle_path=data.get("subtitle_path"),
metadata=VideoMetadata.from_dict(data["metadata"]))
@dataclass(frozen=True)
class VideoOutputContract:
result: VideoResult
def __post_init__(self) -> None:
if not isinstance(self.result, VideoResult):
raise TypeError("result must be VideoResult")
def to_dict(self) -> dict:
return {"result": self.result.to_dict()}
class ROIDetector:
@staticmethod
def detect(frame: np.ndarray) -> ROIBox:
if not isinstance(frame, np.ndarray):
raise TypeError("frame must be numpy.ndarray")
if frame.size == 0:
raise ValueError("frame must not be empty")
if frame.ndim not in (2, 3):
raise ValueError("frame dimensionality must be 2D or 3D")
height, width = frame.shape[:2]
roi_width = max(1, int(width * 0.8))
roi_height = max(1, int(height * 0.2))
x = max(0, (width - roi_width) // 2)
y = max(0, height - roi_height)
if x + roi_width > width:
roi_width = width - x
if y + roi_height > height:
roi_height = height - y
return ROIBox(x=x, y=y, width=roi_width, height=roi_height)
class OCRModule:
@staticmethod
def extract(frame: np.ndarray, roi: ROIBox) -> list[str]:
if not isinstance(frame, np.ndarray):
raise TypeError("frame must be numpy.ndarray")
if not isinstance(roi, ROIBox):
raise TypeError("roi must be ROIBox")
if frame.size == 0:
return []
if frame.ndim not in (2, 3):
raise ValueError("frame dimensionality must be 2D or 3D")
height, width = frame.shape[:2]
if roi.x + roi.width > width or roi.y + roi.height > height:
raise ValueError("ROI must be inside frame bounds")
cropped = frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
if cropped.size == 0:
return []
checksum = hashlib.sha256(cropped.tobytes()).hexdigest()
return [f"ocr_{checksum[:16]}"]
@dataclass
class _SubtitleBlock:
text: str
start_frame: int
end_frame: int
class SubtitleTracker:
def __init__(self, fps: float = 30.0) -> None:
if not isinstance(fps, (int, float)) or fps <= 0:
raise ValueError("fps must be a positive number")
self._fps = float(fps)
self._blocks: List[_SubtitleBlock] = []
self._current_block: Optional[_SubtitleBlock] = None
self._last_frame_index: Optional[int] = None
def update(self, frame_index: int, ocr_output: list[str]) -> None:
if not isinstance(frame_index, int) or frame_index < 0:
raise ValueError("frame_index must be a non-negative integer")
if not isinstance(ocr_output, list):
raise TypeError("ocr_output must be list[str]")
text = " ".join(ocr_output).strip()
self._last_frame_index = frame_index
if text == "":
if self._current_block is not None:
self._current_block.end_frame = frame_index
self._blocks.append(self._current_block)
self._current_block = None
return
if self._current_block is None:
self._current_block = _SubtitleBlock(text=text, start_frame=frame_index, end_frame=frame_index)
return
if text == self._current_block.text:
self._current_block.end_frame = frame_index
return
self._blocks.append(self._current_block)
self._current_block = _SubtitleBlock(text=text, start_frame=frame_index, end_frame=frame_index)
def finalize(self) -> None:
if self._current_block is not None:
if self._last_frame_index is not None:
self._current_block.end_frame = self._last_frame_index
self._blocks.append(self._current_block)
self._current_block = None
def build_srt(self) -> str:
lines: List[str] = []
for idx, block in enumerate(self._blocks, start=1):
start_ts = self._frame_to_timestamp(block.start_frame)
end_ts = self._frame_to_timestamp(block.end_frame + 1)
lines.append(str(idx))
lines.append(f"{start_ts} --> {end_ts}")
lines.append(block.text)
lines.append("")
return "\n".join(lines).rstrip() + ("\n" if lines else "")
def _frame_to_timestamp(self, frame_index: int) -> str:
total_seconds = frame_index / self._fps
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
milliseconds = int((total_seconds - int(total_seconds)) * 1000)
return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
class TranslationBufferManager:
HARD_CAP: int = 200
def __init__(self, threshold: int = 20) -> None:
if not isinstance(threshold, int) or threshold <= 0:
raise ValueError("threshold must be positive integer")
self._threshold = threshold
self._buffer: List[str] = []
def add(self, block: str) -> Optional[List[str]]:
if not isinstance(block, str):
raise TypeError("block must be string")
if len(self._buffer) >= self.HARD_CAP:
flushed = self.flush()
self._buffer.append(block)
return flushed
self._buffer.append(block)
if len(self._buffer) >= self._threshold:
return self.flush()
return None
def flush(self) -> Optional[List[str]]:
if not self._buffer:
return None
data = list(self._buffer)
self._buffer = []
return data
def finalize(self) -> Optional[List[str]]:
return self.flush()
def size(self) -> int:
return len(self._buffer)
class TranslationCache:
def __init__(self) -> None:
self._store: Dict[str, str] = {}
def get(self, text: str) -> Optional[str]:
if not isinstance(text, str):
raise TypeError("text must be str")
key = self._hash(self._normalize(text))
return self._store.get(key)
def set(self, text: str, translated: str) -> None:
if not isinstance(text, str):
raise TypeError("text must be str")
if not isinstance(translated, str):
raise TypeError("translated must be str")
key = self._hash(self._normalize(text))
self._store[key] = translated
def _normalize(self, text: str) -> str:
return " ".join(text.strip().split())
def _hash(self, text: str) -> str:
return hashlib.sha256(text.encode("utf-8")).hexdigest()
class TranslationAdapter:
def __init__(self, translate_fn: Callable[[str], str], timeout: float = 15.0, max_retries: int = 3) ->
None:
if not callable(translate_fn):
raise TypeError("translate_fn must be callable")
if not isinstance(timeout, (int, float)) or timeout <= 0:
raise ValueError("timeout must be positive")
if not isinstance(max_retries, int) or max_retries < 0:
raise ValueError("max_retries must be >= 0")
self._translate_fn = translate_fn
self._timeout = float(timeout)
self._max_retries = max_retries
def translate_batch(self, texts: list[str]) -> list[str]:
if not isinstance(texts, list):
raise TypeError("texts must be list[str]")
results: List[str] = []
for text in texts:
if not isinstance(text, str):
raise TypeError("texts must contain strings")
try:
translated = self._translate_with_retry(text)
except Exception:
translated = text
results.append(translated)
return results
def _translate_with_retry(self, text: str) -> str:
attempt = 0
while attempt <= self._max_retries:
start = time.monotonic()
try:
result = self._translate_fn(text)
elapsed = time.monotonic() - start
if elapsed > self._timeout:
raise TimeoutError("translation timeout exceeded")
if not isinstance(result, str):
raise TypeError("translation result must be str")
return result
except Exception:
if attempt >= self._max_retries:
return text
attempt += 1
return text
class RecolorEngineAdapter:
def apply(self, frame: np.ndarray, config: MappingConfig) -> np.ndarray:
if not isinstance(frame, np.ndarray):
raise TypeError("frame must be numpy.ndarray")
if frame.size == 0:
raise ValueError("frame must not be empty")
if not isinstance(config, MappingConfig):
raise TypeError("config must be MappingConfig")
output_frame = frame.copy()
if not config.mappings:
return output_frame
mapping: MappingEntry = config.mappings[0]
hue_offset = mapping.target_hue_range[0]
if output_frame.ndim == 2:
channel = output_frame
else:
channel = output_frame[..., 0]
channel = channel.astype(np.int32) + int(hue_offset)
np.clip(channel, 0, 255, out=channel)
if output_frame.ndim == 2:
output_frame = channel.astype(frame.dtype)
else:
output_frame[..., 0] = channel.astype(frame.dtype)
return output_frame
class FaceFusionStageAdapter:
def __init__(self, face_fusion_engine: Optional[FaceFusionEngine] = None, face_fusion_config:
Optional[FaceFusionConfig] = None) -> None:
self._engine = face_fusion_engine if face_fusion_engine is not None else FaceFusionEngine()
self._config: Optional[FaceFusionConfig] = None
self.set_config(face_fusion_config)
def set_config(self, face_fusion_config: Optional[FaceFusionConfig]) -> None:
if face_fusion_config is not None:
_validate_face_fusion_config(face_fusion_config)
self._config = face_fusion_config
def apply(self, frame: np.ndarray) -> np.ndarray:
if self._config is None:
return frame
return self._engine.fuse_face(frame, self._config)
class SubtitleRenderer:
def render(self, frame: np.ndarray, text: str) -> np.ndarray:
if not isinstance(frame, np.ndarray):
raise TypeError("frame must be numpy.ndarray")
if frame.size == 0:
raise ValueError("frame must not be empty")
if not isinstance(text, str):
raise TypeError("text must be string")
output_frame = frame.copy()
if text.strip() == "":
return output_frame
height, width = frame.shape[:2]
overlay_height = max(1, int(height * 0.1))
y_start = max(0, height - overlay_height)
checksum = hashlib.md5(text.encode("utf-8")).digest()
overlay_value = checksum[0] % 255
region = output_frame[y_start:height].astype(np.int32)
region[:] = overlay_value
np.clip(region, 0, 255, out=region)
output_frame[y_start:height] = region.astype(frame.dtype)
return output_frame
class FPSNormalizer:
TARGET_FPS: int = 30
def __init__(self, input_fps: float) -> None:
if not isinstance(input_fps, (int, float)) or input_fps <= 0:
raise ValueError("input_fps must be positive")
self._input_fps = float(input_fps)
def normalize(self, total_frames: int) -> List[int]:
if not isinstance(total_frames, int) or total_frames < 0:
raise ValueError("total_frames must be >= 0")
if total_frames == 0:
return []
if self._input_fps == self.TARGET_FPS:
return list(range(total_frames))
mapping: List[int] = []
if self._input_fps > self.TARGET_FPS:
normalized_count = int(total_frames * self.TARGET_FPS / self._input_fps + 0.5)
for i in range(normalized_count):
original_index = int(i * self._input_fps / self.TARGET_FPS + 0.5)
if original_index >= total_frames:
original_index = total_frames - 1
mapping.append(original_index)
else:
duplication_factor = self.TARGET_FPS / self._input_fps
normalized_count = int(total_frames * duplication_factor + 0.5)
for i in range(normalized_count):
original_index = int(i / duplication_factor)
if original_index >= total_frames:
original_index = total_frames - 1
mapping.append(original_index)
return mapping
class ReaderThread(threading.Thread):
def __init__(self, frame_source: Iterator[np.ndarray], output_queue: queue.Queue, stop_event:
threading.Event, fatal_error_holder: dict, sentinel: object) -> None:
super().__init__(daemon=False)
self._frame_source = frame_source
self._output_queue = output_queue
self._stop_event = stop_event
self._fatal_error_holder = fatal_error_holder
self._sentinel = sentinel
def _enqueue_with_retry(self, item: object) -> None:
while True:
if self._stop_event.is_set() or "error" in self._fatal_error_holder:
return
try:
self._output_queue.put(item, timeout=0.1)
return
except queue.Full:
continue
def run(self) -> None:
try:
for frame in self._frame_source:
if self._stop_event.is_set():
return
if not isinstance(frame, np.ndarray):
raise TypeError("frame must be numpy.ndarray")
self._enqueue_with_retry(frame)
if self._stop_event.is_set() or "error" in self._fatal_error_holder:
return
if not self._stop_event.is_set():
self._enqueue_with_retry(self._sentinel)
except Exception as exc:
if "error" not in self._fatal_error_holder:
self._fatal_error_holder["error"] = exc
self._stop_event.set()
raise
class ProcessorThread(threading.Thread):
def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue, process_fn: Callable[[np.ndarray],
object], stop_event: threading.Event, fatal_error_holder: dict, sentinel: object) -> None:
super().__init__(daemon=False)
self._input_queue = input_queue
self._output_queue = output_queue
self._process_fn = process_fn
self._stop_event = stop_event
self._fatal_error_holder = fatal_error_holder
self._sentinel = sentinel
def _enqueue_with_retry(self, item: object) -> None:
while True:
if self._stop_event.is_set() or "error" in self._fatal_error_holder:
return
try:
self._output_queue.put(item, timeout=0.1)
return
except queue.Full:
continue
def run(self) -> None:
try:
while True:
try:
item = self._input_queue.get(timeout=0.1)
except queue.Empty:
if self._stop_event.is_set() or "error" in self._fatal_error_holder:
return
continue
if item is self._sentinel:
self._enqueue_with_retry(self._sentinel)
return
if self._stop_event.is_set():
return
if not isinstance(item, np.ndarray):
raise TypeError("frame must be numpy.ndarray")
result = self._process_fn(item)
self._enqueue_with_retry(result)
except Exception as exc:
if "error" not in self._fatal_error_holder:
self._fatal_error_holder["error"] = exc
self._stop_event.set()
raise
class EncoderThread(threading.Thread):
def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue, encode_fn: Callable[[np.ndarray],
object], stop_event: threading.Event, fatal_error_holder: dict, sentinel: object) -> None:
super().__init__(daemon=False)
self._input_queue = input_queue
self._output_queue = output_queue
self._encode_fn = encode_fn
self._stop_event = stop_event
self._fatal_error_holder = fatal_error_holder
self._sentinel = sentinel
self._buffer = deque()
def _enqueue_with_retry(self, item: object) -> None:
while True:
if self._stop_event.is_set() or "error" in self._fatal_error_holder:
return
try:
self._output_queue.put(item, timeout=0.1)
return
except queue.Full:
continue
def _flush_buffer(self) -> None:
while self._buffer:
item = self._buffer.popleft()
self._enqueue_with_retry(item)
if "error" in self._fatal_error_holder:
return
def run(self) -> None:
try:
while True:
try:
item = self._input_queue.get(timeout=0.1)
except queue.Empty:
if self._stop_event.is_set() or "error" in self._fatal_error_holder:
return
continue
if item is self._sentinel:
self._flush_buffer()
self._enqueue_with_retry(self._sentinel)
return
if self._stop_event.is_set():
return
if not isinstance(item, np.ndarray):
raise TypeError("frame must be numpy.ndarray")
encoded = self._encode_fn(item)
self._buffer.append(encoded)
except Exception as exc:
if "error" not in self._fatal_error_holder:
self._fatal_error_holder["error"] = exc
self._stop_event.set()
raise
class BackboneExecutionCore:
def __init__(self, frame_source: Iterator[np.ndarray], process_fn: Callable[[np.ndarray], np.ndarray],
encode_fn: Callable[[np.ndarray], object], queue_size: int = 16) -> None:
if not isinstance(queue_size, int) or queue_size <= 0:
raise ValueError("queue_size must be positive integer")
self._stop_event = threading.Event()
self._fatal_error_holder: dict = {}
self._sentinel = object()
self._reader_queue: queue.Queue = queue.Queue(maxsize=queue_size)
self._processor_queue: queue.Queue = queue.Queue(maxsize=queue_size)
self._encoder_queue: queue.Queue = queue.Queue(maxsize=queue_size)
self._reader_thread = ReaderThread(frame_source, self._reader_queue, self._stop_event,
self._fatal_error_holder, self._sentinel)
self._processor_thread = ProcessorThread(self._reader_queue, self._processor_queue, process_fn,
self._stop_event, self._fatal_error_holder, self._sentinel)
self._encoder_thread = EncoderThread(self._processor_queue, self._encoder_queue, encode_fn,
self._stop_event, self._fatal_error_holder, self._sentinel)
def run(self) -> List[object]:
output_frames: List[object] = []
self._encoder_thread.start()
self._processor_thread.start()
self._reader_thread.start()
try:
while True:
if "error" in self._fatal_error_holder:
self._stop_event.set()
break
try:
item = self._encoder_queue.get(timeout=0.1)
except queue.Empty:
if "error" in self._fatal_error_holder:
self._stop_event.set()
break
continue
if item is self._sentinel:
break
output_frames.append(item)
finally:
self._reader_thread.join()
self._processor_thread.join()
self._encoder_thread.join()
if "error" in self._fatal_error_holder:
raise self._fatal_error_holder["error"]
return output_frames
class VideoPipeline:
def __init__(self, roi_detector: ROIDetector, ocr_module: OCRModule, subtitle_tracker: SubtitleTracker,
translation_buffer: TranslationBufferManager, translation_cache: TranslationCache, translation_adapter:
TranslationAdapter, recolor_engine: RecolorEngineAdapter, subtitle_renderer: SubtitleRenderer, fps_normalizer:
FPSNormalizer, face_fusion_engine: Optional[FaceFusionEngine] = None, face_fusion_config:
Optional[FaceFusionConfig] = None) -> None:
self.roi_detector = roi_detector
self.ocr_module = ocr_module
self.subtitle_tracker = subtitle_tracker
self.translation_buffer = translation_buffer
self.translation_cache = translation_cache
self.translation_adapter = translation_adapter
self.recolor_engine = recolor_engine
self.subtitle_renderer = subtitle_renderer
self.fps_normalizer = fps_normalizer
self.face_fusion_engine = face_fusion_engine if face_fusion_engine is not None else FaceFusionEngine()
self.face_fusion_detector = FaceDetector()
self.face_fusion_stage = FaceFusionStageAdapter(face_fusion_engine=self.face_fusion_engine,
face_fusion_config=face_fusion_config)
self.face_fusion_observer = FaceFusionObserver()
self._default_face_fusion_config = face_fusion_config
self._active_face_fusion_config: Optional[FaceFusionConfig] = face_fusion_config
if self._default_face_fusion_config is not None:
_validate_face_fusion_config(self._default_face_fusion_config)
self._frame_index = 0
self._mapping_config: MappingConfig | None = None
def _apply_face_fusion(self, frame: np.ndarray) -> np.ndarray:
if self._active_face_fusion_config is None:
self.face_fusion_observer.record_passthrough()
return frame
self.face_fusion_observer.record_invocation(self._active_face_fusion_config.enabled)
if not self._active_face_fusion_config.enabled:
self.face_fusion_observer.record_passthrough()
return frame
try:
detected_regions = self.face_fusion_detector.detect(frame,
self._active_face_fusion_config.min_confidence)
self.face_fusion_observer.record_face_presence(bool(detected_regions))
return self.face_fusion_stage.apply(frame)
except Exception:
self.face_fusion_observer.record_error()
raise
def _process_frame(self, frame: np.ndarray) -> np.ndarray:
if not isinstance(frame, np.ndarray):
raise TypeError("frame must be numpy.ndarray")
frame_index = self._frame_index
self._frame_index += 1
roi = self.roi_detector.detect(frame)
texts = self.ocr_module.extract(frame, roi)
self.subtitle_tracker.update(frame_index, texts)
translated_texts: List[str] = []
for text in texts:
cached = self.translation_cache.get(text)
if cached is not None:
translated = cached
else:
result = self.translation_adapter.translate_batch([text])
translated = result[0]
self.translation_cache.set(text, translated)
translated_texts.append(translated)
translated_text = " ".join(translated_texts).strip()
if self._mapping_config is not None:
frame = self.recolor_engine.apply(frame, self._mapping_config)
frame = self._apply_face_fusion(frame)
frame = self.subtitle_renderer.render(frame, translated_text)
return frame
def run(self, frame_source: Iterator[np.ndarray], metadata: VideoMetadata, mapping_config: MappingConfig,
output_path: str, face_fusion_config: Optional[FaceFusionConfig] = None) -> VideoResult:
if not isinstance(metadata, VideoMetadata):
raise TypeError("metadata must be VideoMetadata")
if not isinstance(mapping_config, MappingConfig):
raise TypeError("mapping_config must be MappingConfig")
if not isinstance(output_path, str) or not output_path:
raise ValueError("output_path must be non-empty string")
self._frame_index = 0
self._mapping_config = mapping_config
active_config = face_fusion_config if face_fusion_config is not None else
self._default_face_fusion_config
if active_config is not None:
_validate_face_fusion_config(active_config)
self._active_face_fusion_config = active_config
self.face_fusion_stage.set_config(active_config)
with stage_timer("VideoPipeline.backbone"):
backbone = BackboneExecutionCore(frame_source=frame_source, process_fn=self._process_frame,
encode_fn=lambda frame: frame)
encoded_frames = backbone.run()
with stage_timer("VideoPipeline.finalize"):
self.subtitle_tracker.finalize()
srt_text = self.subtitle_tracker.build_srt()
_mapping = self.fps_normalizer.normalize(len(encoded_frames))
_normalized_frames = [encoded_frames[i] for i in _mapping]
subtitle_output_path = output_path + ".srt"
with open(subtitle_output_path, "w", encoding="utf-8") as f:
f.write(srt_text)
return VideoResult(output_path=output_path, subtitle_path=subtitle_output_path, metadata=metadata)
