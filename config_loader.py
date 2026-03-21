"""config_loader.py — Configuration, Context, Logging, Observability, Reliability, Versioning.
Video Pipeline v3.2.0 / Audio Pipeline v4.2.1
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional
try: # Preferred import path inside the project package
from dto.face_fusion import FaceFusionConfig
except ImportError: # pragma: no cover
from video_pipeline.dto.face_fusion import FaceFusionConfig
try:
from processing.face_fusion.exceptions import FaceFusionInputError
except ImportError: # pragma: no cover
from video_pipeline.processing.face_fusion.exceptions import FaceFusionInputError
# ============================================================
# Logging
# ============================================================
logger = logging.getLogger("pipeline")
def setup_logging(level: str = "INFO") -> None:
logging.basicConfig(
level=getattr(logging, level.upper(), logging.INFO),
format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
)
@contextmanager
def stage_timer(stage_name: str) -> Generator[dict, None, None]:
metrics = {"stage": stage_name, "start_time": time.monotonic(), "elapsed_ms": 0.0}
logger.info("STAGE_START: %s", stage_name)
try:
yield metrics
finally:
elapsed = time.monotonic() - metrics["start_time"]
metrics["elapsed_ms"] = round(elapsed * 1000, 2)
logger.info("STAGE_END: %s | elapsed=%.2fms", stage_name, metrics["elapsed_ms"])
# ============================================================
# DTO: ResolvedConfig
# ============================================================
@dataclass(frozen=True, init=False)
class ResolvedConfig:
configuration_parameters: Dict[str, str]
def __init__(self, configuration_parameters: Dict[str, str]):
if not isinstance(configuration_parameters, dict):
raise TypeError("configuration_parameters must be dict")
normalized: Dict[str, str] = {}
for k, v in configuration_parameters.items():
if not isinstance(k, str) or not isinstance(v, str):
raise TypeError("ResolvedConfig must be Dict[str, str]")
normalized[k] = v
object.__setattr__(self, "configuration_parameters", dict(normalized))
@property
def values(self) -> Dict[str, str]:
return self.configuration_parameters
def to_dict(self) -> dict:
return dict(sorted(self.configuration_parameters.items()))
@classmethod
def from_dict(cls, data: dict) -> "ResolvedConfig":
if not isinstance(data, dict):
raise TypeError("data must be dict")
return cls(configuration_parameters={str(k): str(v) for k, v in data.items()})
# ============================================================
# PipelineContext
# ============================================================
@dataclass(frozen=True)
class PipelineContext:
config: ResolvedConfig
global_color_mapping: Dict[str, Any] = field(default_factory=dict)
scene_mapping_cache: Dict[str, Any] = field(default_factory=dict)
translation_cache: Dict[str, str] = field(default_factory=dict)
def __post_init__(self) -> None:
if not isinstance(self.config, ResolvedConfig):
raise TypeError("config must be ResolvedConfig")
object.__setattr__(self, "global_color_mapping", dict(self.global_color_mapping))
object.__setattr__(self, "scene_mapping_cache", dict(self.scene_mapping_cache))
object.__setattr__(self, "translation_cache", dict(self.translation_cache))
if not all(isinstance(k, str) for k in self.translation_cache.keys()):
raise TypeError("translation_cache keys must be strings")
if not all(isinstance(k, str) for k in self.scene_mapping_cache.keys()):
raise TypeError("scene_mapping_cache keys must be strings")
if not all(isinstance(k, str) for k in self.global_color_mapping.keys()):
raise TypeError("global_color_mapping keys must be strings")
class PipelineContextFactory:
@staticmethod
def create(resolved_config: ResolvedConfig) -> PipelineContext:
if not isinstance(resolved_config, ResolvedConfig):
raise TypeError("resolved_config must be ResolvedConfig")
return PipelineContext(config=resolved_config)
# ============================================================
# ConfigLoader (M1)
# ============================================================
class ConfigLoader:
_DEFAULT_CONFIG: Dict[str, str] = {
"input_path": "",
"output_path": "",
"mapping_config": "",
"target_language": "en",
"ocr_language": "en",
"engine_mode": "HSV",
"enable_video": "true",
"enable_audio": "true",
"face_fusion_enabled": "false",
"face_fusion_mode": "single",
"face_fusion_reference_face_path": "reference_face.png",
"face_fusion_strength": "1.0",
"face_fusion_min_confidence": "0.5",
"face_fusion_preserve_background": "true",
}
_FACE_FUSION_ALLOWED_MODES = {"single", "first", "top1", "one", "multi", "all"}
_BOOL_TRUE = {"true", "1", "yes", "y", "on"}
_BOOL_FALSE = {"false", "0", "no", "n", "off"}
@staticmethod
def _validate_key_value(key: str, value: str) -> None:
if not isinstance(key, str):
raise TypeError("configuration keys must be strings")
if key == "":
raise ValueError("configuration key must not be empty")
if not isinstance(value, str):
raise TypeError("configuration values must be strings")
@classmethod
def _normalize_config_mapping(cls, data: Dict[str, Any]) -> Dict[str, str]:
if not isinstance(data, dict):
raise TypeError("configuration must be Dict[str, Any]")
normalized: Dict[str, str] = {}
for key, value in data.items():
cls._validate_key_value(str(key), str(value))
normalized[str(key)] = str(value)
return normalized
@classmethod
def _parse_bool(cls, value: str, field_name: str) -> bool:
normalized = value.strip().lower()
if normalized in cls._BOOL_TRUE:
return True
if normalized in cls._BOOL_FALSE:
return False
raise ValueError(f"{field_name} must be a boolean string")
@classmethod
try:
def _parse_float_range(cls, value: str, field_name: str) -> float:
parsed = float(value)
except (TypeError, ValueError) as exc:
raise ValueError(f"{field_name} must be a number") from exc
if not (0.0 <= parsed <= 1.0):
raise ValueError(f"{field_name} must be within [0, 1]")
return parsed
@classmethod
def _validate_face_fusion_config_values(cls, data: Dict[str, str]) -> None:
cls._parse_bool(data["face_fusion_enabled"], "face_fusion_enabled")
cls._parse_bool(data["face_fusion_preserve_background"], "face_fusion_preserve_background")
mode = data["face_fusion_mode"].strip().lower()
if mode not in cls._FACE_FUSION_ALLOWED_MODES:
raise FaceFusionInputError(f"invalid face_fusion_mode: {data['face_fusion_mode']}")
cls._parse_float_range(data["face_fusion_strength"], "face_fusion_strength")
cls._parse_float_range(data["face_fusion_min_confidence"], "face_fusion_min_confidence")
if not isinstance(data["face_fusion_reference_face_path"], str):
raise FaceFusionInputError("face_fusion_reference_face_path must be a string")
@classmethod
def load_json(cls, path: str) -> Dict[str, str]:
file_path = Path(path)
if not file_path.exists():
raise FileNotFoundError(path)
if not file_path.is_file():
raise ValueError("configuration path must be a file")
with file_path.open("r", encoding="utf-8") as f:
data = json.load(f)
if not isinstance(data, dict):
raise TypeError("JSON configuration must be an object")
validated: Dict[str, str] = {}
for key, value in data.items():
cls._validate_key_value(str(key), str(value))
validated[str(key)] = str(value)
return validated
return validated
@classmethod
def resolve(
cls,
cli_args: Optional[Dict[str, Any]] = None,
json_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
"""
Frozen IF-02-compatible resolution contract:
merge defaults, then JSON config, then CLI args, and return Dict[str, str].
"""
resolved: Dict[str, str] = dict(cls._DEFAULT_CONFIG)
if json_config is not None:
resolved.update(cls._normalize_config_mapping(json_config))
if cli_args is not None:
resolved.update(cls._normalize_config_mapping(cli_args))
for key, value in resolved.items():
cls._validate_key_value(key, value)
cls._validate_face_fusion_config_values(resolved)
return dict(sorted(resolved.items()))
@classmethod
def resolve_from_paths(
cls,
json_path: Optional[str] = None,
cli_overrides: Optional[Dict[str, Any]] = None,
) -> ResolvedConfig:
"""
Backward-compatible helper for path-based configuration loading.
"""
json_config = cls.load_json(json_path) if json_path is not None else None
resolved = cls.resolve(cli_args=cli_overrides, json_config=json_config)
return ResolvedConfig.from_dict(resolved)
@classmethod
def build_face_fusion_config(
cls,
resolved_config: ResolvedConfig | Dict[str, str],
) -> FaceFusionConfig:
"""
Convert resolved string configuration into the frozen FaceFusionConfig DTO.
"""
if isinstance(resolved_config, ResolvedConfig):
data = resolved_config.to_dict()
elif isinstance(resolved_config, dict):
data = cls._normalize_config_mapping(resolved_config)
else:
raise TypeError("resolved_config must be ResolvedConfig or Dict[str, str]")
cls._validate_face_fusion_config_values(data)
reference_face_path = data["face_fusion_reference_face_path"].strip()
if not reference_face_path:
raise FaceFusionInputError("face_fusion_reference_face_path must be non-empty")
return FaceFusionConfig(
enabled=cls._parse_bool(data["face_fusion_enabled"], "face_fusion_enabled"),
mode=data["face_fusion_mode"].strip().lower(),
reference_face_path=reference_face_path,
strength=cls._parse_float_range(data["face_fusion_strength"], "face_fusion_strength"),
min_confidence=cls._parse_float_range(data["face_fusion_min_confidence"],
"face_fusion_min_confidence"),
preserve_background=cls._parse_bool(
data["face_fusion_preserve_background"],
"face_fusion_preserve_background",
),
)
# ============================================================
# Observability
# ============================================================
@dataclass
class StageMetric:
stage_name: str
start_time: float
end_time: float = 0.0
elapsed_ms: float = 0.0
status: str = "pending"
error: Optional[str] = None
def complete(self, status: str = "success", error: Optional[str] = None) -> None:
self.end_time = time.monotonic()
self.elapsed_ms = (self.end_time - self.start_time) * 1000
self.status = status
self.error = error
def to_dict(self) -> dict:
return {
"stage": self.stage_name,
"elapsed_ms": round(self.elapsed_ms, 2),
"status": self.status,
"error": self.error,
}
class PipelineMetrics:
def __init__(self, run_id: str):
self.run_id = run_id
self._stages: List[StageMetric] = []
self._start_time = time.monotonic()
def begin_stage(self, stage_name: str) -> StageMetric:
metric = StageMetric(stage_name=stage_name, start_time=time.monotonic())
self._stages.append(metric)
logger.info("[%s] STAGE_START: %s", self.run_id, stage_name)
return metric
def end_stage(
self,
metric: StageMetric,
status: str = "success",
error: Optional[str] = None,
) -> None:
metric.complete(status=status, error=error)
logger.info(
"[%s] STAGE_END: %s | elapsed=%.2fms | status=%s",
self.run_id,
metric.stage_name,
metric.elapsed_ms,
status,
)
def get_summary(self) -> dict:
total_elapsed = (time.monotonic() - self._start_time) * 1000
return {
"run_id": self.run_id,
"total_elapsed_ms": round(total_elapsed, 2),
"stages": [s.to_dict() for s in self._stages],
"total_stages": len(self._stages),
"passed": sum(1 for s in self._stages if s.status == "success"),
"failed": sum(1 for s in self._stages if s.status == "failed"),
}
def to_json(self) -> str:
return json.dumps(self.get_summary(), indent=2)
# ============================================================
# Reliability
# ============================================================
class StageCheckpoint:
def __init__(self, checkpoint_dir: str):
self._dir = checkpoint_dir
os.makedirs(self._dir, exist_ok=True)
def save(self, stage_name: str, data: dict) -> str:
path = os.path.join(self._dir, f"{stage_name}.json")
with open(path, "w", encoding="utf-8") as f:
json.dump(data, f, ensure_ascii=False, indent=2)
logger.info("Checkpoint saved: %s", path)
return path
def load(self, stage_name: str) -> Optional[dict]:
path = os.path.join(self._dir, f"{stage_name}.json")
if not os.path.isfile(path):
return None
with open(path, "r", encoding="utf-8") as f:
return json.load(f)
def exists(self, stage_name: str) -> bool:
return os.path.isfile(os.path.join(self._dir, f"{stage_name}.json"))
def clear(self) -> None:
for filename in os.listdir(self._dir):
if filename.endswith(".json"):
os.remove(os.path.join(self._dir, filename))
def retry_with_backoff(
max_retries: int = 3,
base_delay: float = 0.1,
max_delay: float = 5.0,
exceptions: tuple = (Exception,),
):
def decorator(func: Callable) -> Callable:
@wraps(func)
def wrapper(*args, **kwargs):
last_exc = None
for attempt in range(max_retries + 1):
try:
return func(*args, **kwargs)
except exceptions as e:
last_exc = e
if attempt < max_retries:
delay = min(base_delay * (2 ** attempt), max_delay)
logger.warning(
"Retry %d/%d for %s after error: %s (delay: %.2fs)",
attempt + 1,
max_retries,
func.__name__,
e,
delay,
time.sleep(delay)
)
raise last_exc
return wrapper
return decorator
class StageIsolation:
@staticmethod
def run_stage(stage_name: str, func: Callable, *args, **kwargs) -> Any:
logger.info("Stage %s starting", stage_name)
start = time.monotonic()
try:
result = func(*args, **kwargs)
elapsed = (time.monotonic() - start) * 1000
logger.info("Stage %s completed in %.2fms", stage_name, elapsed)
return result
except Exception as e:
elapsed = (time.monotonic() - start) * 1000
logger.error("Stage %s FAILED after %.2fms: %s", stage_name, elapsed, e)
raise
# ============================================================
# Versioning
# ============================================================
VIDEO_PIPELINE_VERSION = "3.2.0"
AUDIO_PIPELINE_VERSION = "4.2.1"
PIPELINE_VERSION = VIDEO_PIPELINE_VERSION
@dataclass(frozen=True)
class RunMetadata:
run_id: str
pipeline_version: str
config_hash: str
timestamp: str
input_hash: str = ""
def to_dict(self) -> dict:
return asdict(self)
def generate_run_id(config_values: dict) -> str:
config_str = json.dumps(config_values, sort_keys=True, ensure_ascii=False)
h = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:12]
return f"run_{h}"
def compute_config_hash(config_values: dict) -> str:
config_str = json.dumps(config_values, sort_keys=True, ensure_ascii=False)
return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
def compute_file_hash(path: str) -> str:
if not os.path.isfile(path):
return ""
h = hashlib.sha256()
with open(path, "rb") as f:
for chunk in iter(lambda: f.read(8192), b""):
h.update(chunk)
return h.hexdigest()[:16]
def create_run_metadata(config_values: dict, input_path: str = "") -> RunMetadata:
run_id = generate_run_id(config_values)
config_hash = compute_config_hash(config_values)
input_hash = compute_file_hash(input_path) if input_path else ""
timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
return RunMetadata(
run_id=run_id,
pipeline_version=PIPELINE_VERSION,
config_hash=config_hash,
timestamp=timestamp,
input_hash=input_hash,
)
def save_run_metadata(metadata: RunMetadata, output_dir: str) -> str:
os.makedirs(output_dir, exist_ok=True)
path = os.path.join(output_dir, "metadata.json")
with open(path, "w", encoding="utf-8") as f:
json.dump(metadata.to_dict(), f, indent=2)
return path
