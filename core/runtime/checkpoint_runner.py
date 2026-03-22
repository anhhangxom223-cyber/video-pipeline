from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger("pipeline")


class CheckpointSerializationError(Exception):
    """Raised when a stage result cannot be serialized to checkpoint."""
    pass


class CheckpointDeserializationError(Exception):
    """Raised when a checkpoint cannot be deserialized back to its original type."""
    pass


class CheckpointRunner:
    """
    Execute pipeline stages with checkpoint resume capability.

    Args:
        checkpoint: A StageCheckpoint instance for persistence.
        metrics: Optional PipelineMetrics for observability.
        type_registry: Dict mapping type name strings to classes that implement from_dict().

    Example:
        {"VideoResult": VideoResult, "AudioResult": AudioResult}
    """

    def __init__(
        self,
        checkpoint,
        metrics=None,
        type_registry: Optional[Dict[str, Type]] = None,
    ):
        self.checkpoint = checkpoint
        self.metrics = metrics
        self._type_registry: Dict[str, Type] = type_registry or {}

    def run_stage(
        self, stage_name: str, stage_fn: Callable, *args, **kwargs
    ) -> Any:
        """
        Execute a stage function, or resume from checkpoint if one exists.

        Returns the original typed result on both fresh execution and resume.
        """
        # ---- Resume path ----
        if self.checkpoint.exists(stage_name):
            logger.info("[RESUME] Loading checkpoint for stage: %s", stage_name)
            return self._load(stage_name)

        # ---- Fresh execution path ----
        metric = None
        if self.metrics:
            metric = self.metrics.begin_stage(stage_name)

        try:
            logger.info("[RUN] Stage: %s", stage_name)
            result = stage_fn(*args, **kwargs)

            if self.metrics:
                self.metrics.end_stage(metric, "success")

            self._save(stage_name, result)
            return result

        except Exception as e:
            if self.metrics:
                self.metrics.end_stage(metric, "failed", str(e))
            raise

    def _save(self, stage_name: str, result: Any) -> None:
        """
        Serialize a stage result into a typed checkpoint envelope.

        Envelope format:
            {"__type__": "ClassName", "data": result.to_dict()}

        Raises CheckpointSerializationError if the result does not implement to_dict().
        """
        type_name = type(result).__name__

        if not hasattr(result, "to_dict"):
            raise CheckpointSerializationError(
                f"Stage '{stage_name}': result type '{type_name}' does not implement to_dict(). "
                f"All stage results must implement to_dict()/from_dict() for checkpoint support."
            )

        try:
            data = result.to_dict()
        except Exception as e:
            raise CheckpointSerializationError(
                f"Stage '{stage_name}': to_dict() failed on '{type_name}': {e}"
            ) from e

        envelope = {
            "__type__": type_name,
            "data": data,
        }

        try:
            self.checkpoint.save(stage_name, envelope)
        except Exception as e:
            raise CheckpointSerializationError(
                f"Stage '{stage_name}': checkpoint.save() failed: {e}"
            ) from e

        logger.info(
            "[CHECKPOINT] Saved stage '%s' as type '%s'", stage_name, type_name
        )

    def _load(self, stage_name: str) -> Any:
        """
        Deserialize a typed checkpoint envelope back to its original type.

        If the envelope contains a __type__ field and the type is in the registry,
        calls Type.from_dict(data) to reconstruct the original object.

        Falls back to returning the raw dict if:
        - No __type__ field (legacy checkpoint format)
        - Type not found in registry
        """
        raw = self.checkpoint.load(stage_name)

        if raw is None:
            raise CheckpointDeserializationError(
                f"Stage '{stage_name}': checkpoint file exists but load() returned None"
            )

        # ---- Legacy checkpoint (no envelope) ----
        if not isinstance(raw, dict) or "__type__" not in raw:
            logger.warning(
                "[CHECKPOINT] Stage '%s': no __type__ envelope found, returning raw dict. "
                "This checkpoint may have been created by an older version.",
                stage_name,
            )
            return raw

        # ---- Typed envelope ----
        type_name = raw["__type__"]
        data = raw.get("data", {})

        if type_name not in self._type_registry:
            logger.warning(
                "[CHECKPOINT] Stage '%s': type '%s' not in registry (known: %s). "
                "Returning raw data dict.",
                stage_name,
                type_name,
                list(self._type_registry.keys()),
            )
            return data

        result_type = self._type_registry[type_name]

        if not hasattr(result_type, "from_dict"):
            raise CheckpointDeserializationError(
                f"Stage '{stage_name}': type '{type_name}' is in registry but "
                f"does not implement from_dict()"
            )

        try:
            result = result_type.from_dict(data)
        except Exception as e:
            raise CheckpointDeserializationError(
                f"Stage '{stage_name}': from_dict() failed for type '{type_name}': {e}"
            ) from e

        logger.info(
            "[CHECKPOINT] Restored stage '%s' as %s", stage_name, type_name
        )
        return result
