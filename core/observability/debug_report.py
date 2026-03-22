import json
import os
import traceback
import platform
import time
from typing import Optional, Dict, Any

from config_loader import PIPELINE_VERSION


class DebugReportGenerator:
    MAX_LOG_LINES = 100

    @staticmethod
    def generate(
        *,
        run_metadata,
        config,
        metrics,
        error: Exception,
        checkpoint_dir: Optional[str],
        output_path: str,
        artifacts: Optional[Dict[str, str]] = None,
        input_context: Optional[Dict[str, Any]] = None,
        log_file: Optional[str] = None,
    ):
        failed_stage = DebugReportGenerator._detect_failed_stage(metrics)

        report = {
            "pipeline": {
                "run_id": run_metadata.run_id,
                "pipeline_version": run_metadata.pipeline_version
                or PIPELINE_VERSION,
                "timestamp": run_metadata.timestamp,
                "failed_stage": failed_stage,
                "metrics": metrics.get_summary() if metrics else None,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "stack_trace": traceback.format_exc(),
            },
            "config": config.to_dict() if config else {},
            "environment": DebugReportGenerator._environment(),
            "artifacts": DebugReportGenerator._artifact_status(artifacts),
            "checkpoint_snapshot": DebugReportGenerator._checkpoint(
                checkpoint_dir, failed_stage
            ),
            "logs": DebugReportGenerator._logs(log_file),
            "input": input_context or {},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return output_path

    @staticmethod
    def _detect_failed_stage(metrics):
        if not metrics:
            return None

        summary = metrics.get_summary()
        for stage in summary.get("stages", []):
            if stage.get("status") == "failed":
                return stage.get("stage")

        return None

    @staticmethod
    def _environment():
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cwd": os.getcwd(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def _artifact_status(artifacts):
        if not artifacts:
            return {}

        result = {}
        for name, path in artifacts.items():
            exists = os.path.isfile(path)
            result[name] = {
                "path": path,
                "exists": exists,
                "size_bytes": os.path.getsize(path) if exists else None,
            }

        return result

    @staticmethod
    def _checkpoint(checkpoint_dir, stage_name):
        if not checkpoint_dir or not stage_name:
            return None

        try:
            path = os.path.join(checkpoint_dir, f"{stage_name}.json")
            if not os.path.isfile(path):
                return None

            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception:
            return {"error": "checkpoint read failed"}

    @staticmethod
    def _logs(log_file):
        if not log_file or not os.path.isfile(log_file):
            return None

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            return "".join(
                lines[-DebugReportGenerator.MAX_LOG_LINES :]
            )

        except Exception:
            return "log extraction failed"
