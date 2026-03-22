import time
from pathlib import Path
class RunFolderManager:
"""
Create structured folder for each pipeline run.
"""
def __init__(self, base_dir: str = "outputs"):
self.base_dir = Path(base_dir)
self.run_id = None
self.run_path = None
self.artifacts_dir = None
self.checkpoints_dir = None
self.logs_dir = None
def create(self, run_id: str):
timestamp = time.strftime("%Y%m%d_%H%M%S")
self.run_id = run_id
self.run_path = self.base_dir / f"{run_id}_{timestamp}"
self.artifacts_dir = self.run_path / "artifacts"
self.checkpoints_dir = self.run_path / "checkpoints"
self.logs_dir = self.run_path / "logs"
self.artifacts_dir.mkdir(parents=True, exist_ok=True)
self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
self.logs_dir.mkdir(parents=True, exist_ok=True)
return str(self.run_path)
def artifact_path(self, filename: str):
return str(self.artifacts_dir / filename)
def checkpoint_path(self):
return str(self.checkpoints_dir)
def log_path(self):
return str(self.logs_dir / "pipeline.log")
def debug_report_path(self):
return str(self.run_path / "debug_report.json"
