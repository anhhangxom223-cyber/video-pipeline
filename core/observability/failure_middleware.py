from typing import Callable
from .debug_report import DebugReportGenerator


class FailureCaptureMiddleware:
    """
    Middleware to capture pipeline failures and generate debug reports.
    """

    def __init__(
        self,
        run_metadata,
        config,
        metrics,
        run_folder,
    ):
        self.metadata = run_metadata
        self.config = config
        self.metrics = metrics
        self.run_folder = run_folder

    def run(self, pipeline_callable: Callable):
        try:
            return pipeline_callable()
        except Exception as e:
            DebugReportGenerator.generate(
                run_metadata=self.metadata,
                config=self.config,
                metrics=self.metrics,
                error=e,
                checkpoint_dir=self.run_folder.checkpoint_path(),
                output_path=self.run_folder.debug_report_path(),
                log_file=self.run_folder.log_path(),
            )
            raise
