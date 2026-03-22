import functools


def instrument_stage(metrics, stage_name: str):
    """
    Decorator to automatically instrument stage metrics.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metric = metrics.begin_stage(stage_name)
            try:
                result = func(*args, **kwargs)
                metrics.end_stage(metric, "success")
                return result
            except Exception as e:
                metrics.end_stage(metric, "failed", str(e))
                raise

        return wrapper

    return decorator
