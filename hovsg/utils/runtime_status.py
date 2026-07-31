"""Small, dependency-free runtime status printer for pipeline diagnostics."""


def print_status(stage, status="INFO", **details):
    """Print one grep-friendly status line without dumping large objects."""
    rendered_details = " ".join(
        f"{key}={_format_value(value)}" for key, value in details.items()
    )
    message = f"[HOVSG][{stage}][{status}]"
    if rendered_details:
        message = f"{message} {rendered_details}"
    print(message, flush=True)


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    text = str(value)
    return text.replace("\n", "\\n").replace(" ", "_")
