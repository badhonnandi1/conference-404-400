"""FFmpeg and FFprobe environment checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import shutil
import subprocess


class FFmpegToolError(RuntimeError):
    """Base error for FFmpeg or FFprobe discovery failures."""


class FFmpegToolUnavailableError(FFmpegToolError):
    """Raised when FFmpeg or FFprobe cannot be found or executed."""


@dataclass(frozen=True)
class ToolInfo:
    """Information about an FFmpeg-family executable."""

    name: str
    path: str
    version: str
    version_line: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def _parse_version(name: str, first_line: str) -> str:
    match = re.search(rf"{re.escape(name)}\s+version\s+([^\s]+)", first_line, re.IGNORECASE)
    if match:
        return match.group(1)
    return first_line.strip()


def get_tool_info(name: str) -> ToolInfo:
    """Return executable path and version information for a required tool."""

    executable = shutil.which(name)
    if executable is None:
        raise FFmpegToolUnavailableError(
            f"Required tool '{name}' was not found on PATH. "
            "Install FFmpeg and confirm your shell can run this command."
        )

    try:
        result = subprocess.run(
            [executable, "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "no stderr output"
        raise FFmpegToolUnavailableError(
            f"Required tool '{name}' exists at {executable}, but version check failed. "
            f"Check the executable installation. stderr: {stderr}"
        ) from exc

    first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    if not first_line:
        raise FFmpegToolUnavailableError(
            f"Required tool '{name}' exists at {executable}, but returned no version output."
        )

    return ToolInfo(
        name=name,
        path=executable,
        version=_parse_version(name, first_line),
        version_line=first_line,
    )


def check_required_tools() -> dict[str, ToolInfo]:
    """Check FFmpeg and FFprobe availability."""

    return {
        "ffmpeg": get_tool_info("ffmpeg"),
        "ffprobe": get_tool_info("ffprobe"),
    }
