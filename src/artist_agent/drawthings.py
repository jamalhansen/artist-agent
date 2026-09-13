"""Draw Things local-inference CLI client.

Shells out to `draw-things-cli`, Draw Things' own headless local-inference
binary (github.com/drawthingsai/draw-things-community/releases) instead of
the GUI app's HTTP API. This removes a whole class of problem the HTTP
client had: the app's "HTTP API Server" toggle is genuine in-memory-only
state -- confirmed 2026-09-13 via `defaults read`, the sandboxed container's
Preferences plist, and its config.sqlite3, none of which hold a persisted
flag -- so it never survives an app restart, and a scheduled run could
silently need a human at the machine to flip it. draw-things-cli reads the
same on-disk models directly and needs neither the app nor any server
running: confirmed live 2026-09-13 with Draw Things fully quit.
"""
import subprocess
import tempfile
from pathlib import Path


class DrawThingsError(Exception):
    """Raised when draw-things-cli is missing, times out, or a generation fails."""


_SETTING_FLAGS = {
    "steps": "--steps",
    "cfg": "--cfg",
    "width": "--width",
    "height": "--height",
    "seed": "--seed",
}


def generate_image(
    prompt: str,
    model: str,
    settings: dict | None = None,
    cli_path: str = "draw-things-cli",
    timeout: float = 300.0,
) -> bytes:
    """Generate one image via draw-things-cli's local inference and return its raw PNG bytes.

    `model` is a model filename already present in Draw Things' Models directory
    (--no-download-missing keeps a misconfigured model name a fast, clear failure
    rather than a silent multi-GB download).
    """
    settings = settings or {}
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "generated.png"
        cmd = [
            cli_path,
            "generate",
            "--model", model,
            "--prompt", prompt,
            "--output", str(output_path),
            "--no-download-missing",
        ]
        for key, flag in _SETTING_FLAGS.items():
            if key in settings:
                cmd += [flag, str(settings[key])]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError as e:
            raise DrawThingsError(f"draw-things-cli not found (looked for '{cli_path}' on PATH): {e}") from e
        except subprocess.TimeoutExpired as e:
            raise DrawThingsError(f"draw-things-cli generation timed out after {timeout}s") from e

        if result.returncode != 0:
            raise DrawThingsError(f"draw-things-cli generation failed: {result.stderr.strip()}")

        if not output_path.exists():
            raise DrawThingsError("draw-things-cli reported success but wrote no output file")

        return output_path.read_bytes()
