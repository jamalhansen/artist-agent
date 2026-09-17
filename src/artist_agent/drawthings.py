"""Draw Things inference CLI client.

Shells out to `draw-things-cli`, Draw Things' own headless CLI
(github.com/drawthingsai/draw-things-community/releases), in one of two
modes:

- **Local** (default): draw-things-cli does the Metal/GPU inference itself,
  in the same process launchd spawns for the run. No app or server needed.
- **Remote**: draw-things-cli is just a thin network client; a separately
  running `gRPCServerCLI-macOS` process does the actual Metal work and stays
  alive across runs.

2026-09-13: switched from the GUI app's HTTP API to draw-things-cli local
inference -- removed the "HTTP API Server" toggle problem (genuine
in-memory-only state, confirmed via `defaults read`/config.sqlite3, never
survived an app restart).

2026-09-17: local inference turned out to have its own unattended-only
failure mode -- six straight scheduled (launchd) runs hung with near-zero
CPU for the full timeout, reproduced on demand via `launchctl kickstart -k`
regardless of time of day or display state, while every manual/interactive
run succeeded in ~60-90s. Four other hypotheses (displaysleep, Accessibility
permission, stdin/TTY, LaunchAgent session type) were directly tested and
ruled out. Leading theory, not confirmed: `MTLCreateSystemDefaultDevice()`
is documented as unreliable in non-interactive/daemon process contexts
(Apple Developer Forums, Blender's own bug tracker, GitHub Actions
runner-images all report the same class of failure) -- remote mode moves
that risky one-time Metal initialization into a long-lived server process
instead of repeating it fresh every single scheduled run.
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
    timeout: float = 500.0,
    remote_url: str | None = None,
    remote_port: int = 7859,
    remote_tls: bool = False,
) -> bytes:
    """Generate one image via draw-things-cli and return its raw PNG bytes.

    Default timeout is 500s, not the ~60-90s a real generation normally takes -- the
    2026-09-14 scheduled run hit the previous 300s default on a one-time transient
    slowdown (a manual re-run minutes later took 66s, so it wasn't reproducible), and
    300s left no real headroom under artist-agent-run's own 600s outer watchdog. 500s
    still leaves ~100s for prompt composition + vision critique + file I/O within that
    600s budget, while giving generation itself room to survive a slow morning.

    `model` is a model filename already present in Draw Things' Models directory.

    If `remote_url` is set, generates against a running `gRPCServerCLI-macOS` instead
    of local inference -- see module docstring for why. `--no-download-missing` is a
    local-mode-only flag (model resolution happens server-side in remote mode).
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
        ]
        if remote_url:
            cmd += ["--remote", "--remote-url", remote_url, "--remote-port", str(remote_port)]
            if not remote_tls:
                cmd += ["--no-remote-tls"]
        else:
            cmd += ["--no-download-missing"]
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
