import subprocess
from unittest.mock import patch

import pytest

from artist_agent.drawthings import DrawThingsError, generate_image


class TestGenerateImage:
    def _mock_run(self, returncode=0, stderr="", write_output=True, image_bytes=b"fake-png-bytes"):
        def _run(cmd, capture_output, text, timeout, check):
            if write_output:
                output_path = cmd[cmd.index("--output") + 1]
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
            return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

        return _run

    def test_returns_written_image_bytes(self):
        with patch("artist_agent.drawthings.subprocess.run", side_effect=self._mock_run()):
            result = generate_image("a prompt", "sdxl.ckpt")
        assert result == b"fake-png-bytes"

    def test_includes_model_and_prompt_flags(self):
        with patch("artist_agent.drawthings.subprocess.run", side_effect=self._mock_run()) as mock_run:
            generate_image("a prompt", "sdxl.ckpt")
        cmd = mock_run.call_args.args[0]
        assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "sdxl.ckpt"
        assert "--prompt" in cmd and cmd[cmd.index("--prompt") + 1] == "a prompt"
        assert "--no-download-missing" in cmd

    def test_merges_extra_settings_into_flags(self):
        with patch("artist_agent.drawthings.subprocess.run", side_effect=self._mock_run()) as mock_run:
            generate_image("a prompt", "sdxl.ckpt", settings={"steps": 30})
        cmd = mock_run.call_args.args[0]
        assert "--steps" in cmd and cmd[cmd.index("--steps") + 1] == "30"

    def test_raises_on_missing_cli(self):
        with (
            patch("artist_agent.drawthings.subprocess.run", side_effect=FileNotFoundError("no such file")),
            pytest.raises(DrawThingsError, match="not found"),
        ):
            generate_image("a prompt", "sdxl.ckpt")

    def test_raises_on_timeout(self):
        with (
            patch(
                "artist_agent.drawthings.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="draw-things-cli", timeout=300.0),
            ),
            pytest.raises(DrawThingsError, match="timed out"),
        ):
            generate_image("a prompt", "sdxl.ckpt")

    def test_raises_on_nonzero_exit(self):
        with (
            patch(
                "artist_agent.drawthings.subprocess.run",
                side_effect=self._mock_run(returncode=1, stderr="model not found", write_output=False),
            ),
            pytest.raises(DrawThingsError, match="model not found"),
        ):
            generate_image("a prompt", "sdxl.ckpt")

    def test_raises_when_output_file_missing_despite_success(self):
        with (
            patch("artist_agent.drawthings.subprocess.run", side_effect=self._mock_run(write_output=False)),
            pytest.raises(DrawThingsError, match="wrote no output file"),
        ):
            generate_image("a prompt", "sdxl.ckpt")
