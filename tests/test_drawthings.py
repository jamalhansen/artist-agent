import base64
from unittest.mock import MagicMock, patch

import pytest
import requests

from artist_agent.drawthings import DrawThingsError, generate_image, is_reachable


class TestIsReachable:
    def test_true_on_200(self):
        with patch("artist_agent.drawthings.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert is_reachable("http://127.0.0.1:7860") is True

    def test_false_on_non_200(self):
        with patch("artist_agent.drawthings.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=500)
            assert is_reachable("http://127.0.0.1:7860") is False

    def test_false_on_connection_error(self):
        with patch("artist_agent.drawthings.requests.get", side_effect=requests.ConnectionError):
            assert is_reachable("http://127.0.0.1:7860") is False


class TestGenerateImage:
    def _mock_response(self, image_bytes: bytes):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"images": [base64.b64encode(image_bytes).decode()]}
        return resp

    def test_returns_decoded_image_bytes(self):
        with patch("artist_agent.drawthings.requests.post") as mock_post:
            mock_post.return_value = self._mock_response(b"fake-png-bytes")
            result = generate_image("a prompt", "http://127.0.0.1:7860", "sdxl.ckpt")
        assert result == b"fake-png-bytes"

    def test_includes_required_model_field(self):
        with patch("artist_agent.drawthings.requests.post") as mock_post:
            mock_post.return_value = self._mock_response(b"x")
            generate_image("a prompt", "http://127.0.0.1:7860", "sdxl.ckpt")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "sdxl.ckpt"
        assert payload["prompt"] == "a prompt"

    def test_merges_extra_settings_into_payload(self):
        with patch("artist_agent.drawthings.requests.post") as mock_post:
            mock_post.return_value = self._mock_response(b"x")
            generate_image("a prompt", "http://127.0.0.1:7860", "sdxl.ckpt", settings={"steps": 30})
        payload = mock_post.call_args.kwargs["json"]
        assert payload["steps"] == 30

    def test_raises_on_request_failure(self):
        with (
            patch("artist_agent.drawthings.requests.post", side_effect=requests.ConnectionError("down")),
            pytest.raises(DrawThingsError, match="generation request failed"),
        ):
            generate_image("a prompt", "http://127.0.0.1:7860", "sdxl.ckpt")

    def test_raises_on_empty_images_list(self):
        with patch("artist_agent.drawthings.requests.post") as mock_post:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"images": []}
            mock_post.return_value = resp
            with pytest.raises(DrawThingsError, match="no images"):
                generate_image("a prompt", "http://127.0.0.1:7860", "sdxl.ckpt")

    def test_raises_on_undecodable_image(self):
        with patch("artist_agent.drawthings.requests.post") as mock_post:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"images": ["not valid base64!!!"]}
            mock_post.return_value = resp
            with pytest.raises(DrawThingsError, match="undecodable"):
                generate_image("a prompt", "http://127.0.0.1:7860", "sdxl.ckpt")
