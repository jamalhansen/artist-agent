"""Draw Things HTTP API client.

Endpoints and required fields verified against the real app (Draw Things must be
switched to HTTP mode in Settings > Advanced -- that toggle does not survive an
app restart, which is exactly what the health check below exists to catch before
wasting a generation attempt on a dead connection).
"""
import base64

import requests


class DrawThingsError(Exception):
    """Raised when Draw Things is unreachable or a generation request fails."""


def is_reachable(base_url: str, timeout: float = 3.0) -> bool:
    """True if Draw Things' HTTP API is up and responding right now."""
    try:
        response = requests.get(base_url + "/", timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code == 200


def generate_image(
    prompt: str,
    base_url: str,
    model: str,
    settings: dict | None = None,
    timeout: float = 180.0,
) -> bytes:
    """Generate one image via txt2img and return its raw PNG bytes.

    `model` is required -- Draw Things returns 422 "Missing 'model' parameter"
    without it, unlike Automatic1111-style servers that default to whatever's
    loaded.
    """
    payload = {"prompt": prompt, "model": model, **(settings or {})}
    try:
        response = requests.post(f"{base_url}/sdapi/v1/txt2img", json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise DrawThingsError(f"Draw Things generation request failed: {e}") from e

    images = response.json().get("images") or []
    if not images:
        raise DrawThingsError("Draw Things returned no images")

    try:
        return base64.b64decode(images[0])
    except (ValueError, TypeError) as e:
        raise DrawThingsError(f"Draw Things returned an undecodable image: {e}") from e
