"""The image models an artist may choose from: ~/.config/local-first/artist-agent.toml.

Model variety is per direction, not per run: a direction pins a model (and settings) so its
score stays attributable to one combination. The catalog is what reflection may pick from
when it moves a direction to another model, and each entry's settings are the defaults for
a direction that pins the model without its own.
"""

from dataclasses import dataclass, field

from local_first_common.config import load_config

SETTING_KEYS = ("steps", "cfg", "width", "height", "seed", "sampler", "clip_skip")


@dataclass
class ImageModel:
    file: str
    name: str = ""
    notes: str = ""
    settings: dict = field(default_factory=dict)


def load_catalog(tool_name: str = "artist-agent") -> list[ImageModel]:
    out = []
    for raw in load_config(tool_name).get("image_models", []) or []:
        if not isinstance(raw, dict) or not raw.get("file"):
            continue
        settings = {k: raw[k] for k in SETTING_KEYS if k in raw}
        out.append(ImageModel(str(raw["file"]), str(raw.get("name", "")), str(raw.get("notes", "")), settings))
    return out


def defaults_for(catalog: list[ImageModel], model: str) -> dict:
    return next((dict(m.settings) for m in catalog if m.file == model), {})


def describe(catalog: list[ImageModel]) -> str:
    if not catalog:
        return "(no catalog configured; keep each direction's current model)"
    lines = []
    for m in catalog:
        preset = ", ".join(f"{k} {v}" for k, v in m.settings.items())
        lines.append(f"- {m.file}: {m.name or m.file}. {m.notes}" + (f" Defaults: {preset}." if preset else ""))
    return "\n".join(lines)
