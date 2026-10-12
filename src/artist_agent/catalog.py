"""The media an artist may work in: ~/.config/local-first/artist-agent.toml, `[[image_models]]`.

Each entry is one image checkpoint in DRAWTHINGS_MODELS_DIR, with the settings that are known
to work (the defaults), the ranges the artist may explore (`steps_range`, `cfg_range`,
`samplers`, `sizes`), and notes on what the medium does well. The daily studio session picks
a medium and a technique from this; `technique_for` keeps the choice inside the ranges, so a
wild number costs at most one piece at the edge of the range, never a crash.
"""

from dataclasses import dataclass, field

from local_first_common.config import load_config

SETTING_KEYS = ("steps", "cfg", "width", "height", "seed", "sampler", "clip_skip")


@dataclass
class ImageModel:
    file: str
    name: str = ""
    notes: str = ""
    settings: dict = field(default_factory=dict)  # known-good defaults
    steps_range: tuple[int, int] | None = None
    cfg_range: tuple[float, float] | None = None
    samplers: list[str] = field(default_factory=list)  # names from drawthings.SAMPLERS; empty = default only
    sizes: list[str] = field(default_factory=list)  # "WxH"; empty = the default size only
    minutes: float | None = None  # rough render time, so the artist can budget


def _pair(raw, cast) -> tuple | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo, hi = cast(raw[0]), cast(raw[1])
        return (min(lo, hi), max(lo, hi))
    return None


def load_catalog(tool_name: str = "artist-agent") -> list[ImageModel]:
    out = []
    for raw in load_config(tool_name).get("image_models", []) or []:
        if not isinstance(raw, dict) or not raw.get("file"):
            continue
        settings = {k: raw[k] for k in SETTING_KEYS if k in raw}
        minutes = raw.get("minutes")
        out.append(
            ImageModel(
                str(raw["file"]),
                str(raw.get("name", "")),
                str(raw.get("notes", "")),
                settings,
                steps_range=_pair(raw.get("steps_range"), int),
                cfg_range=_pair(raw.get("cfg_range"), float),
                samplers=[str(s) for s in raw.get("samplers", []) or []],
                sizes=[str(s) for s in raw.get("sizes", []) or []],
                minutes=float(minutes) if minutes is not None else None,
            )
        )
    return out


def find(catalog: list[ImageModel], model: str) -> ImageModel | None:
    return next((m for m in catalog if m.file == model), None)


def defaults_for(catalog: list[ImageModel], model: str) -> dict:
    m = find(catalog, model)
    return dict(m.settings) if m else {}


def parse_size(size: str | None) -> tuple[int, int] | None:
    try:
        w, h = str(size).lower().replace(" ", "").split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return None


def _clamp(value, bounds):
    lo, hi = bounds
    return max(lo, min(hi, value))


def technique_for(catalog: list[ImageModel], model: str, requested: dict | None = None) -> dict:
    """The medium's defaults with the artist's requested technique laid over them, kept in range.

    `requested` may carry steps, cfg, sampler, clip_skip and size ("WxH"). A value outside the
    medium's range is clamped; a sampler or size the medium doesn't list is dropped (the default
    stays). A medium with no range listed for a key accepts any value for it. Never sets a seed.
    """
    m = find(catalog, model)
    out = dict(m.settings) if m else {}
    req = dict(requested or {})
    if (steps := req.get("steps")) is not None:
        out["steps"] = _clamp(int(steps), m.steps_range) if m and m.steps_range else int(steps)
    if (cfg := req.get("cfg")) is not None:
        out["cfg"] = _clamp(float(cfg), m.cfg_range) if m and m.cfg_range else float(cfg)
    if (sampler := req.get("sampler")) and (not m or not m.samplers or sampler in m.samplers):
        out["sampler"] = str(sampler)
    if (clip_skip := req.get("clip_skip")) is not None:
        out["clip_skip"] = int(clip_skip)
    if (size := parse_size(req.get("size"))) and (not m or not m.sizes or f"{size[0]}x{size[1]}" in m.sizes):
        out["width"], out["height"] = size
    out.pop("seed", None)
    return out


def describe(catalog: list[ImageModel]) -> str:
    """The catalog as the artist reads it: what each medium is, its known-good technique, the room it has."""
    if not catalog:
        return "(no catalog configured; keep each direction's current model)"
    lines = []
    for m in catalog:
        preset = ", ".join(f"{k} {v}" for k, v in m.settings.items())
        room = []
        if m.steps_range:
            room.append(f"steps {m.steps_range[0]}-{m.steps_range[1]}")
        if m.cfg_range:
            room.append(f"cfg {m.cfg_range[0]:g}-{m.cfg_range[1]:g}")
        if m.samplers:
            room.append("samplers " + " / ".join(m.samplers))
        if m.sizes:
            room.append("sizes " + " / ".join(m.sizes))
        line = f"- {m.file}: {m.name or m.file}. {m.notes}" + (f" Defaults: {preset}." if preset else "")
        if room:
            line += " Room to explore: " + "; ".join(room) + "."
        if m.minutes:
            line += f" About {m.minutes:g} min per image."
        lines.append(line)
    return "\n".join(lines)
