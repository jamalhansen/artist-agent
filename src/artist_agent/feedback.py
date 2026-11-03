"""Who an artist learns from, and the feedback it learns from.

The 2026-10-01 experiment runs two artists from the same code in different folders.
A `self` artist learns only from its own vision critique; a `jamal` artist learns only
from Jamal's star ratings -- of its own pieces and of the other artist's, since he
rates both blind anyway. The self artist never reads a human rating, so it stays a
clean control for "can an agent teach itself to make interesting art."
"""

import re
import tomllib
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import frontmatter

from .interests import Interest

_SELF_NOTES_RE = re.compile(r"^\*\*Self-critique \([^)]*\):\*\*\s*(.*)$", re.MULTILINE)
_HUMAN_NOTES_RE = re.compile(r"^\*\*Human notes:\*\*\s*(.*)$", re.MULTILINE)
LEARNS_FROM = ("self", "jamal")


@dataclass
class ArtistConfig:
    name: str
    learns_from: str = "self"
    feedback_dirs: list[Path] = field(default_factory=list)
    since: date | None = None


def config_path(base: Path) -> Path:
    return base / "_state" / "artist.toml"


def load_config(base: Path) -> ArtistConfig:
    """`_state/artist.toml`; a folder without one is a self-taught artist named after the folder."""
    path = config_path(base)
    if not path.exists():
        return ArtistConfig(name=base.name, feedback_dirs=[base])
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    learns_from = raw.get("learns_from", "self")
    if learns_from not in LEARNS_FROM:
        raise ValueError(f"{path}: learns_from must be one of {LEARNS_FROM}, not {learns_from!r}")
    dirs = [Path(d).expanduser() for d in raw.get("feedback_dirs", [])] or [base]
    since = raw.get("since")
    return ArtistConfig(
        name=raw.get("name", base.name),
        learns_from=learns_from,
        feedback_dirs=dirs,
        since=since if isinstance(since, date) else (date.fromisoformat(since) if since else None),
    )


@dataclass
class Feedback:
    stem: str
    interest: str
    prompt: str
    score: float | None  # None only in recent_work(): the piece is not yet rated
    notes: str
    model: str | None = None  # the image checkpoint, from the item's `models.image`
    settings: dict = field(default_factory=dict)  # what was passed to draw-things-cli
    intention: dict = field(default_factory=dict)  # the studio session's decisions, when the piece has them
    lesson: str = ""  # the critic's lesson line
    day: date | None = None


def _human_scale(value) -> float | None:
    """Scores typed by hand before `calib art rate` existed were out of 10."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v / 10 if v > 1 else v


def _generated_on(meta: dict) -> date | None:
    try:
        return date.fromisoformat(str(meta.get("generated_at", ""))[:10])
    except ValueError:
        return None


def _items(dirs: list[Path]):
    for d in dirs:
        for path in sorted((d / "items").glob("*.md")):
            try:
                yield path, frontmatter.load(path)
            except Exception:  # noqa: BLE001, S112 - one malformed note shouldn't stop learning from the rest
                continue


def _read(path: Path, post, learns_from: str) -> "Feedback":
    meta = post.metadata
    if learns_from == "self":
        score = meta.get("self_score")
        match = _SELF_NOTES_RE.search(post.content)
    else:
        score = _human_scale(meta.get("human_score"))
        match = _HUMAN_NOTES_RE.search(post.content)
    notes = match.group(1).strip() if match else ""
    models = meta.get("models") or {}
    settings = meta.get("settings") or {}
    intention = meta.get("intention") or {}
    return Feedback(
        stem=path.stem,
        interest=str(meta.get("interest", "")),
        prompt=str(meta.get("prompt", "")),
        score=float(str(score)) if score is not None else None,
        notes="" if notes.startswith("_not yet") else notes,
        model=str(models["image"]) if isinstance(models, dict) and models.get("image") else None,
        settings=dict(settings) if isinstance(settings, dict) else {},
        intention=dict(intention) if isinstance(intention, dict) else {},
        lesson=str(meta.get("lesson") or ""),
        day=_generated_on(meta),
    )


def collect(config: ArtistConfig, base: Path) -> list[Feedback]:
    """The feedback this artist learns from, oldest first: only pieces that have a score."""
    return [f for f in recent_work(config, base) if f.score is not None]


def recent_work(config: ArtistConfig, base: Path, limit: int | None = None) -> list[Feedback]:
    """What the artist has made and seen lately, rated or not, oldest first.

    The studio session reads this: an unrated piece still tells the artist what it tried
    yesterday, so it doesn't repeat itself while waiting for a rating. The score and notes
    come from the artist's own feedback source, so the mentored artist never sees a critique.
    """
    out = []
    dirs = [base] if config.learns_from == "self" else config.feedback_dirs
    for path, post in _items(dirs):
        day = _generated_on(post.metadata)
        if config.since and (day is None or day < config.since):
            continue
        out.append(_read(path, post, config.learns_from))
    out.sort(key=lambda f: (f.day or date.min, f.stem))
    return out[-limit:] if limit else out


@dataclass
class ModelRecord:
    """How one direction has done on one image model, over every piece of feedback."""

    mean: float
    count: int
    last_settings: dict = field(default_factory=dict)


def track_record(feedback: list[Feedback]) -> dict[str, dict[str, ModelRecord]]:
    """{direction title: {model file: record}}; pieces with no recorded model are skipped.

    This is what the daily model draw and the Sunday reflection both read: the whole
    history, not just the last week, so "what does well for what" survives reflection
    consuming its feedback.
    """
    sums: dict[str, dict[str, list[float]]] = {}
    last: dict[str, dict[str, dict]] = {}
    for f in feedback:  # oldest first, so the last write wins for last_settings
        if not f.model or f.score is None:
            continue
        sums.setdefault(f.interest, {}).setdefault(f.model, []).append(f.score)
        last.setdefault(f.interest, {})[f.model] = {k: v for k, v in f.settings.items() if k != "seed"}
    return {
        title: {
            model: ModelRecord(sum(scores) / len(scores), len(scores), last[title][model])
            for model, scores in by_model.items()
        }
        for title, by_model in sums.items()
    }


def apply_scores(interests: list[Interest], feedback: list[Feedback]) -> list[Interest]:
    """Each direction's score becomes the mean of its feedback; no feedback means unscored."""
    by_title: dict[str, list[float]] = {}
    for f in feedback:
        if f.score is None:
            continue
        by_title.setdefault(f.interest, []).append(f.score)
    return [
        replace(i, score=sum(by_title[i.title]) / len(by_title[i.title]), generations=len(by_title[i.title]))
        if i.title in by_title
        else replace(i, score=0.0, generations=0)
        for i in interests
    ]
