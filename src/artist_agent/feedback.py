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
    score: float
    notes: str


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


def collect(config: ArtistConfig, base: Path) -> list[Feedback]:
    """The feedback this artist learns from, oldest first."""
    out = []
    dirs = [base] if config.learns_from == "self" else config.feedback_dirs
    for path, post in _items(dirs):
        day = _generated_on(post.metadata)
        if config.since and (day is None or day < config.since):
            continue
        if config.learns_from == "self":
            score = post.metadata.get("self_score")
            match = _SELF_NOTES_RE.search(post.content)
        else:
            score = _human_scale(post.metadata.get("human_score"))
            match = _HUMAN_NOTES_RE.search(post.content)
        if score is None:
            continue
        notes = match.group(1).strip() if match else ""
        out.append(
            Feedback(
                stem=path.stem,
                interest=str(post.metadata.get("interest", "")),
                prompt=str(post.metadata.get("prompt", "")),
                score=float(str(score)),
                notes="" if notes.startswith("_not yet") else notes,
            )
        )
    return out


def apply_scores(interests: list[Interest], feedback: list[Feedback]) -> list[Interest]:
    """Each direction's score becomes the mean of its feedback; no feedback means unscored."""
    by_title: dict[str, list[float]] = {}
    for f in feedback:
        by_title.setdefault(f.interest, []).append(f.score)
    return [
        replace(i, score=sum(by_title[i.title]) / len(by_title[i.title]), generations=len(by_title[i.title]))
        if i.title in by_title
        else replace(i, score=0.0, generations=0)
        for i in interests
    ]
