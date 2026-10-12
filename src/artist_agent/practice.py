"""The artist's practice file: `_state/practice.md`, its own account of its taste and style.

Jamal's brief (2026-10-11): an artist develops themes it likes to explore, colors it likes,
media and techniques it likes to use, a style, and then stretches that style by trying new
techniques. The directions in interests.md are the themes; this file is everything else the
artist knows about itself. The daily studio session reads it whole, and the Sunday reflection
rewrites it from the week's work and feedback (previous versions under `_state/history/`).
Plain markdown, so Jamal can read it and write in it like anything else in the folder.
"""

from pathlib import Path

SECTIONS = (
    "Style",
    "Palettes I like",
    "Media and technique",
    "Themes",
    "Stretching toward",
    "Lessons",
)

DEFAULT_PRACTICE = """\
# Practice

What this artist knows about its own work so far. Rewritten each Sunday from the week's
pieces and the feedback it learns from; read before every piece. Edit freely.

## Style
Not found yet. Make pieces, notice what keeps coming back.

## Palettes I like
None recorded yet.

## Media and technique
Nothing tried yet; see the catalog for what each medium is known to do.

## Themes
The directions in interests.md, in the artist's own words once it has some.

## Stretching toward
Pick one unfamiliar technique or medium and make studies with it this week.

## Lessons
None yet.
"""


def practice_path(base: Path) -> Path:
    return base / "_state" / "practice.md"


def load_practice(base: Path) -> str:
    path = practice_path(base)
    return path.read_text(encoding="utf-8") if path.exists() else DEFAULT_PRACTICE


def save_practice(base: Path, text: str) -> Path:
    path = practice_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def is_complete(text: str) -> bool:
    """A rewrite must keep every section, or the artist loses part of its memory of itself."""
    return all(f"## {s}" in text for s in SECTIONS)
