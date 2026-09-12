"""Paths, config, and the on-disk item-note format.

Everything under the agent's root folder is plain markdown + YAML frontmatter and
plain image files -- deliberately not Obsidian-specific (no wikilinks, no embeds) and
not a database, so the same folder works unmodified as a vault, a static-site source,
or neither, without a migration later.
"""
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import frontmatter

DEFAULT_AI_ARTIST_DIR = os.environ.get("AI_ARTIST_DIR", "~/iCloud/ai-artist/")

DRAW_THINGS_BASE_URL = os.environ.get("DRAW_THINGS_BASE_URL", "http://127.0.0.1:7860")
DRAW_THINGS_MODEL = os.environ.get("DRAW_THINGS_MODEL", "sd_xl_base_1.0_f16.ckpt")


def resolve_ai_artist_dir(path: str | None) -> Path:
    return Path(path or DEFAULT_AI_ARTIST_DIR).expanduser()


def images_dir(base: Path) -> Path:
    return base / "images"


def items_dir(base: Path) -> Path:
    return base / "items"


def interests_path(base: Path) -> Path:
    return base / "_state" / "interests.md"


_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_words: int = 6, max_chars: int = 50) -> str:
    """Short filename slug from free text (an interest title, a prompt)."""
    source = " ".join(text.split()[:max_words])
    slug = _SLUG_NON_ALNUM_RE.sub("-", source.lower()).strip("-")
    return slug[:max_chars].rstrip("-") or "untitled"


@dataclass
class Item:
    """One generated artwork -- the unit this whole tool produces."""

    generated_at: datetime
    prompt: str
    interest_title: str
    settings: dict
    image_filename: str
    self_score: float
    self_score_notes: str
    human_score: float | None = None
    human_notes: str | None = None
    status: str = "new"


def item_stem(item: Item) -> str:
    d = item.generated_at.date().isoformat()
    return f"{d}-{slugify(item.prompt)}"


def render_item_note(item: Item, relative_image_path: str) -> str:
    """Render an item as a portable markdown note: YAML frontmatter, then the
    image (plain markdown syntax, not an Obsidian embed) and the agent's own
    self-critique. human_score/human_notes start null -- Jamal fills those in
    by editing this file directly, no separate review tool needed.

    Built via python-frontmatter (not hand-formatted strings) so the YAML is
    always valid regardless of what characters end up in a prompt -- the same
    library local_first_common.obsidian and weekly-review-generator already
    use to read files like this one back.
    """
    body = (
        f"![{item.interest_title}]({relative_image_path})\n\n"
        f"**Self-critique ({item.self_score:.2f}):** {item.self_score_notes}\n\n"
        "**Human notes:** " + (item.human_notes or "_not yet reviewed_") + "\n"
    )
    post = frontmatter.Post(
        body,
        generated_at=item.generated_at.isoformat(),
        interest=item.interest_title,
        prompt=item.prompt,
        settings=item.settings,
        self_score=item.self_score,
        human_score=item.human_score,
        status=item.status,
    )
    return frontmatter.dumps(post) + "\n"
