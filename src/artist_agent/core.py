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

DRAW_THINGS_MODEL = os.environ.get("DRAW_THINGS_MODEL", "sd_xl_base_1.0_f16.ckpt")
DRAW_THINGS_CLI_PATH = os.environ.get("DRAW_THINGS_CLI_PATH", "draw-things-cli")

# Remote mode: generate against a running gRPCServerCLI-macOS instead of local
# inference. See drawthings.py's module docstring for why (2026-09-17: local
# inference hung on every scheduled/launchd run, reproduced on demand, root
# cause not confirmed but consistent with a known Metal-in-daemon-context
# limitation). Unset by default -- local inference stays the default until
# remote mode is verified to actually fix the unattended case.
DRAW_THINGS_REMOTE_URL = os.environ.get("DRAW_THINGS_REMOTE_URL") or None
DRAW_THINGS_REMOTE_PORT = int(os.environ.get("DRAW_THINGS_REMOTE_PORT", "7859"))


def resolve_ai_artist_dir(path: str | None) -> Path:
    return Path(path or DEFAULT_AI_ARTIST_DIR).expanduser()


def images_dir(base: Path) -> Path:
    return base / "images"


def items_dir(base: Path) -> Path:
    return base / "items"


def interests_path(base: Path) -> Path:
    return base / "_state" / "interests.md"


def signals_dir(base: Path) -> Path:
    return base / "signals"


_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_words: int = 6, max_chars: int = 50) -> str:
    """Short filename slug from free text (an interest title, a prompt)."""
    source = " ".join(text.split()[:max_words])
    slug = _SLUG_NON_ALNUM_RE.sub("-", source.lower()).strip("-")
    return slug[:max_chars].rstrip("-") or "untitled"


@dataclass
class Item:
    """One generated artwork -- the unit this whole tool produces.

    `models` records which model did which of the three independent, swappable
    steps -- keys "prompt" (composed the image prompt from the chosen interest),
    "image" (Draw Things' generation model), "critique" (the vision model that
    scored the result) -- so a prompt or critique that reads oddly later can be
    traced to a specific model rather than guessed at.
    """

    generated_at: datetime
    prompt: str
    interest_title: str
    settings: dict
    image_filename: str
    self_score: float
    self_score_notes: str
    models: dict
    human_score: float | None = None
    human_notes: str | None = None
    status: str = "new"
    current_event_title: str | None = None
    current_event_source_url: str | None = None


def item_stem(item: Item) -> str:
    d = item.generated_at.date().isoformat()
    return f"{d}-{slugify(item.prompt)}"


def signal_stem(title: str, captured_at: datetime) -> str:
    d = captured_at.date().isoformat()
    return f"{d}-{slugify(title)}"


def render_signal_note(title: str, source_url: str | None, captured_at: datetime) -> str:
    """Durable local snapshot of a current-event signal fetched from Contexta's
    inbox -- lives in the agent's own portable folder (signals/, alongside
    images/ and items/) so a given day's generation stays self-contained even
    if the original inbox file later moves or gets archived by Contexta's own
    reduce pipeline.
    """
    post = frontmatter.Post(
        title,
        title=title,
        source_url=source_url,
        captured_at=captured_at.isoformat(),
    )
    return frontmatter.dumps(post) + "\n"


def derive_title(prompt: str, max_words: int = 12) -> str:
    """A readable title from the prompt's own words -- no extra LLM call just to
    name the thing. Truncated with an ellipsis when the prompt runs longer, since
    these prompts are often a full descriptive paragraph."""
    words = prompt.split()
    title = " ".join(words[:max_words])
    return title + "…" if len(words) > max_words else title


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
        title=derive_title(item.prompt),
        generated_at=item.generated_at.isoformat(),
        interest=item.interest_title,
        prompt=item.prompt,
        settings=item.settings,
        models=item.models,
        self_score=item.self_score,
        human_score=item.human_score,
        status=item.status,
        current_event_title=item.current_event_title,
        current_event_source_url=item.current_event_source_url,
    )
    return frontmatter.dumps(post) + "\n"
