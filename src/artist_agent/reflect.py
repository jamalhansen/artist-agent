"""Reflection: an artist rewrites its own direction descriptions from its feedback.

Choosing among fixed directions can only teach an artist which brief to repeat, not how
to make better art; rewriting the briefs is the part that can. Every revision is archived
under `_state/history/` so the briefs' evolution can be read back later.

Leeway (2026-10-11): a brief's core idea may move when the feedback says the idea itself
is the problem, and each reflection may add one direction or retire one, within
MIN_DIRECTIONS..MAX_DIRECTIONS. Both artists get the same rule, so the experiment's one
difference (who they learn from) stays the only difference. Existing titles still never
change, so a direction's history stays readable.
"""

from datetime import datetime
from pathlib import Path

from local_first_common.providers.base import BaseProvider
from pydantic import BaseModel, Field

from .catalog import ImageModel, describe
from .feedback import Feedback
from .interests import SETTING_KEYS, Interest, render_interests

FEEDBACK_SOURCE = {
    "self": "your own critic's 0-1 scores and notes on each piece",
    "jamal": "a human viewer's ratings (0-1, from 1-5 stars) and notes on each piece",
}

MIN_DIRECTIONS = 3
MAX_DIRECTIONS = 5

SYSTEM_PROMPT = """\
You maintain the creative briefs for an AI artist's daily image experiment. Each brief
("direction") has a fixed title and a description; every day a prompt writer turns one
description into an image-generation prompt.

You get each direction's current description and the feedback on recent pieces made
from it: {source}. Rewrite each description so future pieces earn better feedback.

Rules:
- Keep each direction's title. Usually revise how it's pursued; but when the feedback says
  the idea itself isn't landing (not just the execution), you may move the core idea, and
  say so in `change`.
- Be concrete: name what to do more of and what to stop, using what the feedback actually says.
- Keep any explicit "avoid" constraints already in a description.
- A direction with no feedback stays exactly as it is.
- `change` is one line on what you changed and why, or "unchanged".
- You may propose at most ONE `new_direction` (a fresh title and description) when the
  feedback points somewhere none of the current directions go, and at most ONE `retire`
  (an existing title) when its feedback has been consistently poor and the brief has
  already been revised before. Leave both null otherwise. There must stay between
  {min_dirs} and {max_dirs} directions.
- Each direction renders with one image model and settings (shown with it, and on each
  piece of feedback). You may move a direction to another model from the catalog below, or
  change its steps, cfg, width, height, sampler or clip_skip, when the feedback suggests the
  renderer rather than the brief is the limit: say so in `change`. Changing the model is a
  real experiment, so do it for at most one direction per reflection, and leave model and
  settings null to keep what the direction has. Never set a seed.

Image models available:
{catalog}
"""


class Revision(BaseModel):
    title: str
    description: str = Field(..., description="The full revised description")
    change: str = Field(..., description="One line: what changed and why, or 'unchanged'")
    model: str | None = Field(None, description="Catalog model file to move this direction to, or null")
    steps: int | None = None
    cfg: float | None = None
    width: int | None = None
    height: int | None = None
    sampler: str | None = None
    clip_skip: int | None = None

    def settings(self) -> dict:
        """Only the settings the model actually set."""
        return {k: getattr(self, k) for k in SETTING_KEYS if k != "seed" and getattr(self, k) is not None}


class Reflection(BaseModel):
    directions: list[Revision]
    new_direction: Revision | None = Field(None, description="One added direction, or null")
    retire: str | None = Field(None, description="Title of one direction to drop, or null")


def _render_settings(settings: dict) -> str:
    shown = {k: v for k, v in settings.items() if k != "seed"}
    return ", ".join(f"{k} {v}" for k, v in shown.items()) if shown else "model defaults"


def build_user(interests: list[Interest], feedback: list[Feedback], default_model: str = "") -> str:
    parts = []
    for i in interests:
        rows = [f for f in feedback if f.interest == i.title]
        renders_with = (
            f"Renders with: {i.model or default_model or 'the default model'} ({_render_settings(i.settings)})"
        )
        lines = [f"## {i.title}", i.description.strip(), "", renders_with, "", f"Feedback ({len(rows)} pieces):"]
        lines += [
            f"- score {f.score:.2f}: {f.notes or '(no notes)'} | prompt: {f.prompt[:300]}"
            + (f" | rendered with {f.model} ({_render_settings(f.settings)})" if f.model else "")
            for f in rows
        ] or ["- none"]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def revise(
    provider: BaseProvider,
    interests: list[Interest],
    feedback: list[Feedback],
    learns_from: str,
    catalog: list[ImageModel] | None = None,
    default_model: str = "",
) -> Reflection:
    system = SYSTEM_PROMPT.format(
        source=FEEDBACK_SOURCE[learns_from],
        min_dirs=MIN_DIRECTIONS,
        max_dirs=MAX_DIRECTIONS,
        catalog=describe(catalog or []),
    )
    return provider.complete(system, build_user(interests, feedback, default_model), response_model=Reflection)


def apply(
    interests: list[Interest],
    reflection: Reflection,
    feedback: list[Feedback],
    catalog: list[ImageModel] | None = None,
) -> tuple[list[Interest], list[str]]:
    """Revised interests and a changelog.

    Only directions that got feedback may change. One direction may be retired (if it got
    feedback and MIN_DIRECTIONS remain) and one added (if its title is new and MAX_DIRECTIONS
    isn't exceeded); a new direction starts unscored with the tool's default model. A model
    move must name a catalog entry and happens for at most one direction; settings changes
    merge over the direction's own (a seed is never taken from the model).
    """
    by_title = {r.title.strip(): r for r in reflection.directions}
    with_feedback = {f.interest for f in feedback}
    known = {m.file for m in catalog or []}
    out, log = [], []
    model_moved = False
    for i in interests:
        r = by_title.get(i.title)
        if not (r and i.title in with_feedback):
            out.append(i)
            continue
        model, settings, notes = i.model, dict(i.settings), []
        if r.model and r.model != i.model and r.model in known and not model_moved:
            model, settings, model_moved = r.model, {}, True  # new model, its catalog defaults apply at run time
            notes.append(f"now renders with {r.model}")
        if r.settings():
            settings.update(r.settings())
            notes.append("settings " + _render_settings(r.settings()))
        description = i.description
        if r.description.strip() and r.change.strip().lower() != "unchanged":
            description = r.description.strip()
            notes.insert(0, r.change.strip())
        if description == i.description and model == i.model and settings == i.settings:
            out.append(i)
            continue
        out.append(Interest(i.title, i.score, i.generations, description, model, settings, i.signal))
        log.append(f"- **{i.title}**: " + "; ".join(notes))
    retire = (reflection.retire or "").strip()
    if retire and retire in with_feedback and any(i.title == retire for i in out) and len(out) > MIN_DIRECTIONS:
        out = [i for i in out if i.title != retire]
        log.append(f"- **{retire}**: retired")
    new = reflection.new_direction
    if new and new.title.strip() and new.description.strip() and len(out) < MAX_DIRECTIONS:
        title = new.title.strip()
        if all(i.title != title for i in out):
            out.append(Interest(title, 0.0, 0, new.description.strip()))
            log.append(f"- **{title}**: new direction ({new.change.strip() or 'no reason given'})")
    return out, log


def consumed_path(base: Path) -> Path:
    return base / "_state" / "reflected.txt"


def unconsumed(base: Path, feedback: list[Feedback]) -> list[Feedback]:
    path = consumed_path(base)
    seen = set(path.read_text(encoding="utf-8").split()) if path.exists() else set()
    return [f for f in feedback if f.stem not in seen]


def record(
    base: Path,
    preamble: str,
    before: list[Interest],
    after: list[Interest],
    log: list[str],
    used: list[Feedback],
    now: datetime,
) -> Path:
    """Archive the pre-revision file, append the changelog, mark the feedback consumed."""
    history = base / "_state" / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d-%H%M")
    snapshot = history / f"interests-{stamp}.md"
    snapshot.write_text(render_interests(before, preamble=preamble), encoding="utf-8")
    with (history / "changelog.md").open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {now.date().isoformat()} ({len(used)} pieces of feedback)\n"
            + ("\n".join(log) or "- no changes")
            + "\n"
        )
    with consumed_path(base).open("a", encoding="utf-8") as f:
        f.write("".join(f"{u.stem}\n" for u in used))
    return snapshot
