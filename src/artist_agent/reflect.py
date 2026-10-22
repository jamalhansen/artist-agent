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

import re
from datetime import datetime
from pathlib import Path

from local_first_common.providers.base import BaseProvider
from pydantic import BaseModel, Field

from .catalog import ImageModel, describe
from .feedback import Feedback, ModelRecord
from .interests import SETTING_KEYS, Interest, render_interests

FEEDBACK_SOURCE = {
    "self": "your own critic's 0-1 scores and notes on each piece",
    "jamal": "a human viewer's ratings (0-1, from 1-5 stars) and notes on each piece",
}

MIN_DIRECTIONS = 3
MAX_DIRECTIONS = 5
# A longer revision is refused: long briefs were how the directions turned into ban lists.
MAX_DESCRIPTION_WORDS = 120
REFUSE_OVER_WORDS = 160
# Phrasing that bans instead of aiming. Not refused (a refusal would keep the older, worse
# brief), but flagged in the changelog so a relapse shows up.
_BAN_RE = re.compile(r"\b(avoid\w*|ban(?:ned|s)?|never|do not|don't|no more)\b", re.IGNORECASE)


def ban_phrases(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _BAN_RE.finditer(text)})


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
- Write each description as an aim, not a rulebook: what the direction is reaching for,
  and what the feedback says is working, in at most {max_words} words. Each rewrite can
  narrow a direction a little more, until it allows only one safe image (between 2026-10-11
  and 10-18 the briefs grew into paragraphs of bans and every piece became the same simple
  shapes). So widen as often as you narrow.
- Speak in positives. Say what to make, how, and in what spirit. When the feedback says
  something isn't working, rewrite or delete the words that led to it; do not add a rule
  against it. No "avoid", "never", "do not", "no X" or lists of banned motifs: every
  sentence should describe something to reach for. A brief that carries such rules from an
  earlier week gets them turned into positive aims or removed.
- A direction with no feedback stays exactly as it is.
- `change` is one line on what you changed and why, or "unchanged".
- You may propose at most ONE `new_direction` (a fresh title and description) when the
  feedback points somewhere none of the current directions go, and at most ONE `retire`
  (an existing title) when its feedback has been consistently poor and the brief has
  already been revised before. One-offs and wild cards are listed after the directions:
  when one earned clearly better feedback than the directions did, it is the best
  candidate for `new_direction`. Leave both null otherwise. There must stay between
  {min_dirs} and {max_dirs} directions.
- By default each direction draws its image model daily from the catalog below, weighted by
  its own track record (shown per direction, with the model behind each piece of feedback).
  You may instead pin one direction per reflection to a single catalog model when the track
  record is clear enough that the daily draw is wasting pieces, or change a direction's
  steps, cfg, width, height, sampler or clip_skip when the feedback suggests the renderer
  rather than the brief is the limit: say so in `change`. Leave model and settings null to
  keep what the direction has. Never set a seed.

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


def _render_record(by_model: dict[str, ModelRecord]) -> str:
    if not by_model:
        return "Track record by model: none yet"
    ranked = sorted(by_model.items(), key=lambda kv: kv[1].mean, reverse=True)
    return "Track record by model (all pieces so far): " + "; ".join(
        f"{m} {r.mean:.2f} over {r.count} piece{'s' if r.count != 1 else ''} ({_render_settings(r.last_settings)})"
        for m, r in ranked
    )


def _intention_line(f: Feedback) -> str:
    it = f.intention or {}
    if not it:
        return ""
    bits = [f"{k}: {it[k]}" for k in ("mode", "palette", "feel", "hypothesis") if it.get(k)]
    if f.lesson:
        bits.append(f"lesson: {f.lesson}")
    return (" | " + "; ".join(bits)) if bits else ""


def build_user(
    interests: list[Interest],
    feedback: list[Feedback],
    default_model: str = "",
    record: dict[str, dict[str, ModelRecord]] | None = None,
) -> str:
    parts = []
    for i in interests:
        rows = [f for f in feedback if f.interest == i.title]
        renders_with = (
            f"Renders with: {i.model} ({_render_settings(i.settings)}), pinned"
            if i.model
            else f"Renders with: a daily draw from the catalog weighted by the track record ({_render_settings(i.settings)})"
        )
        lines = [
            f"## {i.title}",
            i.description.strip(),
            "",
            renders_with,
            _render_record((record or {}).get(i.title, {})),
            "",
            f"Feedback ({len(rows)} pieces since the last reflection):",
        ]
        lines += [
            f"- score {f.score:.2f}: {f.notes or '(no notes)'} | prompt: {f.prompt[:300]}"
            + (f" | rendered with {f.model} ({_render_settings(f.settings)})" if f.model else "")
            + _intention_line(f)
            for f in rows
        ] or ["- none"]
        parts.append("\n".join(lines))
    titles = {i.title for i in interests}
    one_offs = [f for f in feedback if f.interest not in titles]
    if one_offs:
        parts.append(
            "\n".join(
                ["## One-offs and wild cards (not directions; candidates for new_direction)"]
                + [
                    f"- {f.interest}: score {f.score:.2f}: {f.notes or '(no notes)'} | prompt: {f.prompt[:300]}"
                    + (f" | rendered with {f.model}" if f.model else "")
                    + _intention_line(f)
                    for f in one_offs
                ]
            )
        )
    return "\n\n".join(parts)


def revise(
    provider: BaseProvider,
    interests: list[Interest],
    feedback: list[Feedback],
    learns_from: str,
    catalog: list[ImageModel] | None = None,
    default_model: str = "",
    record: dict[str, dict[str, ModelRecord]] | None = None,
) -> Reflection:
    system = SYSTEM_PROMPT.format(
        source=FEEDBACK_SOURCE[learns_from],
        max_words=MAX_DESCRIPTION_WORDS,
        min_dirs=MIN_DIRECTIONS,
        max_dirs=MAX_DIRECTIONS,
        catalog=describe(catalog or []),
    )
    user = build_user(interests, feedback, default_model, record)
    return provider.complete(system, user, response_model=Reflection)


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
            if len(r.description.split()) > REFUSE_OVER_WORDS:
                notes.insert(0, f"description refused ({len(r.description.split())} words, limit {REFUSE_OVER_WORDS})")
            else:
                description = r.description.strip()
                notes.insert(0, r.change.strip())
                if bans := ban_phrases(description):
                    notes.append("still bans (" + ", ".join(bans) + ")")
        if description == i.description and model == i.model and settings == i.settings:
            out.append(i)
            if notes:  # a refused description is worth a line in the changelog
                log.append(f"- **{i.title}**: " + "; ".join(notes))
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


PRACTICE_PROMPT = """\
You are an artist with a daily generative-art practice, writing up what the week's work
taught you. Your feedback comes from {source}. You get your current practice file, your
themes, and the week's pieces with their intentions (idea, palette, feel, medium, technique,
mode, hypothesis) and their feedback.

Rewrite the practice file so that it is true now. Keep all six sections, with exactly these
headings: {sections}. Under each:
- Style: what keeps coming back in the work you and your feedback liked, in a few sentences;
  say "not found yet" when it is honestly too early.
- Palettes I like: named color combinations that were rated well, and what each did; drop
  ones that stopped working.
- Media and technique: per medium, what it does in your hands and which settings fit which
  kind of image, from the record and the lessons; keep it to what the pieces showed.
- Themes: the directions in your own words, and any one-off idea that worked and deserves
  to come back.
- Stretching toward: one or two unfamiliar techniques, media or palettes to make studies
  with next week, chosen because the record has a gap there, not because they sound nice.
  At least one of them must reach outside your current themes.
- Lessons: the ten most useful lessons from all your work, newest first, one line each.

Be concrete and short; this file is read before every piece. Write it in positives: what
works and what to reach for. A lesson about a failure says what to do instead. Keep anything the viewer wrote
in it by hand that still applies. `change` is two or three lines on what you changed.
"""


class PracticeRevision(BaseModel):
    practice: str = Field(..., description="The full rewritten practice file, markdown")
    change: str = Field(..., description="Two or three lines on what changed and why")


def revise_practice(
    provider: BaseProvider,
    practice: str,
    interests: list[Interest],
    feedback: list[Feedback],
    learns_from: str,
    record: dict[str, dict[str, ModelRecord]] | None = None,
) -> PracticeRevision:
    from .practice import SECTIONS

    system = PRACTICE_PROMPT.format(source=FEEDBACK_SOURCE[learns_from], sections=", ".join(SECTIONS))
    user = "# Current practice file\n\n" + practice.strip() + "\n\n# Themes and the week's pieces\n\n"
    user += build_user(interests, feedback, record=record)
    return provider.complete(system, user, response_model=PracticeRevision)


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
    practice_before: str | None = None,
) -> Path:
    """Archive the pre-revision files, append the changelog, mark the feedback consumed."""
    history = base / "_state" / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d-%H%M")
    snapshot = history / f"interests-{stamp}.md"
    snapshot.write_text(render_interests(before, preamble=preamble), encoding="utf-8")
    if practice_before is not None:
        (history / f"practice-{stamp}.md").write_text(practice_before, encoding="utf-8")
    with (history / "changelog.md").open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {now.date().isoformat()} ({len(used)} pieces of feedback)\n"
            + ("\n".join(log) or "- no changes")
            + "\n"
        )
    with consumed_path(base).open("a", encoding="utf-8") as f:
        f.write("".join(f"{u.stem}\n" for u in used))
    return snapshot
