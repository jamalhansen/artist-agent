"""Reflection: an artist rewrites its own direction descriptions from its feedback.

Choosing among fixed directions can only teach an artist which brief to repeat, not how
to make better art; rewriting the briefs is the part that can. Titles stay fixed so the
two artists' directions stay comparable, and every revision is archived under
`_state/history/` so the briefs' evolution can be read back later.
"""

from datetime import datetime
from pathlib import Path

from local_first_common.providers.base import BaseProvider
from pydantic import BaseModel, Field

from .feedback import Feedback
from .interests import Interest, render_interests

FEEDBACK_SOURCE = {
    "self": "your own critic's 0-1 scores and notes on each piece",
    "jamal": "a human viewer's ratings (0-1, from 1-5 stars) and notes on each piece",
}

SYSTEM_PROMPT = """\
You maintain the creative briefs for an AI artist's daily image experiment. Each brief
("direction") has a fixed title and a description; every day a prompt writer turns one
description into an image-generation prompt.

You get each direction's current description and the feedback on recent pieces made
from it: {source}. Rewrite each description so future pieces earn better feedback.

Rules:
- Keep each direction's title and its core idea. Revise how it's pursued, not what it's about.
- Be concrete: name what to do more of and what to stop, using what the feedback actually says.
- Keep any explicit "avoid" constraints already in a description.
- A direction with no feedback stays exactly as it is.
- `change` is one line on what you changed and why, or "unchanged".
"""


class Revision(BaseModel):
    title: str
    description: str = Field(..., description="The full revised description")
    change: str = Field(..., description="One line: what changed and why, or 'unchanged'")


class Reflection(BaseModel):
    directions: list[Revision]


def build_user(interests: list[Interest], feedback: list[Feedback]) -> str:
    parts = []
    for i in interests:
        rows = [f for f in feedback if f.interest == i.title]
        lines = [f"## {i.title}", i.description.strip(), "", f"Feedback ({len(rows)} pieces):"]
        lines += [f"- score {f.score:.2f}: {f.notes or '(no notes)'} | prompt: {f.prompt[:300]}" for f in rows] or ["- none"]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def revise(provider: BaseProvider, interests: list[Interest], feedback: list[Feedback], learns_from: str) -> list[Revision]:
    system = SYSTEM_PROMPT.format(source=FEEDBACK_SOURCE[learns_from])
    return provider.complete(system, build_user(interests, feedback), response_model=Reflection).directions


def apply(interests: list[Interest], revisions: list[Revision], feedback: list[Feedback]) -> tuple[list[Interest], list[str]]:
    """Revised interests and a changelog. Only directions that got feedback may change."""
    by_title = {r.title.strip(): r for r in revisions}
    with_feedback = {f.interest for f in feedback}
    out, log = [], []
    for i in interests:
        r = by_title.get(i.title)
        if r and i.title in with_feedback and r.description.strip() and r.change.strip().lower() != "unchanged":
            out.append(Interest(i.title, i.score, i.generations, r.description.strip(), i.model, dict(i.settings), i.signal))
            log.append(f"- **{i.title}**: {r.change.strip()}")
        else:
            out.append(i)
    return out, log


def consumed_path(base: Path) -> Path:
    return base / "_state" / "reflected.txt"


def unconsumed(base: Path, feedback: list[Feedback]) -> list[Feedback]:
    path = consumed_path(base)
    seen = set(path.read_text(encoding="utf-8").split()) if path.exists() else set()
    return [f for f in feedback if f.stem not in seen]


def record(base: Path, preamble: str, before: list[Interest], after: list[Interest],
           log: list[str], used: list[Feedback], now: datetime) -> Path:
    """Archive the pre-revision file, append the changelog, mark the feedback consumed."""
    history = base / "_state" / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d-%H%M")
    snapshot = history / f"interests-{stamp}.md"
    snapshot.write_text(render_interests(before, preamble=preamble), encoding="utf-8")
    with (history / "changelog.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {now.date().isoformat()} ({len(used)} pieces of feedback)\n" + ("\n".join(log) or "- no changes") + "\n")
    with consumed_path(base).open("a", encoding="utf-8") as f:
        f.write("".join(f"{u.stem}\n" for u in used))
    return snapshot
