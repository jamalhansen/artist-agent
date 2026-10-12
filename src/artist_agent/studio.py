"""The studio session: the artist decides what to make today, and how, before it makes it.

Before 2026-10-12 a weighted coin chose the direction, another chose the image model, and a
small local model wrote a prompt. The artist made no decisions. Now one structured call,
given the practice file, the themes, the recent work with its feedback, and the catalog,
returns an Intention: the idea, palette and feel; the medium and technique; the size;
whether this is a study or a piece; what it expects to learn; and the prompt. All of that is
recorded on the item, so the critic can judge the piece against what was meant and the
Sunday reflection can see which choices paid off.
"""

import random
from dataclasses import dataclass

from local_first_common.providers.base import BaseProvider
from pydantic import BaseModel, Field

from .catalog import ImageModel, describe, find, technique_for
from .feedback import Feedback, ModelRecord
from .interests import Interest, pick_model

MODES = ("study", "piece")
RECENT_PIECES = 14

FEEDBACK_SOURCE = {
    "self": "your own critic's score (0-1), notes and lesson on each piece",
    "jamal": "the viewer's rating (0-1, from 1-5 stars) and notes on each piece; pieces not yet rated say so",
}

SYSTEM_PROMPT = """\
You are an artist with a daily practice, making one generative image a day with Draw Things.
Your goal each day is to make something and learn something. Over time, through practice and
critique, you develop themes you like to explore, colors you like, media and techniques you
like to use, and a style; and you stretch that style on purpose by trying new techniques.

Today you decide everything about the piece before making it:
- the theme: one of your themes, or a one-off idea when the practice file or the recent work
  points somewhere none of them go (give it a short new title);
- the idea: the specific image, as a painter would describe it before touching the canvas;
- the palette (named colors, not mood words), and the feel you are after;
- the medium: one of the models below, chosen for what it does well with this idea (your
  practice file and record say what each has done in your hands);
- the technique: steps, cfg, sampler, clip_skip and size, inside the medium's room, chosen
  to fit the image (not the defaults every time: defaults are where exploration starts);
- the mode: "piece" when you are going for a finished work, "study" when the point is to
  learn something (a new medium, a technique, a palette) and the result may fail;
- the hypothesis: one line on what you expect to learn, specific enough to be wrong;
- follows: when you are trying again, the exact stem of the one earlier piece you return to
  (as listed under recent work, e.g. 2026-10-04-a-holographic-neural-network-lattice), else null.

Then write the prompt for the medium: concrete about subject, composition, light and color;
plain sentences for FLUX and Z Image, denser detail for SDXL. Output ONLY the structured
answer.

Your feedback comes from {source}. Learn from it: do more of what was rated well, change
what was not, and read the lessons. Do not repeat yesterday's theme and medium unless you
are deliberately returning to a piece. About one piece in three should be a study that tries
something from "Stretching toward". Keep any "avoid" rules the themes carry.

Media available:
{catalog}
"""


class Intention(BaseModel):
    theme: str = Field(..., description="A theme title from the list, or a short new title for a one-off")
    idea: str = Field(..., description="The specific image, one to three sentences")
    palette: str = Field(..., description="Named colors")
    feel: str = Field(..., description="The feeling aimed for")
    medium: str = Field(..., description="A model file from the catalog")
    steps: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    clip_skip: int | None = None
    size: str | None = Field(None, description='"WxH" from the medium\'s sizes')
    mode: str = Field("piece", description='"study" or "piece"')
    hypothesis: str = Field("", description="What this piece should teach, one line")
    follows: str | None = Field(
        None, description="Exact stem of the one earlier piece this returns to, as listed under recent work, or null"
    )
    why: str = Field("", description="One or two lines on the choices")
    prompt: str = Field(..., description="The image prompt for the chosen medium")

    def technique(self) -> dict:
        return {
            k: v
            for k, v in {
                "steps": self.steps,
                "cfg": self.cfg,
                "sampler": self.sampler,
                "clip_skip": self.clip_skip,
                "size": self.size,
            }.items()
            if v is not None
        }

    def record(self) -> dict:
        """What goes on the item under `intention:` (the prompt and settings have their own keys)."""
        return {
            "theme": self.theme,
            "idea": self.idea,
            "palette": self.palette,
            "feel": self.feel,
            "medium": self.medium,
            "mode": self.mode if self.mode in MODES else "piece",
            "hypothesis": self.hypothesis,
            "follows": self.follows or None,
            "why": self.why,
        }


def _settings_text(settings: dict) -> str:
    shown = {k: v for k, v in settings.items() if k != "seed"}
    return ", ".join(f"{k} {v}" for k, v in shown.items()) if shown else "defaults"


def _record_text(by_model: dict[str, ModelRecord]) -> str:
    if not by_model:
        return "no record yet"
    ranked = sorted(by_model.items(), key=lambda kv: kv[1].mean, reverse=True)
    return "; ".join(f"{m} {r.mean:.2f} over {r.count}" for m, r in ranked)


def _piece_text(f: Feedback, learns_from: str) -> str:
    it = f.intention or {}
    head = f"- {f.stem}: {it.get('theme') or f.interest}"
    if it.get("mode"):
        head += f" ({it['mode']})"
    head += f" | {f.model or 'unknown model'} ({_settings_text(f.settings)})"
    if it.get("idea"):
        head += f"\n  idea: {it['idea'][:200]}"
    if it.get("palette"):
        head += f" | palette: {it['palette'][:80]}"
    if it.get("hypothesis"):
        head += f"\n  hypothesis: {it['hypothesis'][:160]}"
    if f.score is None:
        head += "\n  not yet rated"
    else:
        head += f"\n  score {f.score:.2f}: {f.notes or '(no notes)'}"
    if learns_from == "self" and f.lesson:
        head += f"\n  lesson: {f.lesson}"
    return head


def build_user(
    practice: str,
    interests: list[Interest],
    record: dict[str, dict[str, ModelRecord]],
    recent: list[Feedback],
    learns_from: str,
    current_event: str | None = None,
    today: str = "",
) -> str:
    parts = [f"Today: {today}" if today else "", "# Your practice file", practice.strip(), "# Your themes"]
    for i in interests:
        pin = f" Pinned to {i.model}." if i.model else ""
        parts.append(
            f"## {i.title}\nscore {i.score:.2f} over {i.generations} pieces; by medium: "
            f"{_record_text(record.get(i.title, {}))}.{pin}\n{i.description.strip()}"
        )
    parts.append(f"# Recent work (last {len(recent)} pieces, oldest first)")
    parts.append("\n".join(_piece_text(f, learns_from) for f in recent) or "- none yet")
    if current_event:
        parts.append(f"# Today's real-world signal (optional; never depict it literally)\n{current_event}")
    return "\n\n".join(p for p in parts if p)


def plan(
    provider: BaseProvider,
    practice: str,
    interests: list[Interest],
    record: dict[str, dict[str, ModelRecord]],
    recent: list[Feedback],
    learns_from: str,
    catalog: list[ImageModel],
    current_event: str | None = None,
    today: str = "",
) -> Intention:
    system = SYSTEM_PROMPT.format(source=FEEDBACK_SOURCE[learns_from], catalog=describe(catalog))
    user = build_user(practice, interests, record, recent, learns_from, current_event, today)
    return provider.complete(system, user, response_model=Intention)


@dataclass
class Choice:
    """What the session settled on after the intention met the constraints."""

    model: str
    settings: dict
    how: str  # "chosen", "pinned by <theme>", or "drawn (medium not in catalog)"
    interest: Interest | None  # the matching theme, if the intention named one


def resolve(
    intention: Intention,
    interests: list[Interest],
    catalog: list[ImageModel],
    record: dict[str, dict[str, ModelRecord]],
    default_model: str,
    rng: random.Random | None = None,
) -> Choice:
    """Turn the intention into a model and settings draw-things-cli can run.

    A theme Jamal pinned to a model wins over the artist's medium; otherwise the artist's
    medium is used if it is in the catalog, else one is drawn by the old weighted record. The
    technique is clamped to the medium's room; a theme's own setting lines (Jamal's) go on
    top; a seed is chosen so the piece can be re-run.
    """
    interest = next((i for i in interests if i.title.strip().lower() == intention.theme.strip().lower()), None)
    if interest and interest.model:
        model, how = interest.model, f"pinned by {interest.title}"
    elif find(catalog, intention.medium):
        model, how = intention.medium, "chosen"
    elif catalog:
        by_model = record.get(interest.title, {}) if interest else {}
        model = pick_model([m.file for m in catalog], {m: r.mean for m, r in by_model.items()}, rng)
        how = f"drawn ({intention.medium!r} is not in the catalog)"
    else:
        model, how = default_model, "default"
    settings = technique_for(catalog, model, intention.technique())
    if interest:
        settings.update({k: v for k, v in interest.settings.items() if k != "seed"})
    settings["seed"] = (rng or random.Random()).randrange(2**32)
    return Choice(model, settings, how, interest)
