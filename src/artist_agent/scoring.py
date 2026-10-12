"""Self-critique: a vision-capable LLM looks at the generated image and scores it against
the intention it was made with, and says what the piece taught."""

import base64
from dataclasses import dataclass

from local_first_common.providers.base import BaseProvider
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """\
You are critiquing a generated image for an artist with a daily practice. You see the
theme it belongs to, the artist's intention for the piece (idea, palette, feel, medium,
technique, mode, and the hypothesis it set out to test) and the prompt that was used.

Score honestly, not encouragingly -- this score is what the artist learns from, so a
generic or cliche-ridden result must score low even if it's technically well-composed.

Rate 0.0-1.0 on, in order of weight:
- Is it genuinely interesting -- would someone stop and look twice? Following the rules is
  not the same thing; a compliant but forgettable image scores low.
- Does it realize the intention: the idea is there, the palette and feel landed, and it
  explores the theme including its genre (a futuristic theme has to read as the future)?
- Does it avoid the cliches the theme asked to avoid?
- Composition and visual coherence (not photorealism -- this is generative art).

A study is judged by what it teaches as much as by how it looks; say so when a study
failed well. Write one or two sentences of rationale, specific to what is in the image.

Then write `lesson`: one sentence the artist should carry forward, about the medium,
the technique, the palette or the idea, answering the hypothesis when there was one
("FLUX at 4 steps holds a two-hue flat palette on black; the pulse read as a glow, not a
line" is a lesson; "keep improving" is not).
"""


class SelfScoreModel(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    notes: str
    lesson: str = ""


@dataclass
class SelfScore:
    """`score` is None when the critic call failed: the piece is kept but stays out of every
    average (a 0.0 used to stand in, and dragged its direction's score down for nothing)."""

    score: float | None
    notes: str
    lesson: str = ""


def _intention_text(intention: dict | None, settings: dict | None) -> str:
    if not intention:
        return ""
    lines = [f"{k}: {v}" for k, v in intention.items() if v and k != "why"]
    if settings:
        shown = {k: v for k, v in settings.items() if k != "seed"}
        lines.append("technique: " + ", ".join(f"{k} {v}" for k, v in shown.items()))
    return "Intention:\n" + "\n".join(lines) + "\n\n"


def self_score(
    vision_provider: BaseProvider,
    interest_description: str,
    image_prompt: str,
    image_bytes: bytes,
    intention: dict | None = None,
    settings: dict | None = None,
) -> SelfScore:
    """Score a generated image against the theme, the intention and the prompt it came from."""
    image_b64 = base64.b64encode(image_bytes).decode()
    user_prompt = (
        f"Theme being explored: {interest_description}\n\n"
        f"{_intention_text(intention, settings)}"
        f"Prompt used: {image_prompt}"
    )

    result = vision_provider.complete(SYSTEM_PROMPT, user_prompt, response_model=SelfScoreModel, images=[image_b64])
    return SelfScore(score=result.score, notes=result.notes, lesson=result.lesson or "")
