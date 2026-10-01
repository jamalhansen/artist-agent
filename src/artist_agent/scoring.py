"""Self-critique: a vision-capable LLM looks at the generated image and scores it
against the direction and prompt it came from."""
import base64
from dataclasses import dataclass

from local_first_common.providers.base import BaseProvider
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """\
You are critiquing a generated image against the creative direction and prompt it was
made from, for a personal daily art project.

Score honestly, not encouragingly -- this score is used to decide whether to keep
exploring this creative direction, so a generic or cliche-ridden result must score low
even if it's technically well-composed.

Rate 0.0-1.0 on, in order of weight:
- Is it genuinely interesting -- would someone stop and look twice? Following the
  direction's rules is not the same thing; a compliant but forgettable image scores low.
- Does it actually explore the stated direction, including its genre (a futuristic
  direction has to read as the future), or is it generic imagery that could illustrate
  almost anything?
- Does it avoid the genre cliches the direction explicitly asked to avoid?
- Composition and visual coherence (not photorealism -- this is generative art).

Write one or two sentences of rationale, specific to what's actually in the image.
"""


class SelfScoreModel(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    notes: str


@dataclass
class SelfScore:
    score: float
    notes: str


def self_score(
    vision_provider: BaseProvider, interest_description: str, image_prompt: str, image_bytes: bytes
) -> SelfScore:
    """Score a generated image against the direction and prompt it came from."""
    image_b64 = base64.b64encode(image_bytes).decode()
    user_prompt = f"Direction being explored: {interest_description}\n\nPrompt used: {image_prompt}"

    result = vision_provider.complete(
        SYSTEM_PROMPT, user_prompt, response_model=SelfScoreModel, images=[image_b64]
    )
    return SelfScore(score=result.score, notes=result.notes)
