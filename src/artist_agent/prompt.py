"""Turns a chosen interest's free-text description into a concrete Draw Things prompt."""

from local_first_common.providers.base import BaseProvider

SYSTEM_PROMPT = """\
You write image-generation prompts for a Stable Diffusion XL model, for a personal daily
art project.

You will be given one creative direction the artist is currently exploring -- a
description of a visual territory, not a specific image. Write ONE concrete, vivid
prompt for a single image that explores this direction freshly. Do not just restate the
direction's description -- invent a specific scene, composition, or subject within it
that hasn't obviously been done before.

Rules:
- Avoid genre cliches: no rain-slicked streets, no giant neon kanji signs, no "hacker in
  a hoodie," no glowing blue circuit-board skin. Avoiding the cliche is not avoiding the
  genre: if the direction is cyberpunk or otherwise futuristic, the image still has to
  read as the future -- invent technology, materials, and infrastructure that don't
  exist yet rather than retreating to an ordinary present-day scene.
- Be concrete about composition, lighting, and color -- vague mood words ("moody",
  "atmospheric") without a concrete visual to hang them on are worse than nothing.
- Output ONLY the prompt itself, as one paragraph. No preamble, no title, no
  explanation of your choices.

If you are also given a current-event signal, treat it exactly like any other genre
cliche to avoid depicting literally: no recognizable people, logos, headlines, screens,
or literal scenes from it. Instead let its mood, tempo, or emotional temperature inform
the image's color, motion, and composition -- the same translation a synesthete makes
from sound to shape, applied to a real event instead.
"""


def compose_prompt(provider: BaseProvider, interest_description: str, current_event: str | None = None) -> str:
    """Send an interest's description to the provider and return a ready-to-use prompt.

    current_event, when given, is a real topic (a headline/title, not a full
    article) to draw mood/tempo/feeling from -- see SYSTEM_PROMPT's rule on
    treating it like any other cliche to avoid depicting literally.
    """
    user_message = interest_description
    if current_event:
        user_message += f"\n\nToday's real-world signal to draw mood from (do not depict literally): {current_event}"
    return provider.complete(SYSTEM_PROMPT, user_message).strip()
