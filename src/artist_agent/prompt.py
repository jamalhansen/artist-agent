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
  a hoodie," no glowing blue circuit-board skin. If the direction mentions cyberpunk,
  find an ordinary, specific, slightly-off detail instead of the genre's stock imagery.
- Be concrete about composition, lighting, and color -- vague mood words ("moody",
  "atmospheric") without a concrete visual to hang them on are worse than nothing.
- Output ONLY the prompt itself, as one paragraph. No preamble, no title, no
  explanation of your choices.
"""


def compose_prompt(provider: BaseProvider, interest_description: str) -> str:
    """Send an interest's description to the provider and return a ready-to-use prompt."""
    return provider.complete(SYSTEM_PROMPT, interest_description).strip()
