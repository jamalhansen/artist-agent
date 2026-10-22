"""Keeping the practice from collapsing onto one theme.

Between 2026-10-12 and 10-21 the studio session, free to choose any theme, chose the same
one every day: ten days running for the self-taught artist, six for the mentored one, while
Jamal rated every piece 1/5. The planner reads its recent work and treats the day as fixing
yesterday's piece, so a failing theme pulls it back harder each time. Two rules, the same
for both artists so the experiment stays fair:

- A theme made STREAK_LIMIT days running rests for a day: it is left off the list.
- A wild card replaces the themes when the artist's own feedback says it keeps failing
  (its last FAILING_RUN rated pieces all below FAIL_BELOW), and on about one day in seven
  regardless. The code draws a subject, a material and an art movement at random and a
  medium the artist hasn't used lately; the planner, shown none of its themes or recent
  work, makes a one-off from the draw. The randomness is the code's: a model asked to be
  random repeats itself.
"""

import random
from dataclasses import dataclass

from .catalog import ImageModel
from .feedback import Feedback

STREAK_LIMIT = 2
FAILING_RUN = 3
FAIL_BELOW = 0.4  # on the 0-1 scale: 1-2 stars from Jamal, or a weak self-critique
WILD_CARD_CHANCE = 1 / 7

SUBJECTS = [
    "a lighthouse keeper's kitchen",
    "a flooded library",
    "a beekeeper at dawn",
    "a night market",
    "a whale skeleton in a museum hall",
    "an abandoned greenhouse",
    "a child's treehouse",
    "a glacier calving",
    "a jazz trio mid-song",
    "a fox crossing a parking lot",
    "a mountain village in fog",
    "a cathedral of mushrooms",
    "an orchard in a storm",
    "a clockmaker's bench",
    "a ferry at night",
    "a dancer in mid-leap",
    "a desert after rain",
    "a coral reef at night",
    "a train station in snow",
    "a heron fishing",
    "a laundromat at 3am",
    "a field of sunflowers turning",
    "an octopus in a shipwreck",
    "a crowded bus window",
    "a migrating flock",
    "a tidal pool",
    "a grandmother's hands knitting",
    "a volcano under the stars",
    "a kite festival",
    "a moth drawn to a porch light",
]

MATERIALS = [
    "stained glass",
    "cut paper",
    "woven textile",
    "ink wash",
    "chalk on a blackboard",
    "oil impasto",
    "embroidery",
    "mosaic tile",
    "linocut",
    "watercolor bleeding on wet paper",
    "clay",
    "folded origami",
    "neon tubing",
    "rusted metal",
    "bioluminescence",
    "smoke",
    "frost on glass",
    "pressed flowers",
    "cyanotype",
    "torn magazine collage",
    "carved wood",
    "beadwork",
    "spray paint on brick",
    "porcelain",
]

MOVEMENTS = [
    "Art Nouveau",
    "ukiyo-e woodblock",
    "Bauhaus",
    "Fauvism",
    "Surrealism",
    "Hudson River School",
    "Russian Constructivism",
    "Mexican muralism",
    "Impressionism",
    "De Stijl",
    "Pre-Raphaelite",
    "Abstract Expressionism",
    "Hilma af Klint",
    "Moebius",
    "Studio Ghibli backgrounds",
    "Edward Hopper",
    "Aboriginal dot painting",
    "Islamic geometric pattern",
    "Art Deco travel poster",
    "Gustav Klimt",
    "Hokusai",
    "Georgia O'Keeffe",
    "Wassily Kandinsky",
    "medieval illuminated manuscript",
]


@dataclass
class WildCard:
    reason: str  # "failing" or "chance"
    subject: str
    material: str
    movement: str
    medium: str | None  # a catalog model file, or None when the catalog is empty

    def record(self) -> dict:
        return {
            "reason": self.reason,
            "subject": self.subject,
            "material": self.material,
            "movement": self.movement,
            "medium": self.medium,
        }


def streak(own: list[Feedback]) -> tuple[str, int]:
    """The theme of the artist's latest pieces and how many days running it has been made."""
    if not own:
        return "", 0
    theme = own[-1].interest
    n = 0
    for f in reversed(own):
        if f.interest != theme:
            break
        n += 1
    return theme, n


def resting_theme(own: list[Feedback]) -> str | None:
    theme, n = streak(own)
    return theme if theme and n >= STREAK_LIMIT else None


def failing(own: list[Feedback]) -> bool:
    """The artist's last FAILING_RUN rated pieces all scored below FAIL_BELOW.

    An unrated wild card among the latest two pieces holds this off: its feedback hasn't
    arrived, and Jamal's ratings come days later, so without the hold one bad week would
    make every day a wild card until he caught up.
    """
    if any(f.intention.get("wild_card") and f.score is None for f in own[-2:]):
        return False
    rated = [f for f in own if f.score is not None][-FAILING_RUN:]
    return len(rated) == FAILING_RUN and all(f.score is not None and f.score < FAIL_BELOW for f in rated)


def _lists(settings: dict | None) -> tuple[list[str], list[str], list[str]]:
    s = settings or {}
    return (
        list(s.get("subjects") or SUBJECTS),
        list(s.get("materials") or MATERIALS),
        list(s.get("movements") or MOVEMENTS),
    )


def draw(
    own: list[Feedback],
    catalog: list[ImageModel],
    rng: random.Random,
    reason: str,
    settings: dict | None = None,
) -> WildCard:
    """A random subject, material and movement, and a medium not used in the last three pieces."""
    subjects, materials, movements = _lists(settings)
    recent_models = {f.model for f in own[-3:] if f.model}
    files = [m.file for m in catalog]
    fresh = [m for m in files if m not in recent_models] or files
    return WildCard(
        reason=reason,
        subject=rng.choice(subjects),
        material=rng.choice(materials),
        movement=rng.choice(movements),
        medium=rng.choice(fresh) if fresh else None,
    )


def wild_card_today(
    own: list[Feedback],
    catalog: list[ImageModel],
    rng: random.Random,
    settings: dict | None = None,
    chance: float = WILD_CARD_CHANCE,
) -> WildCard | None:
    if failing(own):
        return draw(own, catalog, rng, "failing", settings)
    if rng.random() < chance:
        return draw(own, catalog, rng, "chance", settings)
    return None
