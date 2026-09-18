"""Reads and updates the agent's own evolving list of creative directions.

Deliberately plain markdown, not a database row: Jamal reads and edits it directly
(add a direction, delete one, reword one) with the same tool he'd use to read the
items it produces. The agent's only job is to bias which direction it leans toward
next and to fold each generation's outcome back in -- not to silently prune or
invent new directions on its own yet. Per the schema draft this design started
from: don't build an auto-mutation scheme before there's enough real generations
to know whether the existing, simpler loop even needs one.

A direction can optionally pin its own model and generation settings (model, steps,
cfg, width, height, seed) alongside its score line. These are paired with the
direction deliberately, not chosen per-run at random: record_outcome() folds each
generation's score back into the direction it came from, so a direction's score
stays attributable to one consistent model+settings combo instead of conflating
"this prompt/direction didn't work" with "that run happened to use a different
checkpoint or step count."
"""
import random
import re
from dataclasses import dataclass, field, replace

_HEADING_RE = re.compile(r"^##\s+(.+)$")
_SCORE_RE = re.compile(r"^score:\s*([\d.]+)\s*\((\d+)\s*generations?\)\s*$")
_KV_RE = re.compile(r"^(model|steps|cfg|width|height|seed):\s*(\S.*)$")
_INT_SETTING_KEYS = ("steps", "width", "height", "seed")

# New/never-scored interests still get picked sometimes -- a bare 0.0 score would
# otherwise never win a weighted draw against anything with real history.
EXPLORATION_FLOOR = 0.15


@dataclass
class Interest:
    title: str
    score: float
    generations: int
    description: str
    model: str | None = None
    settings: dict = field(default_factory=dict)


def _coerce_setting(key: str, value: str) -> int | float:
    return int(value) if key in _INT_SETTING_KEYS else float(value)


def parse_interests(text: str) -> tuple[str, list[Interest]]:
    """Parse interests.md into (preamble, [Interest, ...]), in file order.

    The preamble -- any text above the first "## " heading -- round-trips
    unchanged through render_interests; dropping it would silently delete
    whatever intro text Jamal wrote every time the agent saves an update.

    A section missing a "score: X (N generations)" line (e.g. one Jamal just
    added by hand) defaults to score=0.0, generations=0 -- a fresh direction,
    not an error. Same for model/setting lines -- entirely optional, absent
    means "use the tool's defaults."
    """
    interests = []
    preamble_lines: list[str] = []
    title = None
    score = 0.0
    generations = 0
    model: str | None = None
    settings: dict = {}
    description_lines: list[str] = []

    def flush():
        if title is not None:
            interests.append(
                Interest(
                    title,
                    score,
                    generations,
                    "\n".join(description_lines).strip(),
                    model=model,
                    settings=dict(settings),
                )
            )

    for line in text.splitlines():
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            title = heading_match.group(1).strip()
            score, generations = 0.0, 0
            model, settings = None, {}
            description_lines = []
            continue
        if title is None:
            preamble_lines.append(line)
            continue
        score_match = _SCORE_RE.match(line.strip())
        if score_match:
            score, generations = float(score_match.group(1)), int(score_match.group(2))
            continue
        kv_match = _KV_RE.match(line.strip())
        if kv_match:
            key, value = kv_match.group(1), kv_match.group(2).strip()
            if key == "model":
                model = value
            else:
                settings[key] = _coerce_setting(key, value)
            continue
        description_lines.append(line)

    flush()
    return "\n".join(preamble_lines).strip(), interests


def render_interests(interests: list[Interest], preamble: str = "") -> str:
    """Inverse of parse_interests -- round-trips a list back to the same file shape."""
    parts = [preamble.rstrip()] if preamble.strip() else []
    for interest in interests:
        lines = [
            f"## {interest.title}",
            f"score: {interest.score:.2f} ({interest.generations} generations)",
        ]
        if interest.model:
            lines.append(f"model: {interest.model}")
        for key in ("steps", "cfg", "width", "height", "seed"):
            if key in interest.settings:
                lines.append(f"{key}: {interest.settings[key]}")
        parts.append("\n".join(lines) + "\n\n" + interest.description)
    return "\n\n".join(parts).strip() + "\n"


def pick_direction(interests: list[Interest], rng: random.Random | None = None) -> Interest:
    """Weighted-random pick, biased toward higher-scoring directions.

    A plain proportional-to-score draw would starve anything at 0.0 forever
    (never scored, or never able to recover from one bad run), so every
    interest's weight has an EXPLORATION_FLOOR added -- the agent still leans
    hard toward what's worked, but nothing is permanently locked out.
    """
    if not interests:
        raise ValueError("no interests to pick from")
    rng = rng or random.Random()
    weights = [interest.score + EXPLORATION_FLOOR for interest in interests]
    return rng.choices(interests, weights=weights, k=1)[0]


def record_outcome(interests: list[Interest], title: str, observed_score: float) -> list[Interest]:
    """Fold one generation's self_score into its interest's running average.

    Returns a new list (interests are immutable dataclasses) -- callers persist
    it with render_interests.
    """
    updated = []
    for interest in interests:
        if interest.title != title:
            updated.append(interest)
            continue
        new_generations = interest.generations + 1
        new_score = (interest.score * interest.generations + observed_score) / new_generations
        updated.append(replace(interest, score=new_score, generations=new_generations))
    return updated
