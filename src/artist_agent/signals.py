"""Optional real-world signal for the 'current events' creative direction.

content-discovery-agent already scans and scores AI/tech content daily and
writes each kept item into Contexta's inbox/ as plain markdown -- this reuses
that existing, already-flowing output rather than adding a new fetch. Scope
note: it's Jamal's own AI/tech radar, not general news; that's a deliberate
choice (real data today vs. building a new headline fetcher), not an
oversight.

Best-effort by design: if Contexta's inbox is missing, empty, or unreadable,
callers get None and generation proceeds exactly as it did before this
existed. A missing signal is never a reason to fail a whole day's image.
"""
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter

logger = logging.getLogger(__name__)

CONTEXTA_INBOX_DIR = os.environ.get("CONTEXTA_INBOX_DIR", "~/vaults/Contexta/inbox/")

_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


@dataclass
class CurrentEvent:
    title: str
    source_url: str | None


def _title_from_post(post: frontmatter.Post) -> str | None:
    match = _TITLE_RE.search(post.content)
    return match.group(1).strip() if match else None


def fetch_current_event(inbox_dir: str | Path | None = None) -> CurrentEvent | None:
    """Return the most recently captured content-discovery-agent item, if any.

    "Most recent" is a deliberately simple proxy for "top" -- captured inbox
    files don't retain the numeric relevance score content-discovery-agent
    scored them with internally, only whether they cleared its keep threshold.
    """
    base = Path(inbox_dir or CONTEXTA_INBOX_DIR).expanduser()
    if not base.is_dir():
        return None

    candidates = []
    for path in base.glob("*.md"):
        try:
            post = frontmatter.load(path)
        except Exception as e:  # noqa: BLE001 - a malformed inbox file must never break generation
            logger.warning("skipping unreadable inbox file %s: %s", path, e)
            continue
        if post.get("source_type") != "content-discovery-agent":
            continue
        captured = post.get("captured")
        if not captured:
            continue
        candidates.append((str(captured), post))

    if not candidates:
        return None

    _, best_post = max(candidates, key=lambda pair: pair[0])
    title = _title_from_post(best_post) or best_post.get("title")
    if not title:
        return None
    return CurrentEvent(title=title, source_url=best_post.get("source_url"))
