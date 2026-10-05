from datetime import date, datetime
from pathlib import Path

import pytest

from artist_agent import reflect
from artist_agent.feedback import ArtistConfig, apply_scores, collect, load_config
from artist_agent.interests import Interest


def _item(
    base: Path, stem: str, interest: str, self_score: float, human=None, notes="_not yet reviewed_", day="2026-10-02"
):
    d = base / "items"
    d.mkdir(parents=True, exist_ok=True)
    human_line = f"human_score: {human}\n" if human is not None else "human_score: null\n"
    (d / f"{stem}.md").write_text(
        f"---\ngenerated_at: '{day}T09:00:00-05:00'\ninterest: {interest}\nprompt: a prompt for {stem}\n"
        f"self_score: {self_score}\n{human_line}---\n![x](../images/{stem}.png)\n\n"
        f"**Self-critique ({self_score:.2f}):** critic says {stem}\n\n**Human notes:** {notes}\n",
        encoding="utf-8",
    )


@pytest.fixture
def two_artists(tmp_path):
    a, b = tmp_path / "self-taught", tmp_path / "mentored"
    _item(a, "a1", "Cyber", 0.8, human=0.25, notes="not cyberpunk (2/5)")
    _item(a, "a0", "Cyber", 0.9, human=0.0, day="2026-09-20")  # before the experiment
    _item(b, "b1", "Cyber", 0.4, human=0.75, notes="love it (4/5)")
    _item(b, "b2", "Wire", 0.6)  # unrated
    return a, b


def test_missing_config_is_a_self_taught_artist(tmp_path):
    cfg = load_config(tmp_path)
    assert (cfg.name, cfg.learns_from, cfg.feedback_dirs) == (tmp_path.name, "self", [tmp_path])


def test_config_file(tmp_path):
    (tmp_path / "_state").mkdir()
    (tmp_path / "_state" / "artist.toml").write_text(
        'name = "mentored"\nlearns_from = "jamal"\nfeedback_dirs = ["~/a", "~/b"]\nsince = 2026-10-01\n'
    )
    cfg = load_config(tmp_path)
    assert cfg.learns_from == "jamal" and cfg.since == date(2026, 10, 1)
    assert cfg.feedback_dirs == [Path("~/a").expanduser(), Path("~/b").expanduser()]


def test_bad_learns_from_rejected(tmp_path):
    (tmp_path / "_state").mkdir()
    (tmp_path / "_state" / "artist.toml").write_text('learns_from = "everyone"\n')
    with pytest.raises(ValueError, match="learns_from"):
        load_config(tmp_path)


def test_self_artist_sees_only_its_own_critiques(two_artists):
    a, b = two_artists
    fb = collect(ArtistConfig("self-taught", "self", [a, b], date(2026, 10, 1)), a)
    assert [(f.stem, f.score, f.notes) for f in fb] == [("a1", 0.8, "critic says a1")]


def test_jamal_artist_sees_ratings_of_both_artists_since_start(two_artists):
    a, b = two_artists
    fb = collect(ArtistConfig("mentored", "jamal", [a, b], date(2026, 10, 1)), b)
    assert sorted((f.stem, f.score, f.notes) for f in fb) == [
        ("a1", 0.25, "not cyberpunk (2/5)"),
        ("b1", 0.75, "love it (4/5)"),
    ]


def test_apply_scores_means_and_resets_unscored():
    interests = [Interest("Cyber", 0.9, 15, "d"), Interest("Wire", 0.3, 4, "d")]
    fb = [reflect.Feedback("x", "Cyber", "p", 0.25, ""), reflect.Feedback("y", "Cyber", "p", 0.75, "")]
    out = apply_scores(interests, fb)
    assert (out[0].score, out[0].generations) == (0.5, 2)
    assert (out[1].score, out[1].generations) == (0.0, 0)


class FakeProvider:
    def __init__(self, result):
        self.result, self.calls = result, []

    def complete(self, system, user, response_model=None):
        self.calls.append((system, user))
        return self.result


def test_reflection_only_revises_directions_with_feedback(tmp_path):
    interests = [Interest("Cyber", 0.5, 2, "old cyber", settings={"steps": 30}), Interest("Wire", 0, 0, "old wire")]
    fb = [reflect.Feedback("a1", "Cyber", "p", 0.25, "not cyberpunk")]
    llm = FakeProvider(
        reflect.Reflection(
            directions=[
                reflect.Revision(title="Cyber", description="new cyber", change="ask for the future"),
                reflect.Revision(title="Wire", description="sneaky rewrite", change="changed anyway"),
            ]
        )
    )
    revised, log = reflect.apply(interests, reflect.revise(llm, interests, fb, "jamal"), fb)
    assert [i.description for i in revised] == ["new cyber", "old wire"]
    assert revised[0].settings == {"steps": 30}
    assert log == ["- **Cyber**: ask for the future"]
    system, user = llm.calls[0]
    assert "human viewer" in system and "not cyberpunk" in user and "- none" in user


def test_record_archives_logs_and_consumes(tmp_path):
    before = [Interest("Cyber", 0, 0, "old")]
    fb = [reflect.Feedback("a1", "Cyber", "p", 0.25, "")]
    assert [f.stem for f in reflect.unconsumed(tmp_path, fb)] == ["a1"]
    snap = reflect.record(
        tmp_path, "# Pre", before, before, ["- **Cyber**: x"], fb, datetime(2026, 10, 4, 9, 0).astimezone()
    )
    assert snap.name == "interests-2026-10-04-0900.md" and "old" in snap.read_text()
    assert "## 2026-10-04 (1 pieces of feedback)" in (tmp_path / "_state" / "history" / "changelog.md").read_text()
    assert reflect.unconsumed(tmp_path, fb) == []
