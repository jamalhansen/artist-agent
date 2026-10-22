"""Variety: a theme rests after a streak, and a wild card breaks a run of failures."""

import random

from artist_agent import catalog, reflect, studio, variety
from artist_agent.feedback import Feedback
from artist_agent.interests import Interest


def _piece(theme, score, model="sdxl.ckpt", wild=False, stem=None):
    return Feedback(
        stem=stem or f"p-{theme}-{score}",
        interest=theme,
        prompt="p",
        score=score,
        notes="",
        model=model,
        intention={"wild_card": {"reason": "chance"}} if wild else {},
    )


def _catalog():
    return [
        catalog.ImageModel("sdxl.ckpt", "SDXL", "", {}),
        catalog.ImageModel("flux.ckpt", "FLUX", "", {}),
        catalog.ImageModel("z.ckpt", "Z", "", {}),
    ]


def _interests():
    return [Interest("Wire", 0.1, 9, "lines"), Interest("Cyber", 0.3, 4, "streets")]


def test_streak_counts_the_latest_run():
    own = [_piece("Cyber", 0.5), _piece("Wire", 0.1), _piece("Wire", 0.2)]
    assert variety.streak(own) == ("Wire", 2)
    assert variety.resting_theme(own) == "Wire"
    assert variety.resting_theme(own[:2]) is None
    assert variety.streak([]) == ("", 0)


def test_failing_needs_a_full_run_of_low_rated_pieces():
    low = [_piece("Wire", 0.0), _piece("Wire", 0.25), _piece("Wire", 0.0)]
    assert variety.failing(low)
    assert not variety.failing(low[:2])
    assert not variety.failing([*low[:2], _piece("Wire", 0.75)])
    # Unrated pieces are skipped, not counted as failures.
    assert variety.failing([*low, _piece("Wire", None)])


def test_an_unrated_wild_card_holds_off_the_next_one():
    low = [_piece("Wire", 0.0), _piece("Wire", 0.0), _piece("Wire", 0.0)]
    assert not variety.failing([*low, _piece("Wild card: x", None, wild=True)])
    # Once rated, it counts like any other piece.
    assert variety.failing([*low[1:], _piece("Wild card: x", 0.0, wild=True)])


def test_draw_avoids_recent_media_and_uses_config_lists():
    own = [_piece("Wire", 0.0, model="sdxl.ckpt"), _piece("Wire", 0.0, model="z.ckpt")]
    lists = {"subjects": ["a heron"], "materials": ["clay"], "movements": ["Fauvism"]}
    wild = variety.draw(own, _catalog(), random.Random(1), "failing", lists)
    assert (wild.subject, wild.material, wild.movement, wild.medium) == ("a heron", "clay", "Fauvism", "flux.ckpt")
    assert wild.record()["reason"] == "failing"


def test_wild_card_today_on_failure_or_by_chance():
    low = [_piece("Wire", 0.0)] * 3
    failed = variety.wild_card_today(low, _catalog(), random.Random(0), chance=0.0)
    assert failed is not None and failed.reason == "failing"
    fine = [_piece("Wire", 0.9)] * 3
    assert variety.wild_card_today(fine, _catalog(), random.Random(0), chance=0.0) is None
    lucky = variety.wild_card_today(fine, _catalog(), random.Random(0), chance=1.0)
    assert lucky is not None and lucky.reason == "chance"


def test_resting_theme_is_left_off_the_list():
    user = studio.build_user("practice", _interests(), {}, [], "jamal", resting="Wire")
    assert '"Wire" is resting today' in user
    assert "## Wire" not in user
    assert "## Cyber" in user


def test_wild_card_brief_hides_themes_recent_work_and_practice():
    wild = variety.WildCard("failing", "a heron", "clay", "Fauvism", "flux.ckpt")
    user = studio.build_wild_card_user(wild, _interests(), "Wednesday 2026-10-21")
    assert "a heron" in user and "clay" in user and "Fauvism" in user and "flux.ckpt" in user
    assert "## Wire" not in user and "Recent work" not in user and "practice" not in user.lower()
    assert '"Wire"' in user  # named only as a title not to reuse


def test_resolve_takes_the_drawn_medium_and_no_theme_on_a_wild_card():
    pinned = [Interest("Wire", 0.1, 9, "lines", model="z.ckpt", settings={"cfg": 2.0})]
    intention = studio.Intention.model_validate(
        {"theme": "Wire", "idea": "i", "palette": "p", "feel": "f", "medium": "sdxl.ckpt", "prompt": "x"}
    )
    wild = variety.WildCard("chance", "s", "m", "v", "flux.ckpt")
    choice = studio.resolve(intention, pinned, _catalog(), {}, "default.ckpt", random.Random(0), wild_card=wild)
    assert choice.model == "flux.ckpt"
    assert choice.how == "drawn (wild card)"
    assert choice.interest is None
    assert "cfg" not in choice.settings or choice.settings.get("cfg") != 2.0


def test_reflection_lists_one_offs_and_refuses_long_descriptions():
    interests = _interests()
    feedback = [_piece("Wire", 0.0), _piece("Wild card: a heron", 0.75, wild=True)]
    user = reflect.build_user(interests, feedback)
    assert "One-offs and wild cards" in user
    assert "Wild card: a heron: score 0.75" in user

    long = " ".join(["word"] * (reflect.REFUSE_OVER_WORDS + 1))
    reflection = reflect.Reflection.model_validate(
        {"directions": [{"title": "Wire", "description": long, "change": "longer"}]}
    )
    out, log = reflect.apply(interests, reflection, feedback)
    assert next(i for i in out if i.title == "Wire").description == "lines"
    assert any("description refused" in line for line in log)


def test_reflection_flags_ban_phrasing_but_keeps_the_rewrite():
    interests = _interests()
    feedback = [_piece("Wire", 0.0)]
    reflection = reflect.Reflection.model_validate(
        {"directions": [{"title": "Wire", "description": "Roots grown through circuitry. Avoid glow.", "change": "c"}]}
    )
    out, log = reflect.apply(interests, reflection, feedback)
    assert next(i for i in out if i.title == "Wire").description.startswith("Roots")
    assert any("still bans (avoid)" in line for line in log)
    assert reflect.ban_phrases("Reach for depth; cannot say where it ends") == []
