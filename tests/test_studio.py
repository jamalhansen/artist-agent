"""The studio session: the artist's daily decisions, and how they meet the catalog's limits."""

import json
import random

from local_first_common.testing import MockProvider

from artist_agent import catalog, studio
from artist_agent.feedback import Feedback, ModelRecord
from artist_agent.interests import Interest
from artist_agent.practice import DEFAULT_PRACTICE, SECTIONS, is_complete


def _catalog():
    return [
        catalog.ImageModel("sdxl.ckpt", "SDXL", "dense", {"steps": 30, "cfg": 7.0}, steps_range=(20, 40)),
        catalog.ImageModel(
            "flux.ckpt",
            "FLUX",
            "flat",
            {"steps": 4, "cfg": 4.5, "width": 1024, "height": 576, "sampler": "Euler A Trailing"},
            steps_range=(1, 8),
            cfg_range=(1.0, 5.0),
            samplers=["Euler A Trailing", "DPM++ 2M Karras"],
            sizes=["1024x576", "1024x1024"],
            minutes=2.5,
        ),
    ]


def _intention(**over):
    base = {
        "theme": "Wire",
        "idea": "one ring",
        "palette": "cyan, white on black",
        "feel": "a held breath",
        "medium": "flux.ckpt",
        "mode": "study",
        "hypothesis": "does FLUX hold flat black",
        "prompt": "a single thin cyan ring on black",
    }
    return studio.Intention(**{**base, **over})


def test_technique_stays_inside_the_mediums_room():
    cat = _catalog()
    t = catalog.technique_for(cat, "flux.ckpt", {"steps": 40, "cfg": 0.2, "sampler": "Magic", "size": "2048x2048"})
    assert t == {"steps": 8, "cfg": 1.0, "width": 1024, "height": 576, "sampler": "Euler A Trailing"}
    t = catalog.technique_for(cat, "flux.ckpt", {"steps": 6, "sampler": "DPM++ 2M Karras", "size": "1024x1024"})
    assert (t["steps"], t["sampler"], t["width"], t["height"]) == (6, "DPM++ 2M Karras", 1024, 1024)
    # no range listed for a key = any value accepted; unknown medium = whatever was asked
    assert catalog.technique_for(cat, "sdxl.ckpt", {"cfg": 12.0, "seed": 1})["cfg"] == 12.0
    assert "seed" not in catalog.technique_for(cat, "sdxl.ckpt", {"seed": 1})
    assert catalog.technique_for(cat, "nope.ckpt", {"steps": 3}) == {"steps": 3}


def test_describe_shows_the_room_and_the_time():
    text = catalog.describe(_catalog())
    assert "Room to explore: steps 1-8; cfg 1-5; samplers Euler A Trailing / DPM++ 2M Karras; sizes 1024x576" in text
    assert "About 2.5 min per image" in text
    assert "Room to explore: steps 20-40." in text


def test_catalog_loads_ranges(monkeypatch, tmp_path):
    cfg = tmp_path / "artist-agent.toml"
    cfg.write_text(
        '[[image_models]]\nfile = "flux.ckpt"\nsteps = 4\nsteps_range = [8, 1]\ncfg_range = [1, 5]\n'
        'samplers = ["Euler A Trailing"]\nsizes = ["1024x576"]\nminutes = 2\n'
    )
    monkeypatch.setattr(catalog, "load_config", lambda tool: __import__("tomllib").loads(cfg.read_text()))
    m = catalog.load_catalog()[0]
    assert (m.steps_range, m.cfg_range, m.samplers, m.sizes, m.minutes) == (
        (1, 8),
        (1.0, 5.0),
        ["Euler A Trailing"],
        ["1024x576"],
        2.0,
    )


def test_resolve_uses_the_chosen_medium_and_records_a_seed():
    interests = [
        Interest("Wire", 0.3, 2, "d"),
        Interest("Cyber", 0.3, 2, "d", model="sdxl.ckpt", settings={"cfg": 9.0}),
    ]
    c = studio.resolve(_intention(steps=6, size="1024x1024"), interests, _catalog(), {}, "x.ckpt", random.Random(1))
    assert (c.model, c.how, c.interest.title) == ("flux.ckpt", "chosen", "Wire")
    assert c.settings["steps"] == 6 and c.settings["width"] == 1024 and c.settings["height"] == 1024
    assert isinstance(c.settings["seed"], int)
    # a pinned theme wins over the artist's medium, and its own setting lines go on top
    c = studio.resolve(_intention(theme="cyber", steps=6), interests, _catalog(), {}, "x.ckpt")
    assert (c.model, c.how, c.settings["cfg"]) == ("sdxl.ckpt", "pinned by Cyber", 9.0)
    # a medium that isn't in the catalog falls back to the record-weighted draw
    c = studio.resolve(_intention(medium="dream.ckpt"), interests, _catalog(), {}, "x.ckpt", random.Random(1))
    assert c.model in ("sdxl.ckpt", "flux.ckpt") and c.how.startswith("drawn (")
    # a one-off theme has no interest
    c = studio.resolve(_intention(theme="Rust studies"), interests, _catalog(), {}, "x.ckpt")
    assert c.interest is None and c.model == "flux.ckpt"
    # no catalog at all: the tool default
    assert studio.resolve(_intention(), interests, [], {}, "x.ckpt").model == "x.ckpt"


def test_intention_record_and_technique():
    it = _intention(steps=6, follows="2026-10-02-ring", mode="weird")
    assert it.technique() == {"steps": 6}
    rec = it.record()
    assert rec["mode"] == "piece" and rec["follows"] == "2026-10-02-ring" and rec["theme"] == "Wire"
    assert "prompt" not in rec


def test_build_user_shows_practice_themes_record_and_recent_work():
    interests = [Interest("Wire", 0.3, 2, "thin lines"), Interest("Cyber", 0, 0, "the future", model="sdxl.ckpt")]
    record = {"Wire": {"flux.ckpt": ModelRecord(0.62, 1), "sdxl.ckpt": ModelRecord(0.31, 4)}}
    recent = [
        Feedback(
            "a",
            "Wire",
            "p",
            0.4,
            "muddy",
            model="sdxl.ckpt",
            settings={"steps": 30, "seed": 1},
            lesson="L1",
            intention={"theme": "Wire", "mode": "study", "idea": "a ring", "palette": "cyan", "hypothesis": "h"},
        ),
        Feedback("b", "Cyber", "p", None, "", model="flux.ckpt"),
    ]
    user = studio.build_user(DEFAULT_PRACTICE, interests, record, recent, "self", "Some headline", "Monday 2026-10-12")
    assert user.startswith("Today: Monday 2026-10-12")
    assert "## Stretching toward" in user
    assert "## Wire\nscore 0.30 over 2 pieces; by medium: flux.ckpt 0.62 over 1; sdxl.ckpt 0.31 over 4." in user
    assert "by medium: no record yet. Pinned to sdxl.ckpt." in user
    assert (
        "- a: Wire (study) | sdxl.ckpt (steps 30)\n  idea: a ring | palette: cyan\n  hypothesis: h\n  score 0.40: muddy\n  lesson: L1"
        in user
    )
    assert "- b: Cyber | flux.ckpt (defaults)\n  not yet rated" in user
    assert "Some headline" in user
    # the mentored artist never sees the critic's lesson
    assert "lesson: L1" not in studio.build_user(DEFAULT_PRACTICE, interests, record, recent, "jamal")


def test_plan_parses_the_intention_and_names_the_source():
    provider = MockProvider(response=json.dumps(_intention(steps=6).model_dump()))
    it = studio.plan(provider, DEFAULT_PRACTICE, [], {}, [], "jamal", _catalog(), today="x")
    assert it.medium == "flux.ckpt" and it.steps == 6
    system, _ = provider.calls[0]
    assert "viewer's rating" in system and "flux.ckpt: FLUX. flat Defaults: steps 4" in system
    assert "one piece in three should be a study" in system


def test_practice_file_sections():
    assert is_complete(DEFAULT_PRACTICE)
    assert not is_complete(DEFAULT_PRACTICE.replace("## Lessons", "## Notes"))
    assert len(SECTIONS) == 6
