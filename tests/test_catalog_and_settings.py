"""Model catalog, sampler/clip-skip plumbing, and reflection moving a direction between models."""

import json
import random
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from artist_agent import catalog, reflect
from artist_agent.drawthings import DrawThingsError, config_overrides, generate_image
from artist_agent.feedback import ArtistConfig, ModelRecord, collect, track_record
from artist_agent.interests import EXPLORATION_FLOOR, Interest, parse_interests, pick_model, render_interests


def _catalog():
    return [
        catalog.ImageModel("sdxl.ckpt", "SDXL", "default"),
        catalog.ImageModel("flux.ckpt", "FLUX", "april look", {"steps": 4, "cfg": 4.5, "sampler": "Euler A Trailing"}),
    ]


def test_catalog_loads_from_tool_config(monkeypatch, tmp_path):
    cfg = tmp_path / "artist-agent.toml"
    cfg.write_text(
        '[[image_models]]\nfile = "flux.ckpt"\nname = "FLUX"\nnotes = "n"\nsteps = 4\ncfg = 4.5\n'
        'sampler = "Euler A Trailing"\nclip_skip = 2\nbogus = 1\n[[image_models]]\nname = "no file"\n'
    )
    monkeypatch.setattr(catalog, "load_config", lambda tool: __import__("tomllib").loads(cfg.read_text()))
    models = catalog.load_catalog()
    assert [m.file for m in models] == ["flux.ckpt"]
    assert models[0].settings == {"steps": 4, "cfg": 4.5, "sampler": "Euler A Trailing", "clip_skip": 2}
    assert catalog.defaults_for(models, "flux.ckpt") == models[0].settings
    assert catalog.defaults_for(models, "other.ckpt") == {}
    assert "flux.ckpt: FLUX. n Defaults: steps 4" in catalog.describe(models)
    assert "no catalog" in catalog.describe([])


def test_sampler_and_clip_skip_become_a_config_json_override():
    assert config_overrides({"sampler": "Euler A Trailing", "clip_skip": 2, "steps": 4}) == {
        "sampler": 10,
        "clipSkip": 2,
    }
    assert config_overrides({"sampler": 1}) == {"sampler": 1}
    assert config_overrides({"steps": 4}) == {}
    with pytest.raises(DrawThingsError, match="unknown sampler"):
        config_overrides({"sampler": "Magic"})


def test_generate_passes_overrides_and_flags():
    def _run(cmd, capture_output, text, timeout, check):
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("artist_agent.drawthings.subprocess.run", side_effect=_run) as run:
        generate_image(
            "p", "flux.ckpt", {"steps": 4, "cfg": 4.5, "sampler": "Euler A Trailing", "clip_skip": 2, "seed": 7}
        )
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--steps") + 1] == "4" and cmd[cmd.index("--seed") + 1] == "7"
    assert json.loads(cmd[cmd.index("--config-json") + 1]) == {"sampler": 10, "clipSkip": 2}


def test_interests_round_trip_sampler_and_clip_skip():
    i = Interest("Wire", 0.3, 1, "d", model="flux.ckpt", settings={"sampler": "Euler A Trailing", "clip_skip": 2})
    text = render_interests([i])
    assert "sampler: Euler A Trailing\nclip_skip: 2" in text
    _, back = parse_interests(text)
    assert back[0] == i and "sampler" not in back[0].description


def test_feedback_carries_model_and_settings(tmp_path):
    d = tmp_path / "items"
    d.mkdir()
    (d / "a.md").write_text(
        "---\ngenerated_at: '2026-10-02T09:00:00-05:00'\ninterest: Wire\nprompt: p\nself_score: 0.4\n"
        "models:\n  image: flux.ckpt\n  prompt: x\nsettings:\n  steps: 4\n  seed: 7\n---\n**Self-critique (0.40):** meh\n"
    )
    (d / "b.md").write_text(
        "---\ngenerated_at: '2026-10-02T09:00:00-05:00'\ninterest: Wire\nprompt: p\nself_score: 0.5\n---\n"
    )
    fb = collect(ArtistConfig("s", "self", [tmp_path]), tmp_path)
    assert (fb[0].model, fb[0].settings) == ("flux.ckpt", {"steps": 4, "seed": 7})
    assert (fb[1].model, fb[1].settings) == (None, {})


def test_build_user_shows_what_each_direction_and_piece_rendered_with():
    interests = [
        Interest("Wire", 0.3, 1, "d", model="flux.ckpt", settings={"steps": 4, "seed": 1}),
        Interest("Cyber", 0, 0, "d"),
    ]
    fb = [reflect.Feedback("a", "Wire", "p", 0.4, "meh", model="sdxl.ckpt", settings={"seed": 9})]
    record = {"Wire": {"sdxl.ckpt": ModelRecord(0.31, 12, {"steps": 30}), "flux.ckpt": ModelRecord(0.62, 1)}}
    user = reflect.build_user(interests, fb, default_model="sdxl.ckpt", record=record)
    assert "Renders with: flux.ckpt (steps 4), pinned" in user and "seed" not in user
    assert "Renders with: a daily draw from the catalog" in user
    assert "rendered with sdxl.ckpt (model defaults)" in user
    assert (
        "Track record by model (all pieces so far): flux.ckpt 0.62 over 1 piece (model defaults); sdxl.ckpt 0.31 over 12 pieces (steps 30)"
        in user
    )
    assert "Track record by model: none yet" in user  # Cyber


def test_reflection_moves_one_direction_to_a_catalog_model_and_merges_settings():
    interests = [Interest("Wire", 0.3, 2, "old", settings={"steps": 30}), Interest("Cyber", 0.3, 2, "old")]
    fb = [reflect.Feedback("a", "Wire", "p", 0.2, "soft"), reflect.Feedback("b", "Cyber", "p", 0.2, "dull")]
    reflection = reflect.Reflection(
        directions=[
            reflect.Revision(title="Wire", description="old", change="unchanged", model="flux.ckpt"),
            reflect.Revision(title="Cyber", description="old", change="unchanged", model="flux.ckpt", cfg=3.0),
        ]
    )
    out, log = reflect.apply(interests, reflection, fb, _catalog())
    assert (out[0].model, out[0].settings) == ("flux.ckpt", {})  # catalog defaults apply at run time
    assert (out[1].model, out[1].settings) == (None, {"cfg": 3.0})  # second model move refused, settings kept
    assert log == ["- **Wire**: now renders with flux.ckpt", "- **Cyber**: settings cfg 3.0"]
    # an unknown model is ignored; a seed is never taken
    r = reflect.Reflection(
        directions=[reflect.Revision(title="Wire", description="old", change="unchanged", model="nope.ckpt")]
    )
    out, log = reflect.apply(interests, r, fb, _catalog())
    assert out[0] == interests[0] and log == []
    assert "seed" not in reflect.Revision(title="x", description="d", change="c", steps=4).settings()


def test_system_prompt_lists_the_catalog():
    class P:
        def complete(self, system, user, response_model=None):
            self.system = system
            return reflect.Reflection(directions=[])

    p = P()
    reflect.revise(p, [], [], "self", _catalog())  # type: ignore[arg-type]
    assert "flux.ckpt: FLUX. april look Defaults: steps 4, cfg 4.5, sampler Euler A Trailing." in p.system
    assert "pin one direction per reflection" in p.system


def test_track_record_is_per_direction_and_model_over_all_feedback():
    fb = [
        reflect.Feedback("a", "Wire", "p", 0.2, "", model="sdxl.ckpt", settings={"seed": 1, "steps": 30}),
        reflect.Feedback("b", "Wire", "p", 0.4, "", model="sdxl.ckpt", settings={"seed": 2, "steps": 20}),
        reflect.Feedback("c", "Wire", "p", 0.6, "", model="flux.ckpt"),
        reflect.Feedback("d", "Cyber", "p", 0.9, "", model="sdxl.ckpt"),
        reflect.Feedback("e", "Cyber", "p", 0.1, ""),  # no model recorded: skipped
    ]
    rec = track_record(fb)
    sdxl = rec["Wire"]["sdxl.ckpt"]
    assert (sdxl.mean, sdxl.count, sdxl.last_settings) == (pytest.approx(0.3), 2, {"steps": 20})  # seed dropped
    assert rec["Wire"]["flux.ckpt"] == ModelRecord(0.6, 1, {})
    assert rec["Cyber"] == {"sdxl.ckpt": ModelRecord(0.9, 1, {})}


def test_pick_model_weighs_tried_by_score_and_untried_like_the_best():
    rng = random.Random(7)
    draws = [pick_model(["sdxl", "flux", "z"], {"sdxl": 0.3, "flux": 0.6}, rng) for _ in range(3000)]
    n = {m: draws.count(m) for m in ("sdxl", "flux", "z")}
    # weights 0.45 : 0.75 : 0.75 -> z (untried) is picked about as often as flux, more than sdxl
    assert n["flux"] > n["sdxl"] and abs(n["z"] - n["flux"]) < 250
    assert pick_model(["only"], {}, rng) == "only"
    # with nothing tried every model is an even draw at the floor
    even = [pick_model(["a", "b"], {}, rng) for _ in range(2000)]
    assert abs(even.count("a") - 1000) < 150 and EXPLORATION_FLOOR > 0
    with pytest.raises(ValueError):
        pick_model([], {}, rng)
