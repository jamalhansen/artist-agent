import random

import pytest

from artist_agent.interests import (
    EXPLORATION_FLOOR,
    Interest,
    parse_interests,
    pick_direction,
    record_outcome,
    render_interests,
)

SAMPLE = """\
# Artist Agent -- Current Interests

Evolving list, edit freely.

## Synesthetic wireframe
score: 0.72 (3 generations)

Neon vector line-art on near-black, geometric-organic forms.

## Cyberpunk without cliche
score: 0.55 (2 generations)

Avoid the genre's stock imagery. Look for ordinary, slightly-off details instead.
"""


class TestParseInterests:
    def test_parses_all_sections(self):
        _, interests = parse_interests(SAMPLE)
        assert [i.title for i in interests] == ["Synesthetic wireframe", "Cyberpunk without cliche"]

    def test_parses_score_and_generations(self):
        _, interests = parse_interests(SAMPLE)
        assert interests[0].score == 0.72
        assert interests[0].generations == 3

    def test_parses_description(self):
        _, interests = parse_interests(SAMPLE)
        assert "Neon vector line-art" in interests[0].description

    def test_captures_preamble(self):
        preamble, _ = parse_interests(SAMPLE)
        assert "Evolving list, edit freely." in preamble

    def test_missing_score_line_defaults_to_zero(self):
        text = "## New idea\n\nJust an idea, not scored yet.\n"
        _, interests = parse_interests(text)
        assert interests[0].score == 0.0
        assert interests[0].generations == 0

    def test_empty_file_returns_no_interests(self):
        preamble, interests = parse_interests("")
        assert interests == []
        assert preamble == ""

    def test_parses_model_and_settings(self):
        text = (
            "## Custom direction\n"
            "score: 0.40 (1 generations)\n"
            "model: some_other_model.ckpt\n"
            "steps: 30\n"
            "cfg: 7.5\n\n"
            "A description.\n"
        )
        _, interests = parse_interests(text)
        assert interests[0].model == "some_other_model.ckpt"
        assert interests[0].settings == {"steps": 30, "cfg": 7.5}

    def test_missing_model_and_settings_default_to_none_and_empty(self):
        _, interests = parse_interests(SAMPLE)
        assert interests[0].model is None
        assert interests[0].settings == {}

    def test_settings_do_not_leak_into_description(self):
        text = "## X\nscore: 0.0 (0 generations)\nmodel: foo.ckpt\nsteps: 20\n\nreal description\n"
        _, interests = parse_interests(text)
        assert "model:" not in interests[0].description
        assert "steps:" not in interests[0].description
        assert interests[0].description == "real description"

    def test_parses_signal(self):
        text = "## X\nscore: 0.0 (0 generations)\nsignal: content-discovery\n\ndesc\n"
        _, interests = parse_interests(text)
        assert interests[0].signal == "content-discovery"

    def test_missing_signal_defaults_to_none(self):
        _, interests = parse_interests(SAMPLE)
        assert interests[0].signal is None


class TestRenderInterests:
    def test_round_trips_through_parse(self):
        preamble, interests = parse_interests(SAMPLE)
        rendered = render_interests(interests, preamble)
        preamble2, interests2 = parse_interests(rendered)
        assert interests2 == interests
        assert preamble2 == preamble

    def test_omits_preamble_block_when_empty(self):
        rendered = render_interests(
            [Interest("Only one", 0.5, 1, "desc")], preamble=""
        )
        assert rendered.startswith("## Only one")

    def test_renders_model_and_settings_when_present(self):
        interest = Interest(
            "Custom", 0.4, 1, "desc", model="some_other_model.ckpt", settings={"steps": 30, "cfg": 7.5}
        )
        rendered = render_interests([interest])
        assert "model: some_other_model.ckpt" in rendered
        assert "steps: 30" in rendered
        assert "cfg: 7.5" in rendered

    def test_omits_model_and_settings_lines_when_absent(self):
        rendered = render_interests([Interest("Plain", 0.5, 1, "desc")])
        assert "model:" not in rendered
        assert "steps:" not in rendered

    def test_round_trips_model_and_settings(self):
        interest = Interest(
            "Custom", 0.4, 1, "desc", model="some_other_model.ckpt", settings={"steps": 30, "cfg": 7.5}
        )
        rendered = render_interests([interest])
        _, parsed_back = parse_interests(rendered)
        assert parsed_back[0] == interest

    def test_renders_and_round_trips_signal(self):
        interest = Interest("Headline-reactive", 0.0, 0, "desc", signal="content-discovery")
        rendered = render_interests([interest])
        assert "signal: content-discovery" in rendered
        _, parsed_back = parse_interests(rendered)
        assert parsed_back[0] == interest


class TestPickDirection:
    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="no interests"):
            pick_direction([])

    def test_higher_score_wins_more_often(self):
        strong = Interest("Strong", 10.0, 5, "")
        weak = Interest("Weak", 0.0, 5, "")
        rng = random.Random(42)
        picks = [pick_direction([strong, weak], rng).title for _ in range(200)]
        assert picks.count("Strong") > picks.count("Weak")

    def test_never_scored_interest_can_still_be_picked(self):
        # A 0.0-score interest must not be permanently locked out -- the
        # EXPLORATION_FLOOR exists exactly so this can happen.
        assert EXPLORATION_FLOOR > 0
        never_scored = Interest("Fresh", 0.0, 0, "")
        established = Interest("Established", 0.9, 10, "")
        rng = random.Random(1)
        picks = [pick_direction([never_scored, established], rng).title for _ in range(500)]
        assert picks.count("Fresh") > 0


class TestRecordOutcome:
    def test_updates_running_average(self):
        interests = [Interest("A", 0.5, 1, "desc")]
        updated = record_outcome(interests, "A", 0.9)
        assert updated[0].score == pytest.approx(0.7)
        assert updated[0].generations == 2

    def test_first_observation_sets_score_directly(self):
        interests = [Interest("A", 0.0, 0, "desc")]
        updated = record_outcome(interests, "A", 0.8)
        assert updated[0].score == pytest.approx(0.8)
        assert updated[0].generations == 1

    def test_leaves_other_interests_untouched(self):
        interests = [Interest("A", 0.5, 1, ""), Interest("B", 0.3, 2, "")]
        updated = record_outcome(interests, "A", 0.9)
        assert updated[1] == interests[1]

    def test_unknown_title_leaves_list_unchanged(self):
        interests = [Interest("A", 0.5, 1, "")]
        updated = record_outcome(interests, "Nonexistent", 0.9)
        assert updated == interests
