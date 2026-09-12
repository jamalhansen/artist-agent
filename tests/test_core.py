from datetime import UTC, datetime

import frontmatter

from artist_agent.core import (
    Item,
    derive_title,
    item_stem,
    render_item_note,
    resolve_ai_artist_dir,
    slugify,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Synesthetic Wireframe City") == "synesthetic-wireframe-city"

    def test_truncates_to_max_words(self):
        assert slugify("one two three four five six seven eight") == "one-two-three-four-five-six"

    def test_truncates_to_max_chars(self):
        result = slugify("a " * 40)
        assert len(result) <= 50

    def test_empty_returns_untitled(self):
        assert slugify("") == "untitled"


class TestResolveAiArtistDir:
    def test_expands_default(self):
        path = resolve_ai_artist_dir(None)
        assert not str(path).startswith("~")

    def test_uses_explicit_path(self, tmp_path):
        assert resolve_ai_artist_dir(str(tmp_path)) == tmp_path


class TestItemStem:
    def test_combines_date_and_slug(self):
        item = Item(
            generated_at=datetime(2026, 9, 12, 9, 0, 0, tzinfo=UTC),
            prompt="a neon wireframe city at dawn",
            interest_title="Synesthetic wireframe",
            settings={},
            image_filename="",
            self_score=0.8,
            self_score_notes="good",
            models={"prompt": "ollama/llama3.2:3b", "image": "sdxl.ckpt", "critique": "anthropic/claude"},
        )
        assert item_stem(item) == "2026-09-12-a-neon-wireframe-city-at-dawn"


class TestDeriveTitle:
    def test_short_prompt_returned_as_is(self):
        assert derive_title("a neon wireframe city") == "a neon wireframe city"

    def test_long_prompt_truncated_with_ellipsis(self):
        prompt = " ".join(["word"] * 20)
        title = derive_title(prompt)
        assert title == "word " * 11 + "word…"
        assert len(title.split()) == 12

    def test_exactly_max_words_not_truncated(self):
        prompt = " ".join(["word"] * 12)
        assert derive_title(prompt) == prompt
        assert "…" not in derive_title(prompt)


class TestRenderItemNote:
    def _make_item(self, **overrides):
        defaults = {
            "generated_at": datetime(2026, 9, 12, 9, 0, 0, tzinfo=UTC),
            "prompt": "a neon wireframe city at dawn",
            "interest_title": "Synesthetic wireframe",
            "settings": {"steps": 30},
            "image_filename": "2026-09-12-a-neon-wireframe-city-at.png",
            "self_score": 0.8,
            "self_score_notes": "Strong color use, avoids cliches.",
            "models": {
                "prompt": "ollama/llama3.2:3b",
                "image": "sd_xl_base_1.0_f16.ckpt",
                "critique": "anthropic/claude-haiku-4-5-20251001",
            },
        }
        defaults.update(overrides)
        return Item(**defaults)

    def test_produces_valid_frontmatter(self):
        content = render_item_note(self._make_item(), "../images/foo.png")
        post = frontmatter.loads(content)
        assert post["title"] == "a neon wireframe city at dawn"
        assert post["prompt"] == "a neon wireframe city at dawn"
        assert post["interest"] == "Synesthetic wireframe"
        assert post["settings"] == {"steps": 30}
        assert post["models"] == {
            "prompt": "ollama/llama3.2:3b",
            "image": "sd_xl_base_1.0_f16.ckpt",
            "critique": "anthropic/claude-haiku-4-5-20251001",
        }
        assert post["self_score"] == 0.8
        assert post["human_score"] is None
        assert post["status"] == "new"

    def test_body_uses_plain_markdown_image_syntax(self):
        content = render_item_note(self._make_item(), "../images/foo.png")
        post = frontmatter.loads(content)
        assert "![Synesthetic wireframe](../images/foo.png)" in post.content
        assert "[[" not in post.content  # never an Obsidian wikilink/embed

    def test_unreviewed_human_notes_placeholder(self):
        content = render_item_note(self._make_item(), "../images/foo.png")
        post = frontmatter.loads(content)
        assert "_not yet reviewed_" in post.content

    def test_survives_prompt_with_quotes_and_special_chars(self):
        item = self._make_item(prompt='a "glitchy" city — synesthesia: colors as sound')
        content = render_item_note(item, "../images/foo.png")
        post = frontmatter.loads(content)
        assert post["prompt"] == 'a "glitchy" city — synesthesia: colors as sound'
