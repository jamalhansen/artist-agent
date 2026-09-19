from artist_agent.signals import CurrentEvent, fetch_current_event


def _write_inbox_item(dir_, filename, *, source_type="content-discovery-agent", captured, title, source_url="https://example.com"):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / filename).write_text(
        f"---\nsource_type: {source_type}\ncaptured: {captured}\nsource_url: {source_url}\n---\n\n# {title}\n\nbody text\n"
    )


class TestFetchCurrentEvent:
    def test_missing_inbox_dir_returns_none(self, tmp_path):
        assert fetch_current_event(tmp_path / "does-not-exist") is None

    def test_empty_inbox_returns_none(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert fetch_current_event(tmp_path) is None

    def test_ignores_items_from_other_sources(self, tmp_path):
        _write_inbox_item(
            tmp_path, "a.md", source_type="voice-memo", captured="2026-09-17", title="Not this one"
        )
        assert fetch_current_event(tmp_path) is None

    def test_returns_most_recently_captured_item(self, tmp_path):
        _write_inbox_item(tmp_path, "old.md", captured="2026-09-10", title="Older item")
        _write_inbox_item(tmp_path, "new.md", captured="2026-09-17", title="Newest item")
        event = fetch_current_event(tmp_path)
        assert event == CurrentEvent(title="Newest item", source_url="https://example.com")

    def test_malformed_file_is_skipped_not_fatal(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "broken.md").write_bytes(b"\xff\xfe not valid frontmatter at all {{{")
        _write_inbox_item(tmp_path, "good.md", captured="2026-09-17", title="Fine item")
        event = fetch_current_event(tmp_path)
        assert event.title == "Fine item"

    def test_item_without_captured_date_is_skipped(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "no-date.md").write_text(
            "---\nsource_type: content-discovery-agent\n---\n\n# No date item\n"
        )
        assert fetch_current_event(tmp_path) is None
