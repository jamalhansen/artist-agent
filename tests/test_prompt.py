from local_first_common.testing import MockProvider

from artist_agent.prompt import SYSTEM_PROMPT, compose_prompt


class TestComposePrompt:
    def test_returns_stripped_response(self):
        provider = MockProvider(response="  \nA neon wireframe city at dawn.\n\n")
        result = compose_prompt(provider, "Synesthetic wireframe: neon vector line-art.")
        assert result == "A neon wireframe city at dawn."

    def test_passes_interest_description_as_user_message(self):
        provider = MockProvider(response="a prompt")
        compose_prompt(provider, "the interest description")
        system, user = provider.calls[0]
        assert system == SYSTEM_PROMPT
        assert user == "the interest description"

    def test_no_current_event_leaves_user_message_unchanged(self):
        provider = MockProvider(response="a prompt")
        compose_prompt(provider, "the interest description", current_event=None)
        _, user = provider.calls[0]
        assert user == "the interest description"

    def test_current_event_appended_to_user_message(self):
        provider = MockProvider(response="a prompt")
        compose_prompt(provider, "the interest description", current_event="Some real headline")
        _, user = provider.calls[0]
        assert "the interest description" in user
        assert "Some real headline" in user

    def test_system_prompt_instructs_not_to_depict_current_event_literally(self):
        assert "not depict" in SYSTEM_PROMPT.lower() or "literal" in SYSTEM_PROMPT.lower()
