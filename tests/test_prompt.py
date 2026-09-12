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
