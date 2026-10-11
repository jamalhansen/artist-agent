from local_first_common.testing import MockProvider

from artist_agent.prompt import compose_prompt, system_prompt


class TestComposePrompt:
    def test_returns_stripped_response(self):
        provider = MockProvider(response="  \nA neon wireframe city at dawn.\n\n")
        result = compose_prompt(provider, "Synesthetic wireframe: neon vector line-art.")
        assert result == "A neon wireframe city at dawn."

    def test_passes_interest_description_as_user_message(self):
        provider = MockProvider(response="a prompt")
        compose_prompt(provider, "the interest description")
        system, user = provider.calls[0]
        assert system == system_prompt()
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
        assert "not depict" in system_prompt().lower() or "literal" in system_prompt().lower()

    def test_writer_is_told_which_image_model(self):
        provider = MockProvider(response="a prompt")
        compose_prompt(provider, "d", image_model="z_image_turbo_1.0_q6p.ckpt")
        system, _ = provider.calls[0]
        assert "z_image_turbo_1.0_q6p.ckpt" in system and "Stable Diffusion XL" not in system
        assert "Stable Diffusion XL" in system_prompt()
