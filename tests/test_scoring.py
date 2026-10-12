import base64
import json
from unittest.mock import MagicMock

from local_first_common.testing import MockProvider

from artist_agent.scoring import SelfScoreModel, self_score


class TestSelfScore:
    def test_returns_parsed_score_and_notes(self):
        response = json.dumps({"score": 0.82, "notes": "Strong color use, avoids cliches."})
        provider = MockProvider(response=response)
        result = self_score(provider, "a creative direction", "a generated prompt", b"fake-png-bytes")
        assert result.score == 0.82
        assert result.notes == "Strong color use, avoids cliches."

    def test_passes_image_as_base64_with_response_model(self):
        image_bytes = b"fake-png-bytes"
        provider = MagicMock()
        provider.complete.return_value = SelfScoreModel(score=0.5, notes="ok")

        self_score(provider, "a direction", "a prompt", image_bytes)

        _, kwargs = provider.complete.call_args
        assert kwargs["images"] == [base64.b64encode(image_bytes).decode()]
        assert kwargs["response_model"] is SelfScoreModel

    def test_includes_direction_and_prompt_in_user_message(self):
        response = json.dumps({"score": 0.5, "notes": "ok"})
        provider = MockProvider(response=response)
        result = self_score(provider, "a creative direction", "a generated prompt", b"x")
        _, user = provider.calls[0]
        assert "a creative direction" in user
        assert "a generated prompt" in user
        assert "Intention:" not in user and result.lesson == ""

    def test_intention_and_technique_reach_the_critic_and_the_lesson_comes_back(self):
        response = json.dumps({"score": 0.5, "notes": "ok", "lesson": "FLUX holds black"})
        provider = MockProvider(response=response)
        intention = {
            "theme": "Wire",
            "idea": "a ring",
            "mode": "study",
            "hypothesis": "h",
            "why": "secret",
            "follows": None,
        }
        result = self_score(provider, "d", "p", b"x", intention, {"steps": 4, "seed": 9})
        _, user = provider.calls[0]
        assert "Intention:\ntheme: Wire\nidea: a ring\nmode: study\nhypothesis: h\ntechnique: steps 4\n" in user
        assert "secret" not in user and "seed" not in user
        assert result.lesson == "FLUX holds black"
        assert "lesson" in provider.calls[0][0]
