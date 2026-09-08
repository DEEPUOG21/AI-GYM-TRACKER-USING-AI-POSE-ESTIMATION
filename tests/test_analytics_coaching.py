import json
import httpx
import pytest
from gym_tracker.analytics import WorkoutSession
from gym_tracker.classification import Prediction
from gym_tracker.coaching import build_messages, coach


def session():
    value = WorkoutSession("test")
    value.record(1, "squat", True, Prediction("squat", 0.8, {}), True)
    value.record(2, None, pose_valid=False)
    value.finish()
    return value


def test_metrics_and_snapshot():
    value = session()
    result = value.snapshot()
    squat = next(m for m in result["exercises"] if m["exercise"] == "squat")
    assert squat["reps"] == 1 and squat["duration_seconds"] == 1
    assert squat["confidence"] == 0.8 and squat["rep_timestamps_seconds"] == [1]
    assert result["duration_seconds"] == 2 and result["unassigned_duration_seconds"] == 1
    assert result["started_at"] and result["ended_at"]
    squat["rep_timestamps_seconds"].append(9)
    assert value.snapshot() != result
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        value.record(3)


def test_invalid_time():
    value = WorkoutSession("test")
    for timestamp in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            value.record(timestamp)
    value.record(2)
    with pytest.raises(ValueError):
        value.record(1)


def test_coaching_telemetry():
    messages = build_messages(session().snapshot(), "What next?")
    assert messages[0]["role"] == "system"
    data = json.loads(messages[1]["content"])
    assert data["workout"]["duration_seconds"] == 2
    assert "session_id" not in data["workout"]
    assert "rep_timestamps_seconds" not in data["workout"]["exercises"][0]
    with pytest.raises(ValueError):
        build_messages(WorkoutSession("test").snapshot(), "Hello")


@pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
def test_provider_request(provider):
    def handler(request):
        data = json.loads(request.content)
        assert "workout" in data["messages"][-1]["content"]
        assert data["model"] == "test-model"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Session feedback"}],
                                         "choices": [{"message": {"content": "Session feedback"}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert coach(session().snapshot(), "What next?", provider=provider,
                     api_key="test", model="test-model", client=client) == "Session feedback"
