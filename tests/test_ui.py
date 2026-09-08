import importlib.util
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("streamlit") is None,
                                reason="Streamlit runtime not installed")


def test_dashboard_and_empty_coach():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("streamlit_app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Overview"
    app.sidebar.radio[0].set_value("AI Coach").run()
    assert not app.exception
    assert "workout first" in app.info[0].value
    app.sidebar.radio[0].set_value("Video Analysis").run()
    assert not app.exception
    assert app.button[0].disabled


def workout():
    from gym_tracker.analytics import WorkoutSession
    session = WorkoutSession("test")
    session.record(1, "squat", pose_valid=True)
    session.finish()
    return session.snapshot()


def test_video_results_match_inputs_and_clear_on_failure(monkeypatch):
    from io import BytesIO
    from pathlib import Path
    import gym_tracker.ui.app as ui
    from streamlit.testing.v1 import AppTest
    video = Path("assets/videos/bicep_curl_demo.mp4").read_bytes()
    upload = BytesIO(video)
    upload.name = "demo.mp4"
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **kw: upload)
    monkeypatch.setattr(ui, "make_tracker", lambda *a: object())

    def analyze(source, output, *args):
        output.write_bytes(video)
        return workout()

    monkeypatch.setattr(ui, "analyze_video", analyze)
    app = AppTest.from_file("streamlit_app.py").run()
    app.sidebar.radio[0].set_value("Video Analysis").run()
    app.button[0].click().run()
    assert not app.exception and len(app.metric) == 3
    app.selectbox(key="video_exercise").select("squat").run()
    assert len(app.metric) == 0
    app.button[0].click().run()
    assert len(app.metric) == 3
    upload = BytesIO(video + b"changed")
    upload.name = "other.mp4"
    app.run()
    assert len(app.metric) == 0
    app.button[0].click().run()
    assert len(app.metric) == 3

    def fail(*args):
        raise ValueError("Invalid video")

    monkeypatch.setattr(ui, "analyze_video", fail)
    app.button[0].click().run()
    assert app.error and not app.exception and len(app.metric) == 0


def test_coaching_reply_matches_question_and_failed_retry(monkeypatch):
    import gym_tracker.ui.app as ui
    from streamlit.testing.v1 import AppTest
    snapshot = workout()
    monkeypatch.setattr(ui, "coach", lambda *a, **kw: "Unique session feedback")
    app = AppTest.from_file("streamlit_app.py").run()
    app.session_state["sessions"] = {snapshot["session_id"]: snapshot}
    app.sidebar.radio[0].set_value("AI Coach").run()
    assert not app.error  # Missing optional secrets must not render a red banner.
    app.button[0].click().run()
    assert any(m.value == "Unique session feedback" for m in app.markdown)
    app.text_input[0].set_value("A different question").run()
    assert not any(m.value == "Unique session feedback" for m in app.markdown)
    app.button[0].click().run()

    def fail(*a, **kw):
        raise ValueError("Provider unavailable")

    monkeypatch.setattr(ui, "coach", fail)
    app.button[0].click().run()
    assert app.error and not app.exception
    assert not any(m.value == "Unique session feedback" for m in app.markdown)


def test_webcam_switch_and_navigation_archive_once(monkeypatch):
    from types import SimpleNamespace
    import streamlit_webrtc
    from streamlit.testing.v1 import AppTest
    processors = {}

    class Processor:
        def __init__(self, mode):
            self.mode, self.closed, self.error, self.finished = mode, 0, None, False
            self.data = workout()

        def snapshot(self):
            return self.data

        def on_ended(self):
            self.closed += 1
            self.finished = True

    def streamer(key, **kwargs):
        assert kwargs["rtc_configuration"]["iceServers"] == []
        assert kwargs["async_processing"] is True
        if key not in processors:
            processors[key] = Processor(key.removeprefix("gym-"))
        return SimpleNamespace(video_processor=processors[key])

    monkeypatch.setattr(streamlit_webrtc, "webrtc_streamer", streamer)
    app = AppTest.from_file("streamlit_app.py").run()
    app.sidebar.radio[0].set_value("WebCam Live").run()
    assert not app.exception and not app.error
    first = processors["gym-bicep_curl"]
    app.selectbox(key="webcam_exercise").select("squat").run()
    assert first.closed == 1
    second = processors["gym-squat"]
    app.sidebar.radio[0].set_value("Dashboard").run()
    app.run()
    assert second.closed == 1
    assert len(app.session_state["sessions"]) == 2


def test_blank_environment_uses_secrets(monkeypatch):
    from types import SimpleNamespace
    import gym_tracker.ui.app as ui
    monkeypatch.setenv("COACH_MODEL", "  ")
    monkeypatch.setattr(ui.st, "secrets", SimpleNamespace(
        load_if_toml_exists=lambda: True, get=lambda key: "configured-model"))
    assert ui.setting("COACH_MODEL") == "configured-model"


def test_demo_navigation_and_history_filter(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import gym_tracker.ui.app as ui
    monkeypatch.setattr(ui, "make_tracker", lambda *a: object())

    def analyze(source, output, *args):
        assert source.read_bytes() == Path("assets/videos/bicep_curl_demo.mp4").read_bytes()
        output.write_bytes(source.read_bytes())
        return workout()

    monkeypatch.setattr(ui, "analyze_video", analyze)
    app = AppTest.from_file("streamlit_app.py").run()
    next(b for b in app.button if b.label == "Try the live demo ✧").click().run()
    assert app.sidebar.radio[0].value == "Auto Classify"
    assert app.toggle[0].value
    app.button[0].click().run()
    assert not app.exception and len(app.metric) == 3
    app.sidebar.radio[0].set_value("Workout History").run()
    assert not app.exception and len(app.metric) == 3
    app.multiselect[0].set_value(["push_up"]).run()
    assert "No recorded sessions match" in app.info[0].value
    app.multiselect[0].set_value(["squat"]).run()
    assert not app.exception and len(app.metric) == 3
