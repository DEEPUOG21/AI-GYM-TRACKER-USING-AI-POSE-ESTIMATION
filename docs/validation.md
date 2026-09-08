# Validation record

Validated locally on 2026-09-08 using an isolated Linux x86-64 environment with
Python 3.11.16 and the pinned application dependencies.

- `python -m pytest -q`: **48 passed**, including the real TensorFlow model,
  MediaPipe pose initialization, actual FFmpeg encoding, and Streamlit AppTest.
- `ruff check .`: passed.
- Dependency compatibility check: all 104 installed runtime/development packages
  compatible (the runtime lock excludes pytest/ruff and their test-only dependencies).
- Compared all 22 new features against the original extractor on 100 deterministic
  valid, nonzero synthetic poses: values matched to 1e-12 tolerance. Synthetic
  geometry is a regression check, not recognition-accuracy evidence.
- Ran automatic analysis on all 300 frames of `assets/videos/bicep_curl_demo.mp4`: decoded 10.01
  seconds, generated an annotated H.264 MP4 and structured telemetry, and closed
  the detector/encoder. `screenshots/annotated_frame.jpg` is an actual frame from this output.
- Ran `python -m scripts.benchmark assets/videos/bicep_curl_demo.mp4 --exercise auto --warmup 30
  --max-frames 300`. The local output is `reports/benchmark.json` (generated reports
  are ignored). It measures this machine/input only and includes actual classifier
  windows; it is not a browser latency measurement.

Not validated here:

- Recognition accuracy, precision, recall and F1 on a held-out dataset: no labeled
  data or original training split was supplied. The metric calculations have
  deterministic unit tests; there is no model-quality report.
- Browser camera permissions, remote WebRTC/TURN connectivity and live-provider
  coaching calls. Provider request/response behavior is tested using mocked HTTP.
- Docker image build and remote deployment. The available Docker executable is
  Snap-managed and fails in this sandbox with a missing `cap_dac_override`
  capability. Source-runtime checks passed; a container build was not claimed.

The pinned MediaPipe runtime emits upstream protobuf deprecation and TensorFlow
feedback-manager warnings. These did not fail pose or video processing. They are
retained visibly rather than globally suppressing warnings.

## Streamlit regression checks

Additional tests cover changing a video's exercise or upload, failed analysis
retries, changing coaching questions, failed coaching retries, missing optional
secrets, empty environment values falling back to secrets, webcam exercise/page
switches, initialization failures and concurrent frame processing/shutdown.
The seven UI/adapter tests passed; the full suite now passes **54 tests**, and
`ruff check .` passes. WebRTC component behavior is mocked in AppTest
because it lacks a real browser/session manager; this does not validate camera
permissions or live network connectivity.

## Interactive UI redesign

The full suite passed **55 tests** after adding demo navigation and workout-history
filter coverage. The final UI tests and lint also passed after the layout polish.
The installed Streamlit/Altair combination emits an upstream theme deprecation
warning; the charts rendered successfully.

The redesigned interface was exercised in headless Chrome against a real local
Streamlit server. The included demo completed real inference, session charts
rendered, and telemetry downloaded as `workout.json`. Desktop and 390-pixel mobile
layouts were checked for horizontal overflow. The custom navigation keeps the
native radio controls available for keyboard access. Screenshots in this directory
show the actual running frontend, not design mockups.

This verifies local UI behavior, not cloud deployment, camera permissions on other
devices, or live coaching-provider availability.

## Browser camera fix

Reproduced two startup failures: external STUN gathering remained pending, and
the old 0.45.0 component repeatedly restarted its worker with Streamlit 1.35.
The updated component (0.47.9) includes Streamlit compatibility fixes documented
in the [upstream changelog](https://github.com/whitphx/streamlit-webrtc/blob/main/CHANGELOG.md).
Local mode now uses direct ICE candidates; remote mode retains configurable
STUN/TURN servers. Model loading runs after negotiation in an asynchronous worker.

Verified with headless Chrome's synthetic camera on localhost: peer/ICE states
were `connected`, gathering was `complete`, signaling was `stable`, and the
returned 640×480 video had readyState 4 with its playback clock advancing beyond
16 seconds. No app errors were displayed. No physical camera was accessed.
The camera/UI regression tests pass with the updated component; tests also ensure
the factory does not load models during negotiation and local mode uses no STUN.

## Repetition counter checks

After fixing visibility gates and mirror-sensitive angles, the suite passes
89 tests. Synthetic landmark tests cover each exercise, horizontal mirroring,
partial framing, holding a phase, repeated cycles, tracking gaps and degenerate
joints. Push-up/squat tests cover either visible side and reset on side changes.
Pipeline tests verify that partial poses reach session rep telemetry and that
invisible wrists cannot trigger the stop gesture. Auto classification still
requires the original full-body feature contract.

A real MediaPipe run on `assets/videos/bicep_curl_demo.mp4` in manual curl mode processed 300 frames,
accepted arm visibility in all 300, and emitted three reps. The shoulder clip
smoke test processed 315 frames and emitted four reps. These are observed outputs,
not verified ground truth or accuracy measurements. No push-up or squat recordings
are bundled, so those exercises still need validation on labeled real recordings
and the user's physical camera.
