import pytest
from gym_tracker.repetitions import RepCounter


@pytest.mark.parametrize("exercise,ready,finish", [
    ("push_up", (180, 200), (180, 260)), ("squat", (180, 180), (120, 230)),
    ("bicep_curl", (180, 180), (40, 40)), ("shoulder_press", (270, 90), (200, 160))])
def test_rep_cycle_and_gap(exercise, ready, finish):
    counter = RepCounter(exercise)
    assert not counter.update_angles(*finish)
    assert not counter.update_angles(*ready)
    assert counter.update_angles(*finish)
    assert not counter.update_angles(*finish)
    assert counter.reps == 1
    counter.update_angles(*ready)
    counter.update_angles(None, None)
    assert not counter.update_angles(*finish)
    assert counter.reps == 1
    counter.update_angles(*ready)
    assert counter.update_angles(*finish)
    assert counter.reps == 2


def test_both_arms_required():
    counter = RepCounter("bicep_curl")
    counter.update_angles(180, 180)
    assert not counter.update_angles(40, 180)
    assert counter.update_angles(40, 40)


def test_missing_landmarks_reset():
    counter = RepCounter("push_up")
    counter.update_angles(180, 200)
    assert not counter.update(None, (640, 480))
    assert not counter.update_angles(180, 260)


def test_threshold_hysteresis():
    counter = RepCounter("push_up")
    for left in (230, 235, 240, 241, 230):
        assert not counter.update_angles(180, left)
    assert counter.reps == 0


@pytest.mark.parametrize("mirrored", [False, True])
def test_shoulder_press_upper_body_cycle(shoulder_pose, mirrored):
    counter = RepCounter("shoulder_press")
    bottom = shoulder_pose(mirrored=mirrored)
    top = shoulder_pose(extended=True, mirrored=mirrored)
    assert not counter.update(top, (640, 480))
    for expected in (1, 2):
        assert not counter.update(bottom, (640, 480))
        assert counter.update(top, (640, 480))
        assert not counter.update(top, (640, 480))
        assert counter.reps == expected


@pytest.mark.parametrize("joint", range(11, 17))
def test_shoulder_press_occluded_arm_resets(shoulder_pose, joint):
    counter = RepCounter("shoulder_press")
    bottom = shoulder_pose()
    counter.update(bottom, (640, 480))
    bottom[joint, 3] = 0.1
    assert not counter.update(bottom, (640, 480))
    assert not counter.update(shoulder_pose(extended=True), (640, 480))
    assert counter.reps == 0


def test_shoulder_press_does_not_count_lowering_arms(shoulder_pose):
    counter = RepCounter("shoulder_press")
    counter.update(shoulder_pose(), (640, 480))
    lowered = shoulder_pose(extended=True)
    lowered[[13, 14], 1] = 0.7
    lowered[[15, 16], 1] = 0.8
    assert not counter.update(lowered, (640, 480))
    assert not counter.update(shoulder_pose(extended=True), (640, 480))


def test_shoulder_press_requires_both_arms_and_hysteresis():
    counter = RepCounter("shoulder_press")
    counter.update_angles(270, 90)
    for angles in ((220, 140), (200, 90), (270, 160), (206, 154)):
        assert not counter.update_angles(*angles)
    assert counter.update_angles(200, 160)
    for angle in (154, 156, 150, 160):
        assert not counter.update_angles(360 - angle, angle)
    assert counter.reps == 1


@pytest.mark.parametrize("exercise,finish", [("bicep_curl", 40), ("push_up", 100), ("squat", 120)])
@pytest.mark.parametrize("mirrored", [False, True])
def test_other_exercises_with_partial_framing(exercise_pose, exercise, finish, mirrored):
    counter = RepCounter(exercise)
    ready = exercise_pose(exercise, 180, mirrored=mirrored)
    end = exercise_pose(exercise, finish, mirrored=mirrored)
    assert not counter.update(end, (640, 480))
    for reps in (1, 2):
        assert not counter.update(ready, (640, 480))
        assert counter.update(end, (640, 480))
        assert not counter.update(end, (640, 480))
        assert counter.reps == reps
    counter.update(ready, (640, 480))
    counter.update(None, (640, 480))
    assert not counter.update(end, (640, 480))
    assert counter.reps == 2


@pytest.mark.parametrize("exercise,finish", [("push_up", 100), ("squat", 120)])
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("mirrored", [False, True])
def test_single_side_and_side_switch(exercise_pose, exercise, finish, side, mirrored):
    counter = RepCounter(exercise)
    def pose(angle, visible_side):
        return exercise_pose(exercise, angle, side=visible_side, mirrored=mirrored)
    counter.update(pose(180, side), (640, 480))
    assert counter.update(pose(finish, side), (640, 480))
    counter.update(pose(180, side), (640, 480))
    assert not counter.update(pose(finish, 1 - side), (640, 480))
    counter.update(pose(180, 1 - side), (640, 480))
    assert counter.update(pose(finish, 1 - side), (640, 480))
    assert counter.reps == 2


@pytest.mark.parametrize("exercise,finish", [("bicep_curl", 40), ("push_up", 100), ("squat", 120)])
def test_degenerate_joints_reset(exercise_pose, exercise, finish):
    counter = RepCounter(exercise)
    ready = exercise_pose(exercise, 180)
    counter.update(ready, (640, 480))
    triples = ((24, 26), (23, 25)) if exercise == "squat" else ((12, 14), (11, 13))
    for a, b in triples:
        ready[a] = ready[b]
    assert not counter.update(ready, (640, 480))
    assert not counter.update(exercise_pose(exercise, finish), (640, 480))


def test_curl_requires_both_visible_arms(exercise_pose):
    counter = RepCounter("bicep_curl")
    counter.update(exercise_pose("bicep_curl", 180), (640, 480))
    assert not counter.update(exercise_pose("bicep_curl", 40, side=0), (640, 480))
    assert not counter.update(exercise_pose("bicep_curl", 40), (640, 480))
