from perception_bringup.actions.find import _center_on_found_object


class FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass


class FakeNode:
    def __init__(self):
        self._logger = FakeLogger()

    def get_logger(self):
        return self._logger


class FakeCmdPub:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class SequenceCache:
    def __init__(self, targets_by_snapshot):
        self.targets_by_snapshot = list(targets_by_snapshot)
        self.index = 0

    def snapshot(self):
        idx = min(self.index, len(self.targets_by_snapshot) - 1)
        self.index += 1
        return {
            "frame_w": 640,
            "frame_h": 480,
            "targets": self.targets_by_snapshot[idx],
        }


def _person(cx, track_id=7):
    return {
        "id": track_id,
        "class": "person",
        "confidence": 0.9,
        "center": {"x": cx, "y": 240},
        "bbox": {"x": cx - 25, "y": 180, "w": 50, "h": 120},
    }


def test_center_on_found_object_rotates_target_toward_center():
    cmd_pub = FakeCmdPub()
    cache = SequenceCache(
        [
            [_person(500)],
            [_person(340)],
            [_person(330)],
            [_person(324)],
        ]
    )

    result = _center_on_found_object(
        FakeNode(),
        cache,
        cmd_pub,
        _person(500),
        "person",
        timeout_sec=0.5,
        yaw_deadband_px=28.0,
        stable_frames_required=2,
        poll_interval_sec=0.0,
    )

    angular_commands = [msg.angular.z for msg in cmd_pub.messages]
    assert any(z < 0 for z in angular_commands)
    assert result["center"]["x"] == 330


def test_center_on_found_object_keeps_track_id_when_multiple_people():
    cmd_pub = FakeCmdPub()
    cache = SequenceCache(
        [
            [_person(500, track_id=1), _person(320, track_id=2)],
            [_person(350, track_id=1), _person(320, track_id=2)],
            [_person(326, track_id=1), _person(320, track_id=2)],
        ]
    )

    result = _center_on_found_object(
        FakeNode(),
        cache,
        cmd_pub,
        _person(500, track_id=1),
        "person",
        timeout_sec=0.5,
        yaw_deadband_px=28.0,
        stable_frames_required=1,
        poll_interval_sec=0.0,
    )

    assert result["id"] == 1
    assert result["center"]["x"] == 326
