"""actions/scan.py — exec_scan() 핵심 로직.

현재 위치에서 로봇 본체 회전(cmd_vel)으로 주변을 스캔합니다.
scan Action은 특정 클래스를 찾지 않고, YOLO가 감지한 모든 객체를 보고합니다.
find() 내부 재사용 경로에서는 filter_classes로 대상을 제한할 수 있습니다.
"""

from collections import Counter
import json
import time

from geometry_msgs.msg import Twist
from std_msgs.msg import String


def exec_scan(
    node,
    perception_cache,
    cmd_pub,
    vlm_client=None,
    sweep_deg: float = 360.0,
    duration_sec: float = 30.0,
    filter_classes: list | None = None,
    snapshot_pub=None,
    feedback_cb=None,
    stop_on_first: bool = False,
    mission_id: str = "",
    request_id: str = "",
    **kwargs,
) -> dict:
    """scan() 실행. 로봇 회전 sweep + detection 모니터링.

    Args:
        feedback_cb: 진행 상태를 상위(ActionServer)에 보고하는 콜백.
            호출 시 dict: {"state", "detail", "elapsed_sec", "objects_found"}.
            1초 간격으로 throttle 적용.
        stop_on_first: True이면 첫 매칭 객체를 찾은 뒤 scan을 즉시 종료합니다.
            find()의 Phase 2에서 사용합니다.

    Returns:
        dict: {
            "success": bool,
            "objects_found": list,
            "class_summary": str,
            "scene_description": str|None,
        }
    """
    angular_speed = 0.3  # rad/s (~17 deg/s)
    poll_interval = 0.1  # 100ms
    feedback_interval = 1.0  # 1초 throttle
    found_objects: dict = {}  # key: "class:tracking_id" -> 중복 snapshot 방지
    scene_desc = None

    twist = Twist()
    twist.angular.z = angular_speed

    start = time.time()
    last_fb_time = 0.0
    node.get_logger().info(
        f"scan 시작: sweep={sweep_deg}°, duration={duration_sec}s, "
        f"filter_classes={filter_classes}"
    )

    while (time.time() - start) < duration_sec:
        should_stop = False
        cmd_pub.publish(twist)

        snap = perception_cache.snapshot()
        for obj in snap["targets"]:
            if _matches(obj, filter_classes):
                key = _tracking_key(obj)
                if key not in found_objects:
                    found_objects[key] = obj
                    node.get_logger().info(
                        f"FOUND: {obj.get('class')} id={obj.get('id')} "
                        f"conf={obj.get('confidence', '?')} "
                        f"range={_format_range(obj.get('range_m'))}"
                    )
                    # Snapshot 요청
                    if snapshot_pub:
                        req = {
                            "snapshot_id": key,
                            "mission_id": mission_id,
                            "request_id": request_id,
                            "requester": "scan",
                            "reason": "FOUND",
                            "message": (
                                f"{obj.get('class', 'object')} "
                                f"id={obj.get('id', '?')} found during scan"
                            ),
                        }
                        snapshot_pub.publish(String(data=json.dumps(req)))
                    if stop_on_first:
                        should_stop = True
                        break

        # Feedback 보고 (throttle)
        now = time.time()
        if feedback_cb and (now - last_fb_time) >= feedback_interval:
            class_summary = _class_count_summary(found_objects.values())
            feedback_cb({
                "state": "scanning",
                "detail": _format_feedback_detail(
                    class_summary, filter_classes
                ),
                "elapsed_sec": now - start,
                "objects_found": len(found_objects),
            })
            last_fb_time = now

        if should_stop:
            break

        time.sleep(poll_interval)

    # 정지 명령은 BestEffort 구독에서 유실될 수 있으므로 짧게 반복 publish합니다.
    _publish_stop(cmd_pub)

    # VLM 장면 요약 (선택)
    if vlm_client and hasattr(node, "latest_cv_image") and node.latest_cv_image is not None:
        try:
            vlm_result = vlm_client.describe_scene(node.latest_cv_image)
            scene_desc = vlm_result.get("scene_summary")
        except Exception:
            pass

    elapsed = round(time.time() - start, 1)
    objects_found = list(found_objects.values())
    class_summary = _class_count_summary(objects_found)
    result = {
        "success": len(objects_found) > 0,
        "objects_found": objects_found,
        "class_summary": class_summary,
        "scene_description": scene_desc,
        "elapsed_sec": elapsed,
    }
    summary_text = class_summary or "none"
    node.get_logger().info(
        f"scan 완료: {len(objects_found)}개 발견 ({summary_text}), {elapsed}s"
    )
    return result


def _matches(obj: dict, filter_classes: list | None) -> bool:
    if filter_classes and obj.get("class") in filter_classes:
        return True
    if not filter_classes:
        return True  # 필터 없으면 모든 객체 매칭
    return False


def _tracking_key(obj: dict) -> str:
    obj_id = obj.get("id")
    if obj_id is None:
        obj_id = "none"
    return f"{obj.get('class', 'unknown')}:{obj_id}"


def _format_range(range_m) -> str:
    return f"{range_m}m" if range_m is not None else "unknown"


def _class_count_summary(objects) -> str:
    counts = Counter(obj.get("class", "unknown") for obj in objects)
    return ", ".join(f"{cls}({count})" for cls, count in sorted(counts.items()))


def _format_feedback_detail(
    class_summary: str,
    filter_classes: list | None,
) -> str:
    detected = class_summary or "none"
    if filter_classes:
        filters = "classes=" + ",".join(filter_classes)
        return f"filtering {filters} | detected {detected}"
    return f"detected {detected}"


def _publish_stop(cmd_pub, repeat: int = 5, interval_sec: float = 0.02):
    stop = Twist()
    for _ in range(repeat):
        cmd_pub.publish(stop)
        time.sleep(interval_sec)
