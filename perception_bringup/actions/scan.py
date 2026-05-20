"""actions/scan.py — exec_scan() 핵심 로직.

현재 위치에서 로봇 본체 회전(cmd_vel)으로 주변을 스캔합니다.
watch_classes/watch_ids에 매칭되는 객체가 발견되면 FOUND 이벤트를 기록합니다.
"""

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
    watch_classes: list | None = None,
    watch_ids: list | None = None,
    snapshot_pub=None,
    feedback_cb=None,
    **kwargs,
) -> dict:
    """scan() 실행. 로봇 회전 sweep + detection 모니터링.

    Args:
        feedback_cb: 진행 상태를 상위(ActionServer)에 보고하는 콜백.
            호출 시 dict: {"state", "detail", "elapsed_sec", "objects_found"}.
            1초 간격으로 throttle 적용.

    Returns:
        dict: {"success": bool, "objects_found": list, "scene_description": str|None}
    """
    angular_speed = 0.3  # rad/s (~17 deg/s)
    poll_interval = 0.1  # 100ms
    feedback_interval = 1.0  # 1초 throttle
    found_objects: dict = {}  # key: "class_id" → 중복 방지
    scene_desc = None

    twist = Twist()
    twist.angular.z = angular_speed

    start = time.time()
    last_fb_time = 0.0
    node.get_logger().info(
        f"scan 시작: sweep={sweep_deg}°, duration={duration_sec}s, "
        f"watch_classes={watch_classes}, watch_ids={watch_ids}"
    )

    while (time.time() - start) < duration_sec:
        cmd_pub.publish(twist)

        snap = perception_cache.snapshot()
        for obj in snap["targets"]:
            if _matches(obj, watch_classes, watch_ids):
                key = f"{obj.get('class', '?')}_{obj.get('id', '?')}"
                if key not in found_objects:
                    found_objects[key] = obj
                    node.get_logger().info(
                        f"FOUND: {obj.get('class')} id={obj.get('id')} "
                        f"range={obj.get('range_m')}m"
                    )
                    # Snapshot 요청
                    if snapshot_pub:
                        req = {"snapshot_id": key, "requester": "scan", "reason": "FOUND"}
                        snapshot_pub.publish(String(data=json.dumps(req)))

        # Feedback 보고 (throttle)
        now = time.time()
        if feedback_cb and (now - last_fb_time) >= feedback_interval:
            feedback_cb({
                "state": "scanning",
                "detail": f"watching {watch_classes or 'all'}",
                "elapsed_sec": now - start,
                "objects_found": len(found_objects),
            })
            last_fb_time = now

        time.sleep(poll_interval)

    # 정지
    twist.angular.z = 0.0
    cmd_pub.publish(twist)

    # VLM 장면 요약 (선택)
    if vlm_client and hasattr(node, "latest_cv_image") and node.latest_cv_image is not None:
        try:
            vlm_result = vlm_client.describe_scene(node.latest_cv_image)
            scene_desc = vlm_result.get("scene_summary")
        except Exception:
            pass

    elapsed = round(time.time() - start, 1)
    result = {
        "success": len(found_objects) > 0,
        "objects_found": list(found_objects.values()),
        "scene_description": scene_desc,
        "elapsed_sec": elapsed,
    }
    node.get_logger().info(f"scan 완료: {len(found_objects)}개 발견, {elapsed}s")
    return result


def _matches(obj: dict, watch_classes: list | None, watch_ids: list | None) -> bool:
    if watch_classes and obj.get("class") in watch_classes:
        return True
    if watch_ids and obj.get("id") in watch_ids:
        return True
    if not watch_classes and not watch_ids:
        return True  # 필터 없으면 모든 객체 매칭
    return False
