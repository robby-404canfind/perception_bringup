"""actions/follow.py — exec_follow() 핵심 로직.

tracked target ID를 유지하면서 화면 중앙 정렬 + 목표 거리 유지.
depth가 있으면 거리 유지, 없으면 yaw-only fallback.
target lost > lost_timeout이면 LOST_TARGET으로 종료.
로봇 본체 회전(cmd_vel)으로 대상을 추종합니다.
"""

import time

from geometry_msgs.msg import Twist


def exec_follow(
    node,
    perception_cache,
    cmd_pub,
    target_id: int = -1,
    target_class: str = "person",
    target_distance_m: float = 2.0,
    max_time_sec: float = 60.0,
    k_yaw: float = 0.0015,
    max_angular_z: float = 0.4,
    yaw_deadband_px: float = 30.0,
    k_dist: float = 0.3,
    lost_timeout: float = 5.0,
    feedback_cb=None,
    **kwargs,
) -> dict:
    """follow() 실행. tracked target 추종.

    Args:
        feedback_cb: 진행 상태 callback. 1초 간격으로 dict 전달:
            {"state", "elapsed_sec", "current_distance_m", "target_status"}.

    Returns:
        dict: {"success": bool, "final_state": str, "elapsed_sec": float}
    """
    last_seen = time.time()
    start = time.time()
    last_fb_time = 0.0
    feedback_interval = 1.0  # 1초 throttle
    ever_seen = False

    node.get_logger().info(
        f"follow 시작: target_class={target_class}, target_id={target_id}, "
        f"distance={target_distance_m}m, max={max_time_sec}s"
    )

    final_state = "timeout"

    while (time.time() - start) < max_time_sec:
        snap = perception_cache.snapshot()
        target = _find_target(snap, target_id, target_class)

        if target is None:
            if (time.time() - last_seen) > lost_timeout:
                final_state = "lost_target"
                node.get_logger().info("follow: LOST_TARGET")
                break
            # 타겟 미발견 → 정지하고 대기
            cmd_pub.publish(Twist())
            # Feedback: searching
            now = time.time()
            if feedback_cb and (now - last_fb_time) >= feedback_interval:
                feedback_cb({
                    "state": "following",
                    "elapsed_sec": now - start,
                    "current_distance_m": 0.0,
                    "target_status": "searching",
                })
                last_fb_time = now
            time.sleep(0.05)
            continue

        last_seen = time.time()
        ever_seen = True
        # 첫 발견 시 target_id 고정
        if target_id < 0 and target.get("id", -1) >= 0:
            target_id = target["id"]
            node.get_logger().info(f"follow: target ID 고정 → {target_id}")

        # Yaw 정렬: 화면 중심에 타겟 유지
        frame_cx = snap["frame_w"] / 2
        error_x = target["center"]["x"] - frame_cx
        twist = Twist()
        if abs(error_x) > yaw_deadband_px:
            twist.angular.z = _clamp(-k_yaw * error_x, -max_angular_z, max_angular_z)

        # 거리 유지 (depth 있을 때만)
        range_m = target.get("range_m")
        if range_m is not None:
            error_d = range_m - target_distance_m
            if abs(error_d) > 0.3:
                twist.linear.x = k_dist * error_d
            # 안전: 너무 가까우면 후진
            if range_m < 0.8:
                twist.linear.x = -0.1

        cmd_pub.publish(twist)

        # Feedback: tracked (throttle)
        now = time.time()
        if feedback_cb and (now - last_fb_time) >= feedback_interval:
            feedback_cb({
                "state": "following",
                "elapsed_sec": now - start,
                "current_distance_m": float(range_m) if range_m is not None else 0.0,
                "target_status": "tracked",
            })
            last_fb_time = now

        time.sleep(0.05)

    # 정지 명령은 BestEffort 구독에서 유실될 수 있으므로 짧게 반복 publish합니다.
    _publish_stop(cmd_pub)

    elapsed = round(time.time() - start, 1)
    if final_state == "timeout" and ever_seen:
        final_state = "completed"
    elif final_state == "timeout":
        final_state = "lost_target"

    node.get_logger().info(f"follow 종료: {final_state}, {elapsed}s")
    return {
        "success": final_state == "completed",
        "final_state": final_state,
        "elapsed_sec": elapsed,
    }


def _find_target(snap: dict, target_id: int, target_class: str) -> dict | None:
    targets = snap.get("targets", [])
    # ID 기반 매칭 (우선)
    if target_id >= 0:
        for obj in targets:
            if obj.get("id") == target_id:
                return obj
    # class 기반 fallback (가장 가까운 것)
    candidates = [o for o in targets if o.get("class") == target_class]
    if not candidates:
        return None
    with_range = [o for o in candidates if o.get("range_m") is not None]
    if with_range:
        return min(with_range, key=lambda o: o["range_m"])
    return candidates[0]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _publish_stop(cmd_pub, repeat: int = 5, interval_sec: float = 0.02):
    stop = Twist()
    for _ in range(repeat):
        cmd_pub.publish(stop)
        time.sleep(interval_sec)
