"""actions/find.py — exec_find() 핵심 로직.

known class local find: detector가 인지하는 COCO class만 탐색합니다.
Phase 1: 현재 시야에서 즉시 탐색.
Phase 2: 로봇 회전하며 scan 기반 탐색.
"""

import time

from geometry_msgs.msg import Twist

from .scan import exec_scan


def exec_find(
    node,
    perception_cache,
    cmd_pub,
    vlm_client=None,
    target_class: str = "person",
    timeout_sec: float = 30.0,
    sweep_deg: float = 360.0,
    snapshot_pub=None,
    feedback_cb=None,
    mission_id: str = "",
    request_id: str = "",
    center_after_find: bool = True,
    **kwargs,
) -> dict:
    """find() 실행. known class local find.

    Args:
        feedback_cb: 진행 상태 callback.
            Phase 1/2 전환 시, Phase 2 중에는 exec_scan의 throttle callback을 그대로 전달.

    Returns:
        dict: {"success": bool, "found_object": dict|None, "search_method": str}
    """
    import time as _time
    node.get_logger().info(f"find 시작: target={target_class}, timeout={timeout_sec}s")
    phase_start = _time.time()

    # Phase 1: 현재 시야에서 즉시 탐색합니다.
    if feedback_cb:
        feedback_cb({
            "state": "phase1_direct",
            "detail": f"searching {target_class} in current view",
            "elapsed_sec": 0.0,
        })
    snap = perception_cache.snapshot()
    match = _find_in_snapshot(snap, target_class)
    if match:
        node.get_logger().info(
            f"find Phase 1 성공: {target_class} id={match.get('id')} 즉시 발견"
        )
        if center_after_find:
            match = _center_on_found_object(
                node,
                perception_cache,
                cmd_pub,
                match,
                target_class,
                feedback_cb=feedback_cb,
            )
        return {"success": True, "found_object": match, "search_method": "direct"}

    # Phase 2: scan을 재사용하되 목표 class를 찾으면 즉시 멈춥니다.
    node.get_logger().info(f"find Phase 2: 회전 탐색 시작 ({sweep_deg}°)")
    if feedback_cb:
        feedback_cb({
            "state": "phase2_rotate",
            "detail": f"rotating {sweep_deg} deg",
            "elapsed_sec": _time.time() - phase_start,
        })

    # Phase 2의 중간 feedback은 exec_scan이 주기적으로 전달
    def _scan_feedback_forwarder(scan_fb: dict):
        if feedback_cb:
            feedback_cb({
                "state": "phase2_rotate",
                "detail": scan_fb.get("detail", ""),
                "elapsed_sec": _time.time() - phase_start,
            })

    scan_result = exec_scan(
        node,
        perception_cache,
        cmd_pub,
        vlm_client=vlm_client,
        sweep_deg=sweep_deg,
        duration_sec=timeout_sec,
        filter_classes=[target_class],
        snapshot_pub=snapshot_pub,
        feedback_cb=_scan_feedback_forwarder,
        stop_on_first=True,
        mission_id=mission_id,
        request_id=request_id,
    )

    if scan_result["success"] and scan_result["objects_found"]:
        best = _pick_best(scan_result["objects_found"])
        node.get_logger().info(
            f"find Phase 2 성공: {target_class} id={best.get('id')}"
        )
        if center_after_find:
            best = _center_on_found_object(
                node,
                perception_cache,
                cmd_pub,
                best,
                target_class,
                feedback_cb=feedback_cb,
            )
        return {"success": True, "found_object": best, "search_method": "rotate"}

    node.get_logger().info(f"find 실패: {target_class} 미발견")
    return {"success": False, "found_object": None, "search_method": "rotate"}


def _find_in_snapshot(snap: dict, target_class: str) -> dict | None:
    for obj in snap.get("targets", []):
        if obj.get("class") == target_class:
            return obj
    return None


def _pick_best(objects: list) -> dict:
    """최고 confidence 또는 최근거리 객체를 선택합니다."""
    # range_m이 있는 객체 중 가장 가까운 것 우선
    with_range = [o for o in objects if o.get("range_m") is not None]
    if with_range:
        return min(with_range, key=lambda o: o["range_m"])
    # range_m 없으면 confidence 최고
    return max(objects, key=lambda o: o.get("confidence", 0))


def _center_on_found_object(
    node,
    perception_cache,
    cmd_pub,
    found_object: dict,
    target_class: str,
    *,
    feedback_cb=None,
    timeout_sec: float = 4.0,
    poll_interval_sec: float = 0.05,
    yaw_deadband_px: float = 28.0,
    stable_frames_required: int = 2,
    k_yaw: float = 0.0020,
    max_angular_z: float = 0.35,
) -> dict:
    """찾은 객체가 화면 중앙에 오도록 짧게 yaw 보정합니다.

    scan 중 발견한 bbox는 회전 중인 frame일 수 있어 화면 가장자리에 남을 수
    있습니다. 성공 직후 stop을 보낸 뒤 같은 track id/class를 보며 yaw-only
    P 제어를 몇 frame 수행합니다.
    """
    target_id = _safe_int(found_object.get("id"), default=-1)
    _publish_stop(cmd_pub, repeat=5, interval_sec=0.02)

    node.get_logger().info(
        f"find center 보정 시작: target={target_class}, id={target_id}, "
        f"deadband={yaw_deadband_px}px"
    )
    if feedback_cb:
        feedback_cb({
            "state": "centering",
            "detail": f"centering {target_class}",
            "elapsed_sec": 0.0,
        })

    start = time.time()
    stable_frames = 0
    latest = found_object

    while (time.time() - start) < timeout_sec:
        snap = perception_cache.snapshot()
        target = _find_matching_target(snap, latest, target_class)
        if target is None:
            cmd_pub.publish(Twist())
            time.sleep(poll_interval_sec)
            continue

        latest = target
        center = target.get("center") or {}
        frame_w = float(snap.get("frame_w") or 640)
        frame_cx = frame_w / 2.0
        target_cx = _safe_float(center.get("x"), default=frame_cx)
        error_x = target_cx - frame_cx

        if abs(error_x) <= yaw_deadband_px:
            stable_frames += 1
            cmd_pub.publish(Twist())
            if stable_frames >= stable_frames_required:
                _publish_stop(cmd_pub)
                node.get_logger().info(
                    f"find center 보정 완료: id={target.get('id')} "
                    f"error_x={error_x:.1f}px"
                )
                return target
        else:
            stable_frames = 0
            twist = Twist()
            twist.angular.z = _clamp(
                -k_yaw * error_x,
                -max_angular_z,
                max_angular_z,
            )
            cmd_pub.publish(twist)

        time.sleep(poll_interval_sec)

    _publish_stop(cmd_pub)
    final_error = _center_error_px(perception_cache.snapshot(), latest, target_class)
    if final_error is None:
        node.get_logger().warn("find center 보정 timeout: target lost")
    else:
        node.get_logger().warn(
            f"find center 보정 timeout: residual_error={final_error:.1f}px"
        )
    return latest


def _find_matching_target(
    snap: dict,
    reference: dict,
    target_class: str,
) -> dict | None:
    targets = snap.get("targets", [])
    ref_id = _safe_int(reference.get("id"), default=-1)
    if ref_id >= 0:
        for obj in targets:
            same_class = obj.get("class") == target_class
            same_id = _safe_int(obj.get("id"), default=-1) == ref_id
            if same_class and same_id:
                return obj

    candidates = [obj for obj in targets if obj.get("class") == target_class]
    if not candidates:
        return None

    ref_center = reference.get("center") or {}
    if "x" in ref_center and "y" in ref_center:
        ref_x = _safe_float(ref_center.get("x"), default=0.0)
        ref_y = _safe_float(ref_center.get("y"), default=0.0)
        return min(candidates, key=lambda obj: _center_distance_sq(obj, ref_x, ref_y))

    return _pick_best(candidates)


def _center_distance_sq(obj: dict, ref_x: float, ref_y: float) -> float:
    center = obj.get("center") or {}
    dx = _safe_float(center.get("x"), default=ref_x) - ref_x
    dy = _safe_float(center.get("y"), default=ref_y) - ref_y
    return dx * dx + dy * dy


def _center_error_px(
    snap: dict,
    reference: dict,
    target_class: str,
) -> float | None:
    target = _find_matching_target(snap, reference, target_class)
    if target is None:
        return None
    center = target.get("center") or {}
    frame_w = float(snap.get("frame_w") or 640)
    return _safe_float(center.get("x"), default=frame_w / 2.0) - frame_w / 2.0


def _publish_stop(cmd_pub, repeat: int = 5, interval_sec: float = 0.02):
    stop = Twist()
    for _ in range(repeat):
        cmd_pub.publish(stop)
        time.sleep(interval_sec)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
