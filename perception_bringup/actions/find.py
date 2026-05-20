"""actions/find.py — exec_find() 핵심 로직.

known class local find: detector가 인지하는 COCO class만 탐색합니다.
Phase 1: 현재 시야에서 즉시 탐색.
Phase 2: 로봇 회전하며 scan 기반 탐색.
"""

import time

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

    # Phase 1: 현재 시야에서 즉시 탐색
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
        return {"success": True, "found_object": match, "search_method": "direct"}

    # Phase 2: 로봇 회전 탐색 (scan 기반)
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
        watch_classes=[target_class],
        snapshot_pub=snapshot_pub,
        feedback_cb=_scan_feedback_forwarder,
    )

    if scan_result["success"] and scan_result["objects_found"]:
        best = _pick_best(scan_result["objects_found"])
        node.get_logger().info(
            f"find Phase 2 성공: {target_class} id={best.get('id')}"
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
