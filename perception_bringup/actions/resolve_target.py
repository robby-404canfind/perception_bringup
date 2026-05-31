"""actions/resolve_target.py — 자연어 target query를 YOLO track id로 연결."""

import json
import re
import time
from typing import Any


RESOLVE_SCHEMA_HINT = {
    "target_query": "",
    "matched": False,
    "target_id": -1,
    "target_class": "person",
    "reason": "no matching target",
    "confidence": 0.0,
}


def exec_resolve_target(
    node,
    vlm_client,
    cv_image,
    context: dict | None,
    target_query: str,
    timeout_sec: float = 30.0,
    feedback_cb=None,
    **kwargs,
) -> dict:
    """target_query와 가장 잘 맞는 person track id를 반환합니다."""
    start = time.time()
    context = context or {}
    objects = context.get("objects", [])
    people = [obj for obj in objects if obj.get("class") == "person"]

    if feedback_cb:
        feedback_cb({
            "state": "preparing",
            "detail": f"resolving target: {target_query}",
            "elapsed_sec": 0.0,
        })

    resolved = None
    if vlm_client is not None and cv_image is not None and people:
        try:
            if feedback_cb:
                feedback_cb({
                    "state": "vlm",
                    "detail": f"asking VLM to select among {len(people)} person tracks",
                    "elapsed_sec": time.time() - start,
                })
            annotated = annotate_people(cv_image, people)
            prompt = _build_resolve_prompt(target_query, people)
            resolved = vlm_client.describe_scene(annotated, prompt=prompt)
        except Exception as e:
            node.get_logger().warn(f"resolve_target VLM 실패: {e}")

    if not isinstance(resolved, dict):
        resolved = _fallback_resolve(target_query, people)

    normalized = normalize_resolved_target(resolved, target_query=target_query)
    elapsed = round(time.time() - start, 1)
    normalized["elapsed_sec"] = elapsed
    normalized["source"] = "vlm" if cv_image is not None else "context_fallback"

    if feedback_cb:
        feedback_cb({
            "state": "done",
            "detail": normalized.get("reason", ""),
            "elapsed_sec": elapsed,
        })

    node.get_logger().info(
        "resolve_target 완료: "
        f"matched={normalized['matched']} target_id={normalized['target_id']} "
        f"confidence={normalized['confidence']:.2f}"
    )
    return {
        "success": bool(normalized["matched"] and normalized["target_id"] >= 0),
        "target": normalized,
        "message": _target_message(normalized),
        "elapsed_sec": elapsed,
    }


def annotate_people(cv_image, people: list):
    """VLM이 track id를 읽기 쉽도록 bbox와 id label을 이미지에 그립니다."""
    try:
        import cv2
    except ImportError:
        return cv_image.copy()

    annotated = cv_image.copy()
    palette = [
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 180, 255),
        (255, 180, 0),
    ]
    for idx, obj in enumerate(people):
        bbox = obj.get("bbox") or {}
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("w", 0))
        h = int(bbox.get("h", 0))
        color = palette[idx % len(palette)]
        obj_id = obj.get("id", -1)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        label = f"id {obj_id}"
        cv2.putText(
            annotated,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def normalize_resolved_target(data: dict[str, Any], target_query: str = "") -> dict:
    result = dict(RESOLVE_SCHEMA_HINT)
    result["target_query"] = str(data.get("target_query") or target_query)
    result["matched"] = bool(data.get("matched", False))
    result["target_id"] = _safe_int(data.get("target_id"), -1)
    result["target_class"] = str(data.get("target_class") or "person")
    result["reason"] = str(data.get("reason") or "판단 근거가 충분하지 않습니다.")
    result["confidence"] = _clamp_float(data.get("confidence"), 0.0, 1.0, 0.0)
    if result["target_id"] < 0:
        result["matched"] = False
    return result


def _build_resolve_prompt(target_query: str, people: list) -> str:
    candidates = [
        {
            "id": obj.get("id", -1),
            "direction": obj.get("direction", "unknown"),
            "range_m": obj.get("range_m"),
            "confidence": obj.get("confidence", 0.0),
        }
        for obj in people[:8]
    ]
    return f"""\
너는 모바일 로봇의 target resolver이다.
이미지에는 person bbox와 track id가 표시되어 있다.
사용자 target_query와 가장 잘 맞는 사람의 track id를 고르라.
관찰 가능한 단서(옷 색상, 소지품, 위치, id label)만 사용하고 성별/나이/신원은 추정하지 마라.

[target_query]
{target_query}

[candidate tracks]
{json.dumps(candidates, ensure_ascii=False)}

반드시 아래 JSON object 하나만 반환하라.
{{
  "target_query": "{target_query}",
  "matched": true 또는 false,
  "target_id": 선택한 track id 또는 -1,
  "target_class": "person",
  "reason": "짧은 한국어 근거",
  "confidence": 0.0~1.0
}}
"""


def _fallback_resolve(target_query: str, people: list) -> dict:
    if not people:
        return {
            "target_query": target_query,
            "matched": False,
            "target_id": -1,
            "target_class": "person",
            "reason": "현재 감지된 person track이 없습니다.",
            "confidence": 0.0,
        }

    explicit_id = _extract_explicit_id(target_query)
    if explicit_id is not None:
        for obj in people:
            if _safe_int(obj.get("id"), -1) == explicit_id:
                return _resolved_from_obj(
                    target_query, obj, "query에 명시된 track id와 일치합니다.", 0.9
                )

    query_l = target_query.lower()
    direction = None
    if any(token in query_l for token in ("left", "왼쪽", "좌측")):
        direction = "left"
    elif any(token in query_l for token in ("right", "오른쪽", "우측")):
        direction = "right"
    elif any(token in query_l for token in ("center", "middle", "가운데", "중앙")):
        direction = "center"

    if direction:
        matches = [obj for obj in people if obj.get("direction") == direction]
        if matches:
            return _resolved_from_obj(
                target_query,
                _nearest(matches),
                f"{direction} 위치 단서와 일치합니다.",
                0.55,
            )

    if len(people) == 1:
        return _resolved_from_obj(
            target_query,
            people[0],
            "후보 person이 하나뿐이므로 해당 track을 선택했습니다.",
            0.45,
        )

    return {
        "target_query": target_query,
        "matched": False,
        "target_id": -1,
        "target_class": "person",
        "reason": "이미지 VLM 없이 여러 사람 중 옷 색상/소지품 단서를 구분할 수 없습니다.",
        "confidence": 0.2,
    }


def _resolved_from_obj(query: str, obj: dict, reason: str, confidence: float) -> dict:
    return {
        "target_query": query,
        "matched": True,
        "target_id": _safe_int(obj.get("id"), -1),
        "target_class": "person",
        "reason": reason,
        "confidence": confidence,
    }


def _nearest(people: list) -> dict:
    with_range = [p for p in people if p.get("range_m") is not None]
    if with_range:
        return min(with_range, key=lambda p: float(p.get("range_m") or 999.0))
    return people[0]


def _target_message(target: dict) -> str:
    if target.get("matched"):
        return (
            f"target_id={target.get('target_id')} resolved "
            f"confidence={target.get('confidence'):.2f}: {target.get('reason')}"
        )
    return f"target unresolved: {target.get('reason')}"


def _extract_explicit_id(query: str) -> int | None:
    match = re.search(r"\b(?:id|track)\s*#?\s*(\d+)\b", query.lower())
    if match:
        return int(match.group(1))
    return None


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_float(value, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))
