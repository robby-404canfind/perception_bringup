"""actions/assess_scene.py — VLM 기반 장면 평가 핵심 로직."""

import json
import time
from typing import Any


ASSESSMENT_SCHEMA_HINT = {
    "query": "original query",
    "has_suspicious_person": False,
    "candidate_track_id": -1,
    "risk_level": "none",
    "reason": "short reason",
    "confidence": 0.0,
    "recommended_action": "ignore",
}


def exec_assess_scene(
    node,
    vlm_client,
    cv_image,
    context: dict | None,
    query: str,
    timeout_sec: float = 30.0,
    feedback_cb=None,
    **kwargs,
) -> dict:
    """현재 장면이 query에 부합하는지 VLM으로 평가합니다.

    VLM 호출이 불가능한 상황에서도 System2가 보고를 이어갈 수 있도록
    최신 detection/context 기반 fallback JSON을 반환합니다.
    """
    start = time.time()
    context = context or {}
    objects = context.get("objects", [])
    vlm_scene = context.get("vlm_scene", {})

    if feedback_cb:
        feedback_cb({
            "state": "preparing",
            "detail": f"assessing scene for query: {query}",
            "elapsed_sec": 0.0,
        })

    assessment = None
    if vlm_client is not None and cv_image is not None:
        try:
            if feedback_cb:
                feedback_cb({
                    "state": "vlm",
                    "detail": "calling VLM for scene assessment",
                    "elapsed_sec": time.time() - start,
                })
            prompt = _build_assessment_prompt(query, objects, vlm_scene)
            assessment = vlm_client.describe_scene(cv_image, prompt=prompt)
        except Exception as e:
            node.get_logger().warn(f"assess_scene VLM 실패: {e}")

    if not isinstance(assessment, dict):
        assessment = _fallback_assessment(query, objects, vlm_scene)

    normalized = normalize_assessment(assessment, query=query)
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
        "assess_scene 완료: "
        f"has_suspicious_person={normalized['has_suspicious_person']} "
        f"candidate_track_id={normalized['candidate_track_id']} "
        f"confidence={normalized['confidence']:.2f}"
    )
    return {
        "success": True,
        "assessment": normalized,
        "message": _assessment_message(normalized),
        "elapsed_sec": elapsed,
    }


def normalize_assessment(data: dict[str, Any], query: str = "") -> dict:
    """VLM 응답을 Action Result가 기대하는 고정 JSON shape로 정규화합니다."""
    result = dict(ASSESSMENT_SCHEMA_HINT)
    result["query"] = str(data.get("query") or query)
    result["has_suspicious_person"] = bool(data.get("has_suspicious_person", False))
    result["candidate_track_id"] = _safe_int(data.get("candidate_track_id"), -1)
    result["risk_level"] = _normalize_choice(
        data.get("risk_level"), {"none", "low", "medium", "high"}, "none"
    )
    result["reason"] = str(data.get("reason") or "판단 근거가 충분하지 않습니다.")
    result["confidence"] = _clamp_float(data.get("confidence"), 0.0, 1.0, 0.0)
    result["recommended_action"] = _normalize_choice(
        data.get("recommended_action"),
        {"ignore", "report", "inspect", "keep_distance"},
        "ignore",
    )
    return result


def _build_assessment_prompt(query: str, objects: list, vlm_scene: dict) -> str:
    objects_json = json.dumps(objects[:8], ensure_ascii=False)
    scene_json = json.dumps(vlm_scene or {}, ensure_ascii=False)
    return f"""\
너는 모바일 로봇의 scene assessment 모듈이다.
사용자 query와 현재 장면을 비교해 staged scenario cue만 판단하라.
실제 범죄, 성별, 나이, 신원 같은 민감한 추정은 하지 마라.

[사용자 query]
{query}

[YOLO tracked objects]
{objects_json}

[기존 VLM scene context]
{scene_json}

반드시 아래 JSON object 하나만 반환하라.
{{
  "query": "{query}",
  "has_suspicious_person": true 또는 false,
  "candidate_track_id": person track id 또는 -1,
  "risk_level": "none | low | medium | high",
  "reason": "옷 색상, 위치, 박스 근처 행동 등 관찰 가능한 단서만 사용한 짧은 한국어 근거",
  "confidence": 0.0~1.0,
  "recommended_action": "ignore | report | inspect | keep_distance"
}}
"""


def _fallback_assessment(query: str, objects: list, vlm_scene: dict) -> dict:
    people = [obj for obj in objects if obj.get("class") == "person"]
    social_hints = vlm_scene.get("social_hints", []) if isinstance(vlm_scene, dict) else []
    strongest_hint = max(
        social_hints,
        key=lambda h: float(h.get("confidence", 0.0) or 0.0),
        default={},
    )
    best_person = _nearest_person(people)
    has_person = best_person is not None
    is_suspicious_query = any(
        token in query.lower()
        for token in ("suspicious", "strange", "이상", "수상", "위험")
    )
    confidence = 0.35 if has_person and is_suspicious_query else 0.2
    reason = "VLM 이미지 판단 없이 detection/context만 사용했습니다."
    if strongest_hint:
        reason += f" social_hint={strongest_hint.get('type', 'unknown')}."
    return {
        "query": query,
        "has_suspicious_person": False,
        "candidate_track_id": _safe_int(best_person.get("id"), -1) if best_person else -1,
        "risk_level": "low" if has_person and is_suspicious_query else "none",
        "reason": reason,
        "confidence": confidence,
        "recommended_action": "inspect" if has_person and is_suspicious_query else "ignore",
    }


def _nearest_person(people: list) -> dict | None:
    with_range = [p for p in people if p.get("range_m") is not None]
    if with_range:
        return min(with_range, key=lambda p: float(p.get("range_m") or 999.0))
    return people[0] if people else None


def _assessment_message(assessment: dict) -> str:
    if assessment.get("has_suspicious_person"):
        return (
            f"수상한 후보 id={assessment.get('candidate_track_id')} "
            f"risk={assessment.get('risk_level')} "
            f"confidence={assessment.get('confidence'):.2f}: "
            f"{assessment.get('reason')}"
        )
    return (
        f"수상한 후보 없음 "
        f"confidence={assessment.get('confidence'):.2f}: "
        f"{assessment.get('reason')}"
    )


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


def _normalize_choice(value, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default
