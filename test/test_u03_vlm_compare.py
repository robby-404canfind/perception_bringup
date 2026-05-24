"""test_u03_vlm_compare.py — 단일 이미지 VLM 지연 시간 비교."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception_bringup.vlm_client import VLMClient  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_HINT_TYPES = {
    "avoid_between_people",
    "prefer_side_pass",
    "slow_down",
    "clear_path",
}
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
DEFAULT_PROMPT = """\
너는 모바일 로봇의 시각 분석가입니다.
첨부 이미지를 분석하여 로봇이 안전하게 이동하기 위해 알아야 할 정보를 JSON으로만 반환하세요.

반환 형식:
{
  "scene_summary": "한국어 한 줄 요약",
  "social_hints": [
    {
      "type": "avoid_between_people",
      "reason": "짧은 이유",
      "confidence": 0.0
    }
  ]
}

type 필드는 반드시 avoid_between_people, prefer_side_pass, slow_down, clear_path 중 하나의 문자열만 사용하세요.
social_hints가 없으면 빈 배열 []을 반환하세요.
"""


def default_image_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "predict" / "sample.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="u03 VLM Client를 ROS2 없이 단일 이미지로 확인합니다."
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openrouter", "both"],
        default="both",
        help="테스트할 backend (기본: both)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=default_image_path(),
        help="입력 이미지 경로 (기본: test/data/predict/sample.jpg)",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama VLM 모델 (기본: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--openrouter-model",
        default=DEFAULT_OPENROUTER_MODEL,
        help=f"OpenRouter VLM 모델 (기본: {DEFAULT_OPENROUTER_MODEL})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="VLM에 전달할 프롬프트",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="VLM 호출 timeout 초 (기본: 90)",
    )
    return parser.parse_args()


def validate_image(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"입력 이미지를 찾을 수 없습니다: {path}\n"
            "권장 위치(컨테이너): "
            "/home/hunav_webots_ws/src/perception_bringup/test/data/predict/sample.jpg"
        )
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(
            f"지원하지 않는 이미지 확장자입니다: {path.suffix}\n"
            f"지원 형식: {sorted(IMAGE_SUFFIXES)}"
        )


def selected_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    cases = []
    if args.backend in ("ollama", "both"):
        cases.append(("ollama", args.ollama_model))
    if args.backend in ("openrouter", "both"):
        cases.append(("openrouter", args.openrouter_model))
    return cases


def validate_vlm_schema(result: dict) -> tuple[bool, list[str]]:
    errors = []
    if not isinstance(result, dict):
        return False, ["result가 dict가 아닙니다."]

    if not isinstance(result.get("scene_summary"), str):
        errors.append("scene_summary가 문자열이 아닙니다.")

    social_hints = result.get("social_hints")
    if not isinstance(social_hints, list):
        errors.append("social_hints가 배열이 아닙니다.")
        return False, errors

    for index, hint in enumerate(social_hints):
        if not isinstance(hint, dict):
            errors.append(f"social_hints[{index}]가 객체가 아닙니다.")
            continue

        hint_type = hint.get("type")
        if hint_type not in ALLOWED_HINT_TYPES:
            errors.append(
                f"social_hints[{index}].type이 허용 목록 밖입니다: {hint_type!r}"
            )

        confidence = hint.get("confidence")
        if not isinstance(confidence, (int, float)):
            errors.append(f"social_hints[{index}].confidence가 숫자가 아닙니다.")
        elif not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"social_hints[{index}].confidence가 0.0~1.0 밖입니다.")

    return not errors, errors


def describe_scene(
    backend: str,
    model: str,
    image,
    image_path: Path,
    prompt: str,
    timeout: float,
) -> dict:
    client = VLMClient(
        backend=backend,
        model=model,
        timeout=timeout,
    )

    started = time.perf_counter()
    result = client.describe_scene(image, prompt=prompt)
    elapsed = time.perf_counter() - started
    schema_ok, schema_errors = validate_vlm_schema(result)

    return {
        "backend": backend,
        "model": model,
        "image": str(image_path),
        "latency_sec": round(elapsed, 3),
        "schema_ok": schema_ok,
        "schema_errors": schema_errors,
        "result": result,
    }


def main() -> None:
    args = parse_args()
    validate_image(args.image)

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"이미지를 OpenCV로 읽지 못했습니다: {args.image}")

    print("=" * 60)
    print("u03 Step 1: VLM 단일 이미지 비교")
    print("=" * 60)
    print(f"  image: {args.image}")
    print(f"  backend: {args.backend}")
    print(f"  timeout: {args.timeout:.0f}s")

    results = []
    for backend, model in selected_cases(args):
        if backend == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
            message = "OPENROUTER_API_KEY 환경변수가 없어 OpenRouter 테스트를 건너뜁니다."
            if args.backend == "openrouter":
                raise RuntimeError(message)
            print(f"\n△ {message}")
            continue

        print("\n" + "-" * 60)
        print(f"[{backend}] {model}")
        print("-" * 60)
        data = describe_scene(
            backend=backend,
            model=model,
            image=image,
            image_path=args.image,
            prompt=args.prompt,
            timeout=args.timeout,
        )
        results.append(data)
        print(f"  latency: {data['latency_sec']:.3f}s")
        print(f"  schema_ok: {data['schema_ok']}")
        if data["schema_errors"]:
            print("  schema_errors:")
            for error in data["schema_errors"]:
                print(f"    - {error}")
        print(
            json.dumps(
                data["result"],
                ensure_ascii=False,
                indent=2,
            )
        )

    print("\n" + "=" * 60)
    print("u03 VLM 비교 결과 JSON")
    print("=" * 60)
    print(json.dumps(results, ensure_ascii=False, indent=2))

    if len(results) >= 2:
        local = next((item for item in results if item["backend"] == "ollama"), None)
        cloud = next((item for item in results if item["backend"] == "openrouter"), None)
        if local and cloud:
            print(
                "\n비교: "
                f"Ollama {local['latency_sec']:.3f}s vs "
                f"OpenRouter {cloud['latency_sec']:.3f}s"
            )

    print("\n✓ u03 VLM 단일 이미지 비교 완료")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n✗ 테스트 실패: {exc}")
        sys.exit(1)
