"""vlm_client.py — Edge/Cloud VLM 통합 클라이언트.

Ch03 LLMClient 패턴을 확장하여 이미지 입력을 지원합니다.
Social Navigation scene understanding의 핵심 엔진입니다.
mock_mode=True면 VLM 호출 없이 기본 응답을 반환합니다.
"""

import base64
import json
import os

import cv2
import numpy as np
from openai import OpenAI, APITimeoutError, APIConnectionError

_DEFAULT_SCENE_PROMPT = """\
너는 모바일 로봇의 시각 분석가이다.
첨부된 이미지를 분석하여, 로봇이 안전하게 이동하기 위해 알아야 할 정보를 JSON으로 반환하라.

출력 형식 (반드시 유효한 JSON만 출력):
{
  "scene_summary": "장면 한 줄 요약 (한국어)",
  "social_hints": [
    {
      "type": "avoid_between_people | prefer_side_pass | slow_down | clear_path",
      "reason": "이유 (영어, 짧게)",
      "confidence": 0.0~1.0,
      "side": "left | right (해당 시에만)",
      "offset_m": 0.0~3.0 (해당 시에만)
    }
  ]
}

social_hints가 없으면 빈 배열 []을 반환하라.
"""

_DEFAULT_BACKENDS = {
    "ollama": {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "api_key": "ollama",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
    },
}


class VLMClient:
    """Edge/Cloud VLM 통합 클라이언트 (mock fallback 포함)."""

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "qwen2.5vl:7b",
        mock_mode: bool = False,
        timeout: float = 30.0,
    ):
        self.mock_mode = mock_mode
        self.backend = backend
        self.model = model

        if not mock_mode:
            if backend not in _DEFAULT_BACKENDS:
                raise ValueError(f"Unknown VLM backend: {backend}")
            config = _DEFAULT_BACKENDS[backend].copy()
            if backend == "openrouter":
                config["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
            self.client = OpenAI(**config, timeout=timeout)

    def describe_scene(self, cv_image: np.ndarray, prompt: str | None = None) -> dict:
        """이미지 → 구조화된 장면 분석 JSON."""
        if self.mock_mode:
            return {
                "scene_summary": "mock: 정상 상황",
                "social_hints": [],
                "confidence": "low",
            }

        b64 = self._encode_image(cv_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or _DEFAULT_SCENE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            return {"scene_summary": "VLM 응답 파싱 실패", "social_hints": []}
        except (APITimeoutError, APIConnectionError) as e:
            return {"scene_summary": f"VLM 연결 실패: {e}", "social_hints": []}

    @staticmethod
    def _encode_image(cv_image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode()
