"""vlm_client.py — Edge/Cloud VLM 통합 클라이언트.

Ch03 LLMClient 패턴을 확장하여 이미지 입력을 지원합니다.
Social Navigation scene understanding의 핵심 엔진입니다.
"""

import base64
import contextlib
import fcntl
import json
import os
import tempfile
import time

import cv2
import numpy as np
from openai import OpenAI, APITimeoutError, APIConnectionError

_DEFAULT_SCENE_PROMPT = """\
너는 모바일 로봇의 시각 분석가이다.
첨부된 이미지를 분석하여, 로봇이 안전하게 이동하기 위해 알아야 할 정보를 JSON으로 반환하라.

출력 형식 (반드시 유효한 JSON만 출력):
{
  "scene_summary": "장면 한 줄 요약 (한국어)",
  "detour_instruction": true 또는 false,
  "social_hints": [
    {
      "type": "avoid_between_people",
      "reason": "이유 (영어, 짧게)",
      "confidence": 0.0~1.0,
      "side": "left | right (해당 시에만)",
      "offset_m": 0.0~3.0 (해당 시에만)
    }
  ]
}

type 필드는 반드시 avoid_between_people, prefer_side_pass, slow_down, clear_path 중 하나의 문자열만 사용하라.
두 사람이 대화 중이거나 통로를 함께 점유하면 detour_instruction=true와 avoid_between_people 또는 prefer_side_pass를 반환하라.
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
    """Edge/Cloud VLM 통합 클라이언트."""

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "qwen2.5vl:7b",
        timeout: float = 30.0,
        lock_timeout: float = 5.0,
    ):
        self.backend = backend
        self.model = model
        self.timeout = float(timeout)
        self.lock_timeout = float(lock_timeout)

        if backend not in _DEFAULT_BACKENDS:
            raise ValueError(f"Unknown VLM backend: {backend}")
        config = _DEFAULT_BACKENDS[backend].copy()
        if backend == "openrouter":
            config["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
        self.client = OpenAI(**config, timeout=timeout)

    def describe_scene(self, cv_image: np.ndarray, prompt: str | None = None) -> dict:
        """이미지 → 구조화된 장면 분석 JSON."""
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
            with self._serialized_call():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=500,
                )
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            return {"scene_summary": "VLM 응답 파싱 실패", "social_hints": []}
        except (APITimeoutError, APIConnectionError) as e:
            return {"scene_summary": f"VLM 연결 실패: {e}", "social_hints": []}
        except TimeoutError:
            return {"scene_summary": "VLM 호출 대기 중: 다른 VLM 요청 처리 중", "social_hints": []}

    @contextlib.contextmanager
    def _serialized_call(self):
        """Serialize VLM calls across Ch05 nodes to avoid local/backend contention."""
        lock_name = f"agentic_vla_vlm_{self.backend}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_name)
        deadline = time.time() + self.lock_timeout
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.time() >= deadline:
                        raise TimeoutError("VLM call lock timeout") from exc
                    time.sleep(0.1)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _encode_image(cv_image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode()
