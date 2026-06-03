"""perception_context_builder_node.py — YOLO + Depth + VLM 통합 Context publish 노드.

/perception/detections를 구독하여 /perception/context/raw (JSON)와
/perception/context/summary (한국어 요약)를 publish합니다.
VLM Trigger 조건이 충족되면 최신 RGB 프레임으로 VLM을 호출합니다.
별도의 snapshot request를 받으면 현재 RGB 프레임을 sensor_msgs/Image로 publish합니다.
"""

import json
import time
from copy import deepcopy
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
except ImportError:
    PILImage = None
    ImageDraw = None
    ImageFont = None
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .perception_trigger import PerceptionTrigger
from .vlm_client import VLMClient


class PerceptionContextBuilderNode(Node):

    def __init__(self):
        super().__init__("perception_context_builder")

        # detector 결과에 VLM 판단과 debug 출력을 붙여 context로 정규화합니다.
        self.declare_parameter("vlm_backend", "ollama")
        self.declare_parameter("vlm_model", "qwen2.5vl:7b")
        self.declare_parameter("min_trigger_interval", 5.0)
        self.declare_parameter("image_topic", "/Tiago_Lite/Astra_rgb/image_color")
        self.declare_parameter("detection_topic", "/perception/detections")
        self.declare_parameter("context_raw_topic", "/perception/context/raw")
        self.declare_parameter("context_summary_topic", "/perception/context/summary")
        self.declare_parameter("snapshot_request_topic", "/perception/snapshot/request")
        self.declare_parameter("snapshot_image_topic", "/perception/snapshot/image")
        self.declare_parameter("snapshot_info_topic", "/perception/snapshot/info")
        self.declare_parameter("save_snapshots", True)
        self.declare_parameter("snapshot_save_dir", "snapshots")
        self.declare_parameter(
            "system2_debug_image_topic", "/perception/system2/debug_image"
        )
        self.declare_parameter("system2_debug_state_topic", "/system2/debug_state")
        self.declare_parameter("system2_debug_font_path", "")

        vlm_backend = self.get_parameter("vlm_backend").value
        vlm_model = self.get_parameter("vlm_model").value
        min_interval = self.get_parameter("min_trigger_interval").value

        self.vlm = VLMClient(backend=vlm_backend, model=vlm_model)
        self.trigger = PerceptionTrigger(min_interval=min_interval)
        self.cv_bridge = CvBridge()

        # snapshot/VLM 호출은 callback들이 갱신한 최신 상태를 사용합니다.
        self.latest_cv_image = None
        self.latest_raw_image = None
        self.latest_image_header = None
        self.latest_image_encoding = "bgr8"
        self._latest_vlm_result: dict | None = None
        self._latest_vlm_time: float = 0.0
        self._latest_objects: list = []
        self._latest_frame_w = 0
        self._latest_frame_h = 0
        self._latest_image_w = 0
        self._latest_image_h = 0
        self._vlm_lock = Lock()
        self._vlm_in_flight = False
        self._system2_debug_state: dict = {}
        self._system2_debug_font = self._load_debug_font(
            str(self.get_parameter("system2_debug_font_path").value or "")
        )
        if self._system2_debug_font is None:
            self.get_logger().warn(
                "System2 debug image 한글 폰트를 찾지 못했습니다. "
                "fonts-noto-cjk와 python3-pil 설치를 확인하세요."
            )

        detection_topic = self.get_parameter("detection_topic").value
        context_raw_topic = self.get_parameter("context_raw_topic").value
        context_summary_topic = self.get_parameter("context_summary_topic").value
        snapshot_request_topic = self.get_parameter("snapshot_request_topic").value
        snapshot_image_topic = self.get_parameter("snapshot_image_topic").value
        snapshot_info_topic = self.get_parameter("snapshot_info_topic").value
        self.save_snapshots = bool(self.get_parameter("save_snapshots").value)
        self.snapshot_save_dir = self._resolve_snapshot_save_dir(
            str(self.get_parameter("snapshot_save_dir").value)
        )
        system2_debug_image_topic = self.get_parameter(
            "system2_debug_image_topic"
        ).value
        system2_debug_state_topic = self.get_parameter(
            "system2_debug_state_topic"
        ).value

        # 구독
        self.create_subscription(String, detection_topic, self._on_detections, 10)
        image_topic = self.get_parameter("image_topic").value
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_subscription(String, snapshot_request_topic, self._on_snapshot_req, 10)
        self.create_subscription(String, system2_debug_state_topic, self._on_system2_debug, 10)

        # publish
        self._pub_raw = self.create_publisher(String, context_raw_topic, 10)
        self._pub_summary = self.create_publisher(String, context_summary_topic, 10)
        self._pub_snap_img = self.create_publisher(Image, snapshot_image_topic, 10)
        self._pub_snap_info = self.create_publisher(String, snapshot_info_topic, 10)
        self._pub_debug_img = self.create_publisher(
            Image, system2_debug_image_topic, 10
        )

        self.get_logger().info(
            f"PerceptionContextBuilder 시작 (VLM={vlm_backend}/{vlm_model})"
        )
        if self.save_snapshots:
            self.get_logger().info(f"Snapshot 저장 경로: {self.snapshot_save_dir}")

    def _on_image(self, msg: Image):
        try:
            self.latest_raw_image = np.ascontiguousarray(
                self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").copy()
            )
            self.latest_cv_image = np.ascontiguousarray(
                self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
            )
            self.latest_image_header = msg.header
            self.latest_image_encoding = msg.encoding or "passthrough"
            self._latest_image_w = int(msg.width)
            self._latest_image_h = int(msg.height)
        except Exception as e:
            self.get_logger().warn(
                f"RGB 이미지 변환 실패: {e}", throttle_duration_sec=5.0
            )

    def _on_detections(self, msg: String):
        # Detection frame마다 context와 debug image를 갱신합니다.
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        objects = data.get("objects", [])
        self._latest_objects = deepcopy(objects)
        self._latest_frame_w = int(data.get("frame_w", 0) or 0)
        self._latest_frame_h = int(data.get("frame_h", 0) or 0)

        # VLM Trigger 평가
        trigger_reason = self.trigger.evaluate(objects)
        if trigger_reason and self.latest_cv_image is not None:
            with self._vlm_lock:
                if self._vlm_in_flight:
                    self.get_logger().debug(
                        f"VLM 호출 스킵: 이전 요청 처리 중 ({trigger_reason})"
                    )
                else:
                    self._vlm_in_flight = True
                    Thread(
                        target=self._call_vlm,
                        args=(self.latest_cv_image.copy(), trigger_reason),
                        daemon=True,
                    ).start()

        # context/raw 조립
        vlm_scene = self._latest_vlm_result or {}
        vlm_age = (
            round(time.time() - self._latest_vlm_time, 1)
            if self._latest_vlm_result
            else None
        )

        raw = {
            "frame_w": data.get("frame_w", 640),
            "frame_h": data.get("frame_h", 480),
            "objects": objects,
            "vlm_scene": vlm_scene,
            "vlm_triggered": trigger_reason is not None,
            "vlm_age_sec": vlm_age,
        }
        self._pub_raw.publish(String(data=json.dumps(raw, ensure_ascii=False)))

        # context/summary
        summary = self._to_korean_summary(objects, vlm_scene)
        self._pub_summary.publish(String(data=summary))

        self._publish_debug_image(objects, vlm_scene, vlm_age)

    def _call_vlm(self, cv_image: np.ndarray, reason: str):
        try:
            self.get_logger().info(f"VLM 호출: {reason}")
            result = self.vlm.describe_scene(cv_image)
            self._latest_vlm_result = result
            self._latest_vlm_time = time.time()
            self.get_logger().info(
                f"VLM 응답: {result.get('scene_summary', 'N/A')}"
            )
        finally:
            with self._vlm_lock:
                self._vlm_in_flight = False

    def _on_system2_debug(self, msg: String):
        try:
            data = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            self._system2_debug_state = data

    def _on_snapshot_req(self, msg: String):
        # Snapshot은 stream이 아니라 요청 기반 단발성 출력입니다.
        if self.latest_cv_image is None:
            self.get_logger().warn("Snapshot 요청이지만 이미지 없음")
            return

        try:
            req_data = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            req_data = {}

        raw_source = (
            self.latest_raw_image
            if self.latest_raw_image is not None
            else self.latest_cv_image
        )
        raw_image = np.ascontiguousarray(raw_source.copy())
        debug_source_image = np.ascontiguousarray(self.latest_cv_image.copy())
        image_encoding = self.latest_image_encoding or "passthrough"
        image_h, image_w = raw_image.shape[:2]
        objects = deepcopy(self._latest_objects)
        vlm_scene = deepcopy(self._latest_vlm_result or {})
        vlm_age = (
            round(time.time() - self._latest_vlm_time, 1)
            if self._latest_vlm_result
            else None
        )

        # 기존 topic 호환성을 위해 snapshot publish는 bgr8 Image로 유지합니다.
        snap_msg = self.cv_bridge.cv2_to_imgmsg(debug_source_image, encoding="bgr8")
        if self.latest_image_header is not None:
            snap_msg.header = self.latest_image_header
        self._pub_snap_img.publish(snap_msg)

        # 스냅샷 정보 publish
        info = {
            "snapshot_id": req_data.get("snapshot_id", ""),
            "mission_id": req_data.get("mission_id", ""),
            "request_id": req_data.get("request_id", ""),
            "requester": req_data.get("requester", ""),
            "reason": req_data.get("reason", ""),
            "message": req_data.get("message", ""),
            "objects": objects,
            "vlm_scene": vlm_scene,
            "save_snapshots": self.save_snapshots,
            "image_encoding": image_encoding,
            "image_w": image_w,
            "image_h": image_h,
        }
        if self.save_snapshots:
            save_paths = self._snapshot_file_paths(info["snapshot_id"])
            info["saved_files"] = {
                "raw": str(save_paths["raw"]),
                "debug": str(save_paths["debug"]),
                "metadata": str(save_paths["metadata"]),
            }
            detection_frame_w = self._latest_frame_w or image_w
            detection_frame_h = self._latest_frame_h or image_h
            Thread(
                target=self._save_snapshot_files,
                args=(
                    raw_image,
                    image_encoding,
                    debug_source_image,
                    objects,
                    vlm_scene,
                    vlm_age,
                    image_w,
                    image_h,
                    detection_frame_w,
                    detection_frame_h,
                    deepcopy(info),
                    save_paths,
                ),
                daemon=True,
            ).start()

        self._pub_snap_info.publish(String(data=json.dumps(info, ensure_ascii=False)))

    def _save_snapshot_files(
        self,
        raw_image: np.ndarray,
        image_encoding: str,
        debug_source_image: np.ndarray,
        objects: list,
        vlm_scene: dict,
        vlm_age: float | None,
        image_w: int,
        image_h: int,
        detection_frame_w: int,
        detection_frame_h: int,
        metadata: dict,
        save_paths: dict,
    ) -> None:
        try:
            self.snapshot_save_dir.mkdir(parents=True, exist_ok=True)
            debug_image = self._build_debug_image(
                debug_source_image,
                objects,
                vlm_scene,
                vlm_age,
                source_frame_w=detection_frame_w,
                source_frame_h=detection_frame_h,
            )

            raw_png_image = self._image_for_png(raw_image, image_encoding)
            raw_ok = cv2.imwrite(str(save_paths["raw"]), raw_png_image)
            debug_ok = cv2.imwrite(str(save_paths["debug"]), debug_image)
            if not raw_ok or not debug_ok:
                raise RuntimeError("cv2.imwrite returned False")

            metadata.update(
                {
                    "timestamp": save_paths["timestamp"],
                    "saved_at_unix_sec": time.time(),
                    "frame_w": image_w,
                    "frame_h": image_h,
                    "detection_frame_w": detection_frame_w,
                    "detection_frame_h": detection_frame_h,
                    "image_encoding": image_encoding,
                    "vlm_age_sec": vlm_age,
                    "system2_debug_state": deepcopy(self._system2_debug_state),
                }
            )
            with save_paths["metadata"].open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.get_logger().info(f"Snapshot 파일 저장: {save_paths['metadata']}")
        except Exception as e:
            self.get_logger().warn(f"Snapshot 파일 저장 실패: {e}")

    def _publish_debug_image(
        self, objects: list, vlm_scene: dict, vlm_age: float | None
    ) -> None:
        if self.latest_cv_image is None:
            return

        debug_image = self._build_debug_image(
            self.latest_cv_image,
            objects,
            vlm_scene,
            vlm_age,
            source_frame_w=self._latest_frame_w or self._latest_image_w,
            source_frame_h=self._latest_frame_h or self._latest_image_h,
        )
        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        if self.latest_image_header is not None:
            debug_msg.header = self.latest_image_header
        self._pub_debug_img.publish(debug_msg)

    def _build_debug_image(
        self,
        cv_image: np.ndarray,
        objects: list,
        vlm_scene: dict,
        vlm_age: float | None,
        source_frame_w: int | None = None,
        source_frame_h: int | None = None,
    ) -> np.ndarray:
        debug_image = np.ascontiguousarray(cv_image.copy())
        h, w = debug_image.shape[:2]
        scale_x = w / source_frame_w if source_frame_w else 1.0
        scale_y = h / source_frame_h if source_frame_h else 1.0

        for obj in objects:
            bbox = obj.get("bbox") or {}
            x = int(round(float(bbox.get("x", 0)) * scale_x))
            y = int(round(float(bbox.get("y", 0)) * scale_y))
            bw = int(round(float(bbox.get("w", 0)) * scale_x))
            bh = int(round(float(bbox.get("h", 0)) * scale_y))
            if bw <= 0 or bh <= 0:
                continue

            x0 = max(0, min(w - 1, x))
            y0 = max(0, min(h - 1, y))
            x1 = max(0, min(w - 1, x + bw))
            y1 = max(0, min(h - 1, y + bh))
            if x1 <= x0 or y1 <= y0:
                continue

            cv2.rectangle(debug_image, (x0, y0), (x1, y1), (0, 220, 255), 2)
            label = self._object_label(obj)
            self._draw_text(debug_image, label, (x0, max(18, y0 - 6)), scale=0.5)

        self._draw_vlm_overlay(debug_image, vlm_scene, vlm_age)
        self._draw_system2_overlay(debug_image, self._system2_debug_state)
        return debug_image

    @staticmethod
    def _object_label(obj: dict) -> str:
        obj_id = obj.get("id", "?")
        cls = obj.get("class", "unknown")
        confidence = obj.get("confidence")
        conf_text = (
            f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "?"
        )
        range_m = obj.get("range_m")
        range_text = (
            f"{range_m:.2f}m" if isinstance(range_m, (int, float)) else "range=?"
        )
        return f"id={obj_id} {cls} conf={conf_text} {range_text}"

    @staticmethod
    def _vlm_overlay_lines(
        vlm_scene: dict,
        vlm_age: float | None,
    ) -> tuple[str, str]:
        age_text = f"{vlm_age:.1f}s" if isinstance(vlm_age, (int, float)) else "n/a"
        hints_text = PerceptionContextBuilderNode._social_hints_overlay_text(
            vlm_scene.get("social_hints") if isinstance(vlm_scene, dict) else None
        )
        scene_summary = " ".join(str(vlm_scene.get("scene_summary") or "").split())
        summary_text = scene_summary or "VLM 요약 없음"
        return f"VLM age={age_text} social_hints={hints_text}", summary_text

    @staticmethod
    def _social_hints_overlay_text(social_hints) -> str:
        if not isinstance(social_hints, list):
            return "none"

        hint_types = []
        for hint in social_hints:
            if not isinstance(hint, dict) or not hint.get("type"):
                continue
            hint_type = " ".join(str(hint["type"]).split())
            if hint_type:
                hint_types.append(hint_type)

        if not hint_types:
            return "none"
        return ",".join(hint_types[:3])

    def _draw_vlm_overlay(
        self,
        image: np.ndarray,
        vlm_scene: dict,
        vlm_age: float | None,
    ) -> None:
        lines = self._vlm_overlay_lines(vlm_scene, vlm_age)
        if self._system2_debug_font is not None and PILImage is not None:
            self._draw_unicode_overlay(image, lines, self._system2_debug_font)
            return

        self._draw_text(
            image,
            lines[0],
            (8, 24),
            scale=0.55,
            bg_color=(0, 0, 220),
        )
        self._draw_text(
            image,
            lines[1],
            (8, 48),
            scale=0.55,
            bg_color=(0, 0, 220),
        )

    def _draw_system2_overlay(self, image: np.ndarray, state: dict) -> None:
        if not isinstance(state, dict) or not state:
            return

        lines = self._system2_overlay_lines(state)
        if not lines:
            return

        if self._system2_debug_font is not None and PILImage is not None:
            self._draw_unicode_overlay_block(
                image,
                lines,
                self._system2_debug_font,
                anchor="bottom",
                fill=(0, 95, 165),
            )
            return

        h = image.shape[0]
        y = max(24, h - 24 * len(lines) - 8)
        for line in lines:
            self._draw_text(
                image,
                line,
                (8, y),
                scale=0.5,
                bg_color=(0, 95, 165),
            )
            y += 24

    @staticmethod
    def _system2_overlay_lines(state: dict) -> list[str]:
        mission_id = str(state.get("mission_id") or "")
        phase = str(state.get("phase") or "")
        current_index = state.get("current_step_index")
        total_steps = state.get("total_steps")
        current_step = state.get("current_step") or {}
        plan_steps = state.get("plan_steps") or []

        header = "System2"
        if mission_id:
            header += f" {mission_id}"
        if phase:
            header += f" {phase}"
        if isinstance(current_index, int) and isinstance(total_steps, int) and total_steps > 0:
            header += f" step {current_index + 1}/{total_steps}"

        lines = [header]
        if isinstance(current_step, dict) and current_step.get("task"):
            lines.append(
                "Now: "
                + PerceptionContextBuilderNode._format_step_for_overlay(current_step)
            )

        if isinstance(plan_steps, list) and plan_steps:
            compact = []
            for idx, step in enumerate(plan_steps[:5]):
                prefix = ">" if isinstance(current_index, int) and idx == current_index else " "
                compact.append(
                    f"{prefix}{idx + 1}:{PerceptionContextBuilderNode._format_step_for_overlay(step)}"
                )
            lines.extend(compact[:4])
        return lines[:6]

    @staticmethod
    def _format_step_for_overlay(step: dict) -> str:
        task = str(step.get("task") or "?")
        params = step.get("params") or {}
        if not isinstance(params, dict):
            return task
        for key in (
            "location",
            "area",
            "status",
            "target_class",
            "query",
            "target_query",
        ):
            if key in params and params[key] not in (None, ""):
                value = " ".join(str(params[key]).split())
                if len(value) > 34:
                    value = value[:31].rstrip() + "..."
                return f"{task}({value})"
        return task

    @staticmethod
    def _image_for_png(image: np.ndarray, encoding: str) -> np.ndarray:
        normalized = (encoding or "").lower()
        if image.ndim < 3:
            return image

        if normalized in ("rgb8", "rgb16"):
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if normalized in ("rgba8", "rgba16"):
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        return image

    @staticmethod
    def _draw_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float = 0.5,
        bg_color: tuple[int, int, int] = (0, 0, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        text = text if text.isascii() else text.encode("ascii", "ignore").decode()
        x, y = origin
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        h, w = image.shape[:2]
        x = max(0, min(w - tw - 6, x))
        y = max(th + 4, min(h - baseline - 4, y))
        top_left = (max(0, x - 3), max(0, y - th - 5))
        bottom_right = (min(w - 1, x + tw + 3), min(h - 1, y + baseline + 3))
        cv2.rectangle(image, top_left, bottom_right, bg_color, -1)
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _load_debug_font(font_path: str, size: int = 18):
        if ImageFont is None:
            return None

        candidates = []
        if font_path:
            candidates.append(Path(font_path).expanduser())
        candidates.extend(
            [
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size=size)
                except OSError:
                    continue
        return None

    @staticmethod
    def _draw_unicode_overlay(
        image: np.ndarray,
        lines: tuple[str, str],
        font,
    ) -> None:
        PerceptionContextBuilderNode._draw_unicode_overlay_block(
            image,
            list(lines),
            font,
            anchor="top",
            fill=(210, 0, 0),
        )

    @staticmethod
    def _draw_unicode_overlay_block(
        image: np.ndarray,
        lines: list[str],
        font,
        anchor: str = "top",
        fill: tuple[int, int, int] = (210, 0, 0),
    ) -> None:
        h, w = image.shape[:2]
        max_text_width = max(80, w - 28)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        fitted_lines = [
            PerceptionContextBuilderNode._fit_text_to_width(
                draw,
                line,
                font,
                max_text_width,
            )
            for line in lines
        ]

        line_boxes = [
            draw.textbbox((0, 0), line, font=font)
            for line in fitted_lines
        ]
        line_widths = [box[2] - box[0] for box in line_boxes]
        line_heights = [box[3] - box[1] for box in line_boxes]
        pad_x = 7
        pad_y = 5
        line_gap = 4
        x = 8
        y = 8 if anchor == "top" else max(8, h - (sum(line_heights) + line_gap * (len(line_heights) - 1) + pad_y * 2) - 8)
        rect_w = min(w - x - 1, max(line_widths) + pad_x * 2)
        rect_h = min(
            h - y - 1,
            sum(line_heights) + line_gap * max(0, len(line_heights) - 1) + pad_y * 2,
        )

        draw.rectangle(
            (x, y, x + rect_w, y + rect_h),
            fill=fill,
        )
        cursor_y = y + pad_y
        for line, box, line_h in zip(fitted_lines, line_boxes, line_heights):
            text_y = cursor_y - box[1]
            draw.text(
                (x + pad_x, text_y),
                line,
                font=font,
                fill=(255, 255, 255),
            )
            cursor_y += line_h + line_gap

        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _fit_text_to_width(draw, text: str, font, max_width: int) -> str:
        if draw.textlength(text, font=font) <= max_width:
            return text

        suffix = "..."
        suffix_width = draw.textlength(suffix, font=font)
        target_width = max(0, max_width - suffix_width)
        lo = 0
        hi = len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if draw.textlength(text[:mid], font=font) <= target_width:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + suffix

    @staticmethod
    def _resolve_snapshot_save_dir(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parents[1]
        return repo_root / path

    def _snapshot_file_paths(self, snapshot_id: str) -> dict:
        timestamp = self._timestamp_for_filename()
        safe_snapshot_id = self._safe_filename_part(snapshot_id or "snapshot")
        base = f"{timestamp}_{safe_snapshot_id}"
        return {
            "timestamp": timestamp,
            "raw": self.snapshot_save_dir / f"{base}_raw.png",
            "debug": self.snapshot_save_dir / f"{base}_debug.png",
            "metadata": self.snapshot_save_dir / f"{base}.json",
        }

    @staticmethod
    def _timestamp_for_filename() -> str:
        now = time.time()
        base = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        millis = int((now % 1.0) * 1000)
        return f"{base}_{millis:03d}"

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        cleaned = "".join(
            ch if ch.isascii() and (ch.isalnum() or ch in ("-", "_", ".")) else "_"
            for ch in str(value)
        ).strip("._")
        return (cleaned or "snapshot")[:80]

    @staticmethod
    def _to_korean_summary(objects: list, vlm_scene: dict) -> str:
        parts = []
        for obj in objects:
            obj_id = obj.get("id", "?")
            cls = obj.get("class", "unknown")
            direction = obj.get("direction", "?")
            range_m = obj.get("range_m")
            range_str = f"{range_m}m" if range_m is not None else "거리 미확인"
            parts.append(f"{obj_id}번 {cls}, 화면 {direction}, {range_str}")

        scene_summary = vlm_scene.get("scene_summary")
        if scene_summary:
            parts.append(f"VLM: {scene_summary}")

        return " / ".join(parts) if parts else "감지된 객체 없음"


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionContextBuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
