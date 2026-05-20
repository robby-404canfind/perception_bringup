"""test_u02_yolo_predict.py — YOLO26 .predict() 단독 확인."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_image_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "predict" / "sample.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26 .predict() API를 ROS2 없이 단독 확인합니다."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=default_image_path(),
        help="입력 이미지 경로 (기본: test/data/predict/sample.jpg)",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO 모델 파일명 또는 경로",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="confidence threshold",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="추론 장치 (기본: cuda:0, 예: cpu, cuda:0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="시각화 결과를 저장할 디렉토리 (선택)",
    )
    return parser.parse_args()


def validate_image(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"입력 이미지를 찾을 수 없습니다: {path}\n"
            "권장 위치(컨테이너): /home/hunav_webots_ws/src/perception_bringup/test/data/predict/"
        )
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(
            f"지원하지 않는 이미지 확장자입니다: {path.suffix}\n"
            f"지원 형식: {sorted(IMAGE_SUFFIXES)}"
        )


def maybe_save_result(result, image_path: Path, output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None

    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_predict{image_path.suffix.lower()}"
    cv2.imwrite(str(output_path), result.plot())
    return output_path


def main() -> None:
    args = parse_args()
    validate_image(args.image)
    from ultralytics import YOLO

    print("=" * 60)
    print("u02 Step 1: YOLO26 .predict() 단독 확인")
    print("=" * 60)
    print(f"  이미지: {args.image}")
    print(f"  모델:   {args.model}")
    print(f"  conf:   {args.conf}")
    print(f"  device: {args.device}")

    model = YOLO(args.model)
    if args.device != "cpu":
        model.to(args.device)

    results = model.predict(source=str(args.image), conf=args.conf, verbose=False)
    result = results[0]

    box_count = 0 if result.boxes is None else len(result.boxes)
    print(f"\n탐지 결과: {box_count}개")

    if box_count == 0:
        print("△ 감지된 객체가 없습니다. 다른 이미지나 더 낮은 conf를 시도해보세요.")
    else:
        for idx, box in enumerate(result.boxes, start=1):
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            conf = float(box.conf)
            x, y, w, h = box.xywh[0].tolist()
            print(
                f"  [{idx}] {cls_name:<12} conf={conf:.2f} "
                f"center=({x:.0f},{y:.0f}) size=({w:.0f}x{h:.0f})"
            )

    saved_path = maybe_save_result(result, args.image, args.output_dir)
    if saved_path:
        print(f"\n시각화 저장: {saved_path}")

    print("\n✓ .predict() 확인 완료")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n✗ 테스트 실패: {exc}")
        sys.exit(1)
