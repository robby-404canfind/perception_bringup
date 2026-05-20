"""test_u02_yolo_track.py — YOLO26 .track() 단독 확인."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def default_video_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "track" / "sample.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26 .track() API를 ROS2 없이 단독 확인합니다."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=default_video_path(),
        help="입력 비디오 경로 (기본: test/data/track/sample.mp4)",
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
        "--limit",
        type=int,
        default=0,
        help="앞에서부터 확인할 프레임 수. 0이면 전체",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="시각화 프레임을 저장할 디렉토리 (선택)",
    )
    return parser.parse_args()


def validate_video(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"입력 비디오를 찾을 수 없습니다: {path}\n"
            "권장 위치(컨테이너): /home/hunav_webots_ws/src/perception_bringup/test/data/track/sample.mp4"
        )
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(
            f"지원하지 않는 비디오 확장자입니다: {path.suffix}\n"
            f"지원 형식: {sorted(VIDEO_SUFFIXES)}"
        )


def maybe_save_result(result, frame_index: int, output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None

    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_index:04d}.jpg"
    cv2.imwrite(str(output_path), result.plot())
    return output_path


def main() -> None:
    args = parse_args()
    validate_video(args.video)

    from ultralytics import YOLO

    print("=" * 60)
    print("u02 Step 2: YOLO26 .track() 단독 확인")
    print("=" * 60)
    print(f"  video:  {args.video}")
    print(f"  model:  {args.model}")
    print(f"  conf:   {args.conf}")
    print(f"  device: {args.device}")
    if args.limit > 0:
        print(f"  limit:  {args.limit} frames")

    model = YOLO(args.model)
    if args.device != "cpu":
        model.to(args.device)

    results = model.track(
        source=str(args.video),
        persist=True,
        conf=args.conf,
        verbose=False,
        stream=True,
    )

    seen_tracks: dict[int, dict[str, int | str]] = {}
    processed = 0

    for frame_index, result in enumerate(results, start=1):
        if args.limit > 0 and frame_index > args.limit:
            break

        processed += 1
        box_count = 0 if result.boxes is None else len(result.boxes)
        print(f"\n[Frame {frame_index:02d}] {box_count}개 탐지")

        if box_count == 0:
            continue

        for box in result.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            track_id = int(box.id) if box.id is not None else -1
            print(f"  ID={track_id:>3} {cls_name:<12} conf={conf:.2f}")

            if track_id >= 0:
                if track_id not in seen_tracks:
                    seen_tracks[track_id] = {"class": cls_name, "frames": 0}
                seen_tracks[track_id]["frames"] += 1

        saved_path = maybe_save_result(result, frame_index, args.output_dir)
        if saved_path:
            print(f"  시각화 저장: {saved_path}")

    print("\nTrack summary")
    print(f"  처리한 프레임 수: {processed}")
    if not seen_tracks:
        print("△ tracker ID가 부여된 객체가 없습니다.")
    else:
        for track_id in sorted(seen_tracks):
            item = seen_tracks[track_id]
            print(
                f"  ID={track_id:>3} "
                f"class={item['class']:<12} "
                f"seen_frames={item['frames']}"
            )

    print("\n✓ .track() 확인 완료")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n✗ 테스트 실패: {exc}")
        sys.exit(1)
