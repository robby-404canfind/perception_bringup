# Test Data 안내

u02 실습에서 사용하는 샘플 입력은 이 디렉토리 아래에 둡니다.

## 권장 구조

- `predict/sample.jpg`: `.predict()` 단일 이미지 확인용 사진
- `track/sample.mp4`: `.track()` tracking 확인용 짧은 비디오

## 배치 원칙

- 실행 스크립트가 직접 읽는 입력 파일은 `test/data/` 아래에 둡니다.
- 강의 노트나 슬라이드에 직접 삽입할 이미지는 `assets/`에 두는 편이 관리하기 쉽습니다.
- 같은 장면을 실습과 문서에 모두 사용한다면, 원본 입력은 `test/data/`에 두고 문서용 이미지만 `assets/`에 별도로 둡니다.
