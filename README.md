# MediaPipe Blendshape Emoji Prototype Classifier

## 구조
- 입력: MediaPipe Face Landmarker `face_blendshapes`
- 재구성 특징: 28차원
- 클래스: 30개 이모지
- 분류: Weight 없는 Prototype RMS distance
- 개인 보정: Neutral median calibration
- 프레임 평활화: EMA

## 실행
```bash
pip install -r requirements.txt
python main.py
```

최초 실행 시 `models/face_landmarker.task` 자동 다운로드.

## Prototype
- `prototypes.py`: 30개 Prototype 정의
- `P.shape == (30, 28)`
- 행 순서: `constants.py`의 `EMOJIS`
- 열 순서: `constants.py`의 `FEATURE_NAMES`
- `prototype_matrix.csv`: 전체 행렬
- `prototype_vectors.txt`: P01~P30 배열

## 거리
`D_e = sqrt(mean((F - P_e)^2))`

## 주의
Prototype 수치: 초기 실험용 가설값.
최종 단계: 참가자 데이터 수집 후 클래스별 median 또는 mean으로 P 재산출 권장.

## 제한
- 🤭 손 가림: Face Blendshape만으로 직접 검출 불가
- 😢 / 😭 눈물: 직접 검출 불가
- 😋 / 😛 / 😜 / 🤪 혀: Face Landmarker 52 Blendshape에 직접 tongueOut 항목 없음
- 🤢 피부색: Blendshape만으로 표현 불가
- OpenCV `putText`: 컬러 이모지 표시 부적합
- `assets/emojis/01.png` ~ `30.png` 추가 시 PNG overlay 사용
