from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from classifier import PrototypeClassifier
from constants import EMOJIS, NUM_FEATURES
from feature_extractor import BlendshapeFeatureExtractor, NeutralCalibrator, EMASmoother
from prototypes import P

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)


@dataclass(slots=True)
class AppConfig:
    camera_id: int = 0
    width: int = 1280
    height: int = 720
    calibration_frames: int = 45
    ema_alpha: float = 0.30
    model_path: str = str(PROJECT_DIR / 'models' / 'face_landmarker.task')
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class ModelManager:
    @staticmethod
    def ensure_model(path: str) -> Path:
        model_path = Path(path)
        if model_path.exists():
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'[model] downloading: {MODEL_URL}')
        urlretrieve(MODEL_URL, model_path)
        return model_path


class MediaPipeFaceBlendshape:
    __slots__ = ('_landmarker', '_rgb')

    def __init__(self, config: AppConfig) -> None:
        model_path = ModelManager.ensure_model(config.model_path)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._rgb: np.ndarray | None = None

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int):
        if self._rgb is None or self._rgb.shape != bgr_frame.shape:
            self._rgb = np.empty_like(bgr_frame)
        cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB, dst=self._rgb)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._rgb)
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()


class EmojiPngRenderer:
    """Optional PNG overlay: assets/emojis/01.png ... 30.png"""

    __slots__ = ('_images', 'size')

    def __init__(self, directory: str | None = None, size: int = 128) -> None:
        self.size = size
        self._images: list[np.ndarray | None] = [None] * len(EMOJIS)
        base = Path(directory) if directory is not None else PROJECT_DIR / 'assets' / 'emojis'

        for i in range(len(EMOJIS)):
            path = base / f'{i + 1:02d}.png'
            if not path.exists():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                continue
            self._images[i] = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    def draw(self, frame: np.ndarray, class_index: int, x: int = 20, y: int = 20) -> None:
        image = self._images[class_index]
        if image is None:
            cv2.putText(
                frame, f'class={class_index + 1:02d}',
                (x, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2, cv2.LINE_AA
            )
            return

        h, w = frame.shape[:2]
        x2, y2 = min(w, x + self.size), min(h, y + self.size)
        if x >= x2 or y >= y2:
            return

        roi = frame[y:y2, x:x2]
        overlay = image[: y2 - y, : x2 - x]
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        blended = roi.astype(np.float32) * (1.0 - alpha) + overlay[:, :, :3].astype(np.float32) * alpha
        np.copyto(roi, blended, casting='unsafe')


class RealtimeEmojiApp:
    __slots__ = (
        'config', 'detector', 'extractor', 'calibrator', 'smoother',
        'classifier', 'renderer', '_corrected'
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.detector = MediaPipeFaceBlendshape(config)
        self.extractor = BlendshapeFeatureExtractor()
        self.calibrator = NeutralCalibrator(config.calibration_frames, NUM_FEATURES)
        self.smoother = EMASmoother(config.ema_alpha, NUM_FEATURES)
        self.classifier = PrototypeClassifier(P)
        self.renderer = EmojiPngRenderer(directory="assets/emojis", size = 150)
        self._corrected = np.empty(NUM_FEATURES, dtype=np.float32)
    @staticmethod
    def get_face_box(landmarks, frame):
        h, w = frame.shape[:2]

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        box_w = x2 - x1
        box_h = y2 - y1

        padding_x = int(box_w * 0.15)
        padding_y = int(box_h * 0.20)

        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w - 1, x2 + padding_x)
        y2 = min(h - 1, y2 + padding_y)

        return x1, y1, x2, y2
    @staticmethod
    def _timestamp_ms() -> int:
        return monotonic_ns() // 1_000_000

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        if not cap.isOpened():
            self.detector.close()
            raise RuntimeError('Cannot open webcam')

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = self.detector.process(frame, self._timestamp_ms())

                if result.face_blendshapes and result.face_landmarks:
                    features = self.extractor.transform(result.face_blendshapes[0])

                    feature_text = np.array2string(features, precision=2, suppress_small=True, max_line_width=200)
                    if not self.calibrator.ready:
                        self.calibrator.add(features)
                        pct = int(self.calibrator.progress * 100)
                        cv2.putText(
                            frame, f'Neutral calibration: {pct}%',
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255), 2, cv2.LINE_AA
                        )
                    else:
                        corrected = self.calibrator.apply(features, self._corrected)
                        smoothed = self.smoother.update(corrected)
                        pred = self.classifier.predict(smoothed)

                        landmarks = result.face_landmarks[0]

                        x1, y1, x2, y2 = self.get_face_box(
                            landmarks,
                            frame
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        emoji_size = 150

                        emoji_x = (x1 + x2) // 2 - emoji_size // 2
                        emoji_y = y1 - emoji_size - 10

                        emoji_x = max(0, min(
                            frame.shape[1] - emoji_size,
                            emoji_x
                        ))

                        if emoji_y < 0:
                            emoji_y = y1 + 10

                        self.renderer.draw(
                            frame,
                            pred.index,
                            x=emoji_x,
                            y=emoji_y
                        )
                        cv2.putText(
                            frame,
                            f"Ft = {feature_text}",
                            (20, frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )
                        cv2.putText(
                            frame,
                            f'D1={pred.distance:.3f} D2={pred.second_distance:.3f} margin={pred.margin:.3f}',
                            (20, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA
                        )
                else:
                    cv2.putText(
                        frame, 'Face not detected',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA
                    )

                cv2.imshow('Realtime Emoji Prototype Classifier', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
        finally:
            cap.release()
            self.detector.close()
            cv2.destroyAllWindows()
