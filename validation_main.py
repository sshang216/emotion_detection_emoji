from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from app import AppConfig, EmojiPngRenderer, ModelManager, PROJECT_DIR
from classifier import PrototypeClassifier, Prediction
from constants import EMOJIS, FEATURE_NAMES, NUM_FEATURES
from feature_extractor import BlendshapeFeatureExtractor
from prototypes import P


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@dataclass(slots=True, frozen=True)
class ValidationResult:
    image_path: str
    predicted_emoji: str
    predicted_class: int
    distance: float
    second_emoji: str
    second_class: int
    second_distance: float
    margin: float
    features: np.ndarray


class MediaPipeImageBlendshape:
    """Static-image Face Landmarker. Existing VIDEO detector is left untouched."""

    __slots__ = ('_landmarker',)

    def __init__(self, config: AppConfig) -> None:
        model_path = ModelManager.ensure_model(config.model_path)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(self, bgr_image: np.ndarray):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(mp_image)

    def close(self) -> None:
        self._landmarker.close()


class ImageEmojiValidator:
    """Classifies still facial-expression images with the existing project pipeline."""

    __slots__ = (
        'detector', 'extractor', 'classifier', 'renderer',
        '_neutral', '_work', '_emoji_size'
    )

    def __init__(self, config: AppConfig, emoji_size: int = 128) -> None:
        self.detector = MediaPipeImageBlendshape(config)
        self.extractor = BlendshapeFeatureExtractor()
        self.classifier = PrototypeClassifier(P)
        self.renderer = EmojiPngRenderer(size=emoji_size)
        self._neutral: np.ndarray | None = None
        self._work = np.empty(NUM_FEATURES, dtype=np.float32)
        self._emoji_size = emoji_size

    def close(self) -> None:
        self.detector.close()

    def set_neutral_image(self, image_path: Path) -> None:
        image = self._read_image(image_path)
        result = self.detector.process(image)
        features = self._extract_features(result, image_path)
        # BlendshapeFeatureExtractor reuses its internal output array.
        self._neutral = features.copy()

    def classify(self, image_path: Path) -> tuple[ValidationResult, np.ndarray]:
        image = self._read_image(image_path)
        mp_result = self.detector.process(image)
        raw_features = self._extract_features(mp_result, image_path)

        if self._neutral is None:
            features = raw_features.copy()
        else:
            np.subtract(raw_features, self._neutral, out=self._work)
            np.clip(self._work, 0.0, 1.0, out=self._work)
            features = self._work.copy()

        pred = self.classifier.predict(features)
        annotated = image.copy()
        self._draw_result(annotated, mp_result.face_landmarks[0], pred)

        second_class = EMOJIS.index(pred.second_emoji) + 1
        validation = ValidationResult(
            image_path=str(image_path),
            predicted_emoji=pred.emoji,
            predicted_class=pred.index + 1,
            distance=pred.distance,
            second_emoji=pred.second_emoji,
            second_class=second_class,
            second_distance=pred.second_distance,
            margin=pred.margin,
            features=features,
        )
        return validation, annotated

    @staticmethod
    def _read_image(image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Cannot read image: {image_path}')
        return image

    def _extract_features(self, result, image_path: Path) -> np.ndarray:
        if not result.face_blendshapes or not result.face_landmarks:
            raise RuntimeError(f'Face not detected: {image_path}')
        return self.extractor.transform(result.face_blendshapes[0])

    @staticmethod
    def _face_box(landmarks, image: np.ndarray) -> tuple[int, int, int, int]:
        h, w = image.shape[:2]
        xs = np.fromiter((lm.x for lm in landmarks), dtype=np.float32)
        ys = np.fromiter((lm.y for lm in landmarks), dtype=np.float32)

        x1 = int(np.clip(xs.min() * w, 0, w - 1))
        y1 = int(np.clip(ys.min() * h, 0, h - 1))
        x2 = int(np.clip(xs.max() * w, 0, w - 1))
        y2 = int(np.clip(ys.max() * h, 0, h - 1))

        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * 0.12)
        py = int(bh * 0.15)

        return (
            max(0, x1 - px),
            max(0, y1 - py),
            min(w - 1, x2 + px),
            min(h - 1, y2 + py),
        )

    def _draw_result(self, image: np.ndarray, landmarks, pred: Prediction) -> None:
        x1, y1, x2, y2 = self._face_box(landmarks, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        emoji_x = (x1 + x2) // 2 - self._emoji_size // 2
        emoji_y = y1 - self._emoji_size - 8
        emoji_x = max(0, min(image.shape[1] - self._emoji_size, emoji_x))
        if emoji_y < 0:
            emoji_y = min(image.shape[0] - self._emoji_size, y2 + 8)

        self.renderer.draw(image, pred.index, x=emoji_x, y=emoji_y)

        text = (
            f'class={pred.index + 1:02d}  '
            f'D1={pred.distance:.3f}  '
            f'D2={pred.second_distance:.3f}  '
            f'margin={pred.margin:.3f}'
        )
        text_y = min(image.shape[0] - 12, y2 + 30)
        cv2.putText(
            image, text, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2, cv2.LINE_AA,
        )


class ValidationCsvWriter:
    __slots__ = ('path', '_header_written')

    BASE_COLUMNS = [
        'subject_id',
        'image_name',
        'image_path',
        'human_emoji',
        'human_class',
        'predicted_emoji',
        'predicted_class',
        'D1',
        'second_emoji',
        'second_class',
        'D2',
        'margin',
        'calibration_mode',
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def append(
        self,
        result: ValidationResult,
        subject_id: str,
        calibration_mode: str,
    ) -> None:
        columns = self.BASE_COLUMNS + [f'Ft_{name}' for name in FEATURE_NAMES]

        with self.path.open('a', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True

            row = {
                'subject_id': subject_id,
                'image_name': Path(result.image_path).name,
                'image_path': result.image_path,
                # 실험자가 선택한 값은 실험 후 입력.
                'human_emoji': '',
                'human_class': '',
                'predicted_emoji': result.predicted_emoji,
                'predicted_class': result.predicted_class,
                'D1': f'{result.distance:.6f}',
                'second_emoji': result.second_emoji,
                'second_class': result.second_class,
                'D2': f'{result.second_distance:.6f}',
                'margin': f'{result.margin:.6f}',
                'calibration_mode': calibration_mode,
            }
            for name, value in zip(FEATURE_NAMES, result.features, strict=True):
                row[f'Ft_{name}'] = f'{float(value):.6f}'

            writer.writerow(row)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_annotated(output_dir: Path, source: Path, image: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{source.stem}_result.jpg'
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f'Cannot save result image: {output_path}')
    return output_path


def print_result(result: ValidationResult) -> None:
    print(f'Image      : {result.image_path}')
    print(f'Prediction : class {result.predicted_class:02d} {result.predicted_emoji}')
    print(f'D1         : {result.distance:.6f}')
    print(f'2nd        : class {result.second_class:02d} {result.second_emoji}')
    print(f'D2         : {result.second_distance:.6f}')
    print(f'Margin     : {result.margin:.6f}')
    print('Ft         : [' + ', '.join(f'{v:.3f}' for v in result.features) + ']')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Static-image validation program for the emoji prototype classifier.'
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path, help='Single facial-expression image')
    source.add_argument('--folder', type=Path, help='Folder containing expression images')

    parser.add_argument(
        '--neutral', type=Path, default=None,
        help='Optional neutral-face image for the same subject'
    )
    parser.add_argument('--subject', default='', help='Subject/participant ID saved to CSV')
    parser.add_argument(
        '--csv', type=Path,
        default=PROJECT_DIR / 'validation_results.csv',
        help='CSV output path'
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=PROJECT_DIR / 'validation_output',
        help='Annotated image output directory'
    )
    parser.add_argument('--show', action='store_true', help='Show annotated result window')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    validator = ImageEmojiValidator(config)
    csv_writer = ValidationCsvWriter(args.csv)

    calibration_mode = 'neutral_image' if args.neutral is not None else 'raw'

    try:
        if args.neutral is not None:
            validator.set_neutral_image(args.neutral)
            print(f'[neutral] {args.neutral}')
        else:
            print('[warning] neutral image not supplied: raw reconstructed Ft is used.')

        images = [args.image] if args.image is not None else list_images(args.folder)
        if not images:
            raise RuntimeError('No image files found.')

        for image_path in images:
            try:
                result, annotated = validator.classify(image_path)
                output_path = save_annotated(args.output_dir, image_path, annotated)
                csv_writer.append(result, args.subject, calibration_mode)

                print_result(result)
                print(f'Saved      : {output_path}')
                print('-' * 60)

                if args.show:
                    cv2.imshow('Validation Result', annotated)
                    key = cv2.waitKey(0) & 0xFF
                    if key in (27, ord('q')):
                        break
            except RuntimeError as error:
                print(f'[skip] {error}')

    finally:
        validator.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
