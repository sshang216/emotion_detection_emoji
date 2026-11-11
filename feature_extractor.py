from __future__ import annotations

from typing import Sequence
import numpy as np

from constants import NUM_FEATURES, FEATURE_INDEX


class BlendshapeFeatureExtractor:
    __slots__ = ('_index', '_bound', '_out')

    REQUIRED = (
        'mouthSmileLeft', 'mouthSmileRight',
        'mouthFrownLeft', 'mouthFrownRight',
        'eyeWideLeft', 'eyeWideRight',
        'eyeSquintLeft', 'eyeSquintRight',
        'eyeBlinkLeft', 'eyeBlinkRight',
        'browDownLeft', 'browDownRight',
        'browInnerUp',
        'browOuterUpLeft', 'browOuterUpRight',
        'cheekSquintLeft', 'cheekSquintRight',
        'mouthPressLeft', 'mouthPressRight',
        'mouthDimpleLeft', 'mouthDimpleRight',
        'mouthStretchLeft', 'mouthStretchRight',
        'noseSneerLeft', 'noseSneerRight',
        'mouthUpperUpLeft', 'mouthUpperUpRight',
        'mouthPucker', 'mouthFunnel',
        'jawOpen', 'mouthClose',
        'mouthLeft', 'mouthRight',
        'eyeLookUpLeft', 'eyeLookUpRight',
        'eyeLookOutLeft', 'eyeLookInRight',
        'eyeLookInLeft', 'eyeLookOutRight',
        'cheekPuff',
    )

    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self._bound = False
        self._out = np.empty(NUM_FEATURES, dtype=np.float32)

    def _bind(self, categories: Sequence) -> None:
        by_name = {category.category_name: i for i, category in enumerate(categories)}
        missing = [name for name in self.REQUIRED if name not in by_name]
        if missing:
            raise RuntimeError(f'Missing MediaPipe blendshapes: {missing}')
        self._index = by_name
        self._bound = True

    def transform(self, categories: Sequence) -> np.ndarray:
        if not self._bound:
            self._bind(categories)

        idx = self._index
        out = self._out

        def v(name: str) -> float:
            return float(categories[idx[name]].score)

        smile_l, smile_r = v('mouthSmileLeft'), v('mouthSmileRight')
        frown_l, frown_r = v('mouthFrownLeft'), v('mouthFrownRight')
        wide_l, wide_r = v('eyeWideLeft'), v('eyeWideRight')
        squint_l, squint_r = v('eyeSquintLeft'), v('eyeSquintRight')
        blink_l, blink_r = v('eyeBlinkLeft'), v('eyeBlinkRight')
        brow_down_l, brow_down_r = v('browDownLeft'), v('browDownRight')
        brow_outer_l, brow_outer_r = v('browOuterUpLeft'), v('browOuterUpRight')
        cheek_squint_l, cheek_squint_r = v('cheekSquintLeft'), v('cheekSquintRight')
        press_l, press_r = v('mouthPressLeft'), v('mouthPressRight')
        dimple_l, dimple_r = v('mouthDimpleLeft'), v('mouthDimpleRight')
        stretch_l, stretch_r = v('mouthStretchLeft'), v('mouthStretchRight')
        sneer_l, sneer_r = v('noseSneerLeft'), v('noseSneerRight')
        upper_l, upper_r = v('mouthUpperUpLeft'), v('mouthUpperUpRight')

        out[FEATURE_INDEX['smile']] = (smile_l + smile_r) * 0.5
        out[FEATURE_INDEX['smile_asym']] = abs(smile_l - smile_r)
        out[FEATURE_INDEX['frown']] = (frown_l + frown_r) * 0.5
        out[FEATURE_INDEX['frown_asym']] = abs(frown_l - frown_r)
        out[FEATURE_INDEX['eye_wide']] = (wide_l + wide_r) * 0.5
        out[FEATURE_INDEX['eye_wide_asym']] = abs(wide_l - wide_r)
        out[FEATURE_INDEX['eye_squint']] = (squint_l + squint_r) * 0.5
        out[FEATURE_INDEX['blink']] = (blink_l + blink_r) * 0.5
        out[FEATURE_INDEX['blink_asym']] = abs(blink_l - blink_r)
        out[FEATURE_INDEX['brow_down']] = (brow_down_l + brow_down_r) * 0.5
        out[FEATURE_INDEX['brow_inner_up']] = v('browInnerUp')
        out[FEATURE_INDEX['brow_outer_up']] = (brow_outer_l + brow_outer_r) * 0.5
        out[FEATURE_INDEX['brow_asym']] = abs(brow_outer_l - brow_outer_r)
        out[FEATURE_INDEX['cheek_squint']] = (cheek_squint_l + cheek_squint_r) * 0.5
        out[FEATURE_INDEX['mouth_press']] = (press_l + press_r) * 0.5
        out[FEATURE_INDEX['mouth_dimple']] = (dimple_l + dimple_r) * 0.5
        out[FEATURE_INDEX['dimple_asym']] = abs(dimple_l - dimple_r)
        out[FEATURE_INDEX['mouth_stretch']] = (stretch_l + stretch_r) * 0.5
        out[FEATURE_INDEX['nose_sneer']] = (sneer_l + sneer_r) * 0.5
        out[FEATURE_INDEX['mouth_upper_up']] = (upper_l + upper_r) * 0.5
        out[FEATURE_INDEX['mouth_pucker']] = v('mouthPucker')
        out[FEATURE_INDEX['mouth_funnel']] = v('mouthFunnel')
        out[FEATURE_INDEX['jaw_open']] = v('jawOpen')
        out[FEATURE_INDEX['mouth_close']] = v('mouthClose')
        out[FEATURE_INDEX['mouth_shift']] = max(v('mouthLeft'), v('mouthRight'))
        out[FEATURE_INDEX['gaze_up']] = (v('eyeLookUpLeft') + v('eyeLookUpRight')) * 0.5

        gaze_left = (v('eyeLookOutLeft') + v('eyeLookInRight')) * 0.5
        gaze_right = (v('eyeLookInLeft') + v('eyeLookOutRight')) * 0.5
        out[FEATURE_INDEX['gaze_side']] = max(gaze_left, gaze_right)
        out[FEATURE_INDEX['cheek_puff']] = v('cheekPuff')
        return out


class NeutralCalibrator:
    __slots__ = ('_buffer', '_count', '_baseline')

    def __init__(self, frames: int, num_features: int) -> None:
        self._buffer = np.empty((frames, num_features), dtype=np.float32)
        self._count = 0
        self._baseline = np.zeros(num_features, dtype=np.float32)

    @property
    def ready(self) -> bool:
        return self._count >= self._buffer.shape[0]

    @property
    def progress(self) -> float:
        return min(1.0, self._count / self._buffer.shape[0])

    def add(self, x: np.ndarray) -> None:
        if self.ready:
            return
        self._buffer[self._count] = x
        self._count += 1
        if self.ready:
            self._baseline[:] = np.median(self._buffer, axis=0)

    def apply(self, x: np.ndarray, out: np.ndarray) -> np.ndarray:
        np.subtract(x, self._baseline, out=out)
        np.clip(out, 0.0, 1.0, out=out)
        return out


class EMASmoother:
    __slots__ = ('alpha', '_state', '_initialized')

    def __init__(self, alpha: float, num_features: int) -> None:
        self.alpha = np.float32(alpha)
        self._state = np.zeros(num_features, dtype=np.float32)
        self._initialized = False

    def update(self, x: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self._state[:] = x
            self._initialized = True
            return self._state

        self._state *= np.float32(1.0) - self.alpha
        self._state += self.alpha * x
        return self._state
