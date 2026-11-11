from __future__ import annotations

import numpy as np
from constants import EMOJIS, FEATURE_INDEX, NUM_FEATURES

# 초기 실험용 가설값.
# neutral calibration 이후 재구성 feature 공간의 목표값 [0, 1].
# 논문에서 직접 제시된 threshold가 아님.
PROTOTYPE_SPECS: dict[str, dict[str, float]] = {
    '😀': {'smile': .75, 'cheek_squint': .20, 'jaw_open': .55},
    '🤣': {'smile': .85, 'eye_squint': .75, 'blink': .20, 'cheek_squint': .65, 'jaw_open': .65},
    '🤭': {'smile': .45, 'eye_squint': .20, 'cheek_squint': .35, 'jaw_open': .05},
    '😆': {'smile': .80, 'eye_squint': .75, 'blink': .25, 'cheek_squint': .60, 'jaw_open': .35},
    '🙂': {'smile': .35, 'jaw_open': .05},
    '😊': {'smile': .60, 'eye_squint': .30, 'cheek_squint': .55, 'jaw_open': .05},

    '😉': {'smile': .40, 'blink': .45, 'blink_asym': .80, 'cheek_squint': .20},
    '😏': {'smile': .25, 'smile_asym': .45, 'mouth_dimple': .35, 'dimple_asym': .40},
    '😋': {'smile': .40, 'cheek_squint': .15, 'mouth_pucker': .20, 'jaw_open': .25},
    '😗': {'mouth_pucker': .75, 'mouth_funnel': .45, 'jaw_open': .05},
    '😛': {'smile': .10, 'mouth_stretch': .15, 'jaw_open': .40},
    '😜': {'smile': .15, 'blink': .45, 'blink_asym': .80, 'jaw_open': .40},

    '🤪': {'smile': .15, 'eye_wide': .45, 'eye_wide_asym': .45, 'brow_asym': .25, 'jaw_open': .40},
    '🤨': {'brow_down': .15, 'brow_outer_up': .35, 'brow_asym': .60},
    '😐': {},
    '😑': {'eye_squint': .35, 'blink': .65, 'mouth_press': .15},
    '😒': {'frown': .25, 'eye_squint': .20, 'mouth_dimple': .25, 'gaze_side': .60},
    '🙄': {'eye_wide': .15, 'mouth_press': .10, 'gaze_up': .80},

    '😬': {'eye_wide': .15, 'mouth_press': .45, 'mouth_stretch': .70, 'jaw_open': .15},
    '😞': {'frown': .45, 'eye_squint': .15, 'brow_down': .25, 'brow_inner_up': .35},
    '😔': {'frown': .40, 'eye_squint': .25, 'brow_down': .20, 'brow_inner_up': .50},
    '🙁': {'frown': .55, 'brow_inner_up': .10},
    '🫤': {'frown': .25, 'frown_asym': .50, 'mouth_shift': .50},
    '🤢': {'frown': .20, 'eye_squint': .25, 'nose_sneer': .70, 'mouth_upper_up': .50},

    '😢': {'frown': .50, 'brow_down': .20, 'brow_inner_up': .45},
    '😭': {'frown': .60, 'eye_squint': .65, 'blink': .45, 'brow_down': .35, 'brow_inner_up': .50, 'jaw_open': .55},
    '😠': {'frown': .30, 'eye_squint': .45, 'brow_down': .75, 'mouth_press': .55},
    '😨': {'eye_wide': .65, 'brow_down': .20, 'brow_inner_up': .55, 'brow_outer_up': .25, 'mouth_stretch': .55, 'jaw_open': .35},
    '😱': {'eye_wide': .85, 'brow_inner_up': .65, 'brow_outer_up': .65, 'mouth_stretch': .25, 'jaw_open': .85},
    '😮': {'eye_wide': .45, 'brow_inner_up': .35, 'brow_outer_up': .40, 'mouth_funnel': .25, 'jaw_open': .60},
}


def _build_matrix() -> np.ndarray:
    matrix = np.zeros((len(EMOJIS), NUM_FEATURES), dtype=np.float32)
    for row, emoji in enumerate(EMOJIS):
        for feature_name, value in PROTOTYPE_SPECS[emoji].items():
            matrix[row, FEATURE_INDEX[feature_name]] = np.float32(value)
    return matrix


# P[e, i] = e번째 이모지의 i번째 재구성 feature prototype
P: np.ndarray = _build_matrix()
