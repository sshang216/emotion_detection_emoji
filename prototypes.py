from __future__ import annotations

import numpy as np

from constants import EMOJIS, FEATURE_INDEX, NUM_FEATURES


# Prototype v0.2
#
# summary.csv의 19명 × 8개 감정 이미지에서 얻은 neutral-corrected Ft 분포를
# 기준으로 전체 스케일을 재조정한 값.
#
# - neutral은 원점 {} 유지
# - anger / contempt / disgust / fear / happy / sad / surprised:
#   실제 데이터의 평균 및 분위수 범위를 기준으로 조정
# - 30개 이모지 중 데이터셋에 직접 대응 라벨이 없는 세부 이모지는
#   가장 가까운 기본 감정 중심에서 표정 강도, 좌우 비대칭, 눈/입 움직임을
#   변화시켜 세분화한 초기 실험용 Prototype
# - cheek_squint, nose_sneer는 현재 데이터에서 거의 0으로 관측되어 제외

PROTOTYPE_VERSION = "v0.2-data-scaled"

PROTOTYPE_SPECS: dict[str, dict[str, float]] = {
    '😀': {'smile': .80, 'mouth_upper_up': .32, 'eye_squint': .14, 'blink': .06, 'mouth_press': .04, 'mouth_stretch': .06, 'jaw_open': .06, 'smile_asym': .03},
    '🤣': {'smile': .90, 'mouth_upper_up': .45, 'eye_squint': .25, 'blink': .12, 'mouth_press': .04, 'mouth_stretch': .10, 'jaw_open': .10, 'smile_asym': .03},
    '🤭': {'smile': .55, 'mouth_upper_up': .18, 'eye_squint': .12, 'blink': .07, 'mouth_press': .08, 'jaw_open': .01, 'smile_asym': .03},
    '😆': {'smile': .82, 'mouth_upper_up': .38, 'eye_squint': .23, 'blink': .12, 'mouth_press': .05, 'mouth_stretch': .08, 'jaw_open': .04, 'smile_asym': .03},
    '🙂': {'smile': .40, 'mouth_upper_up': .12, 'eye_squint': .06, 'blink': .04, 'mouth_press': .03, 'jaw_open': .01, 'smile_asym': .02},
    '😊': {'smile': .70, 'mouth_upper_up': .30, 'eye_squint': .16, 'blink': .07, 'mouth_press': .05, 'mouth_stretch': .05, 'jaw_open': .02, 'smile_asym': .03},
    '😉': {'smile': .55, 'mouth_upper_up': .22, 'eye_squint': .12, 'blink': .13, 'blink_asym': .08, 'mouth_press': .04, 'jaw_open': .02, 'smile_asym': .03},

    '😏': {'brow_inner_up': .10, 'brow_outer_up': .08, 'brow_asym': .10, 'brow_down': .07, 'gaze_side': .10, 'eye_squint': .09, 'blink': .06, 'blink_asym': .03, 'mouth_shift': .02, 'smile': .03, 'smile_asym': .04},

    '😋': {'smile': .48, 'mouth_upper_up': .15, 'eye_squint': .10, 'blink': .05, 'mouth_pucker': .18, 'jaw_open': .05, 'mouth_stretch': .04, 'smile_asym': .03},
    '😗': {'mouth_pucker': .25, 'mouth_funnel': .04, 'jaw_open': .01, 'mouth_close': .02},
    '😛': {'smile': .32, 'mouth_upper_up': .12, 'eye_squint': .08, 'mouth_pucker': .08, 'mouth_stretch': .07, 'jaw_open': .18, 'smile_asym': .03},
    '😜': {'smile': .35, 'mouth_upper_up': .12, 'eye_squint': .10, 'blink': .13, 'blink_asym': .08, 'mouth_pucker': .08, 'mouth_stretch': .07, 'jaw_open': .16, 'smile_asym': .03},
    '🤪': {'smile': .25, 'eye_wide': .10, 'eye_wide_asym': .08, 'brow_inner_up': .20, 'brow_outer_up': .25, 'brow_asym': .15, 'mouth_pucker': .10, 'jaw_open': .18, 'smile_asym': .04},
    '🤨': {'brow_inner_up': .16, 'brow_outer_up': .18, 'brow_asym': .18, 'brow_down': .07, 'gaze_side': .08, 'eye_squint': .08, 'blink': .05, 'mouth_shift': .02},

    '😐': {},

    '😑': {'eye_squint': .10, 'blink': .14, 'mouth_press': .03, 'gaze_side': .02},
    '😒': {'brow_down': .10, 'brow_inner_up': .06, 'brow_asym': .06, 'gaze_side': .20, 'eye_squint': .12, 'blink': .06, 'mouth_shift': .02, 'mouth_press': .02},
    '🙄': {'brow_inner_up': .08, 'brow_outer_up': .10, 'gaze_up': .18, 'gaze_side': .08, 'eye_wide': .03, 'blink': .04},
    '😬': {'brow_inner_up': .16, 'brow_outer_up': .14, 'eye_wide': .06, 'eye_wide_asym': .03, 'mouth_press': .07, 'mouth_stretch': .10, 'jaw_open': .05, 'mouth_pucker': .04},

    '😞': {'brow_inner_up': .12, 'brow_outer_up': .04, 'brow_down': .06, 'eye_squint': .08, 'blink': .07, 'blink_asym': .02, 'gaze_side': .04, 'mouth_pucker': .04, 'mouth_stretch': .01},
    '😔': {'brow_inner_up': .18, 'brow_outer_up': .05, 'brow_down': .04, 'eye_squint': .10, 'blink': .10, 'blink_asym': .02, 'gaze_side': .04, 'mouth_pucker': .04},
    '🙁': {'brow_inner_up': .10, 'brow_down': .08, 'eye_squint': .06, 'blink': .04, 'frown': .03, 'mouth_pucker': .05, 'gaze_side': .03},
    '🫤': {'brow_inner_up': .10, 'brow_asym': .08, 'eye_squint': .06, 'blink': .05, 'gaze_side': .08, 'mouth_shift': .04, 'frown': .02, 'mouth_pucker': .04},

    '🤢': {'brow_down': .19, 'eye_squint': .14, 'blink': .08, 'blink_asym': .04, 'gaze_side': .23, 'gaze_up': .03, 'mouth_pucker': .12, 'mouth_upper_up': .06, 'smile': .03},

    '😢': {'brow_inner_up': .20, 'brow_outer_up': .06, 'brow_down': .06, 'eye_squint': .10, 'blink': .10, 'blink_asym': .03, 'gaze_side': .05, 'mouth_pucker': .05},
    '😭': {'brow_inner_up': .28, 'brow_outer_up': .08, 'brow_down': .10, 'eye_squint': .18, 'blink': .16, 'blink_asym': .04, 'gaze_side': .06, 'mouth_pucker': .08, 'jaw_open': .12},

    '😠': {'brow_down': .15, 'eye_squint': .09, 'blink': .02, 'brow_asym': .02, 'gaze_up': .07, 'gaze_side': .04, 'mouth_pucker': .11, 'mouth_press': .02, 'mouth_upper_up': .05, 'jaw_open': .02},

    '😨': {'brow_inner_up': .32, 'brow_outer_up': .32, 'brow_asym': .06, 'eye_wide': .08, 'eye_wide_asym': .05, 'gaze_side': .06, 'mouth_pucker': .09, 'mouth_funnel': .02, 'jaw_open': .15, 'mouth_stretch': .03},

    '😱': {'brow_inner_up': .55, 'brow_outer_up': .55, 'brow_asym': .08, 'eye_wide': .13, 'eye_wide_asym': .06, 'mouth_pucker': .20, 'mouth_funnel': .04, 'jaw_open': .40, 'mouth_stretch': .04},

    '😮': {'brow_inner_up': .41, 'brow_outer_up': .44, 'brow_asym': .09, 'eye_wide': .07, 'eye_wide_asym': .04, 'gaze_side': .10, 'mouth_pucker': .19, 'mouth_funnel': .03, 'jaw_open': .18, 'mouth_close': .02},
}


def _build_matrix() -> np.ndarray:
    matrix = np.zeros((len(EMOJIS), NUM_FEATURES), dtype=np.float32)
    for row, emoji in enumerate(EMOJIS):
        for feature_name, value in PROTOTYPE_SPECS[emoji].items():
            matrix[row, FEATURE_INDEX[feature_name]] = np.float32(value)
    return matrix


# P[e, i] = e번째 이모지의 i번째 재구성 feature prototype
P: np.ndarray = _build_matrix()
