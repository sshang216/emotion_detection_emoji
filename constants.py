from __future__ import annotations

FEATURE_NAMES = [
    'smile', 'smile_asym',
    'frown', 'frown_asym',
    'eye_wide', 'eye_wide_asym',
    'eye_squint', 'blink', 'blink_asym',
    'brow_down', 'brow_inner_up', 'brow_outer_up', 'brow_asym',
    'cheek_squint',
    'mouth_press', 'mouth_dimple', 'dimple_asym',
    'mouth_stretch',
    'nose_sneer', 'mouth_upper_up',
    'mouth_pucker', 'mouth_funnel',
    'jaw_open', 'mouth_close', 'mouth_shift',
    'gaze_up', 'gaze_side',
    'cheek_puff',
]

EMOJIS = [
    '😀','🤣','🤭','😆','🙂','😊',
    '😉','😏','😋','😗','😛','😜',
    '🤪','🤨','😐','😑','😒','🙄',
    '😬','😞','😔','🙁','🫤','🤢',
    '😢','😭','😠','😨','😱','😮',
]

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
NUM_FEATURES = len(FEATURE_NAMES)
NUM_EMOJIS = len(EMOJIS)
