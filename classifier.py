from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from constants import EMOJIS


@dataclass(slots=True, frozen=True)
class Prediction:
    emoji: str
    index: int
    distance: float
    second_emoji: str
    second_distance: float
    margin: float


class PrototypeClassifier:
    __slots__ = ('_P', '_diff', '_dist_sq', '_dist')

    def __init__(self, prototypes: np.ndarray) -> None:
        self._P = np.ascontiguousarray(prototypes, dtype=np.float32)
        self._diff = np.empty_like(self._P)
        self._dist_sq = np.empty(self._P.shape[0], dtype=np.float32)
        self._dist = np.empty(self._P.shape[0], dtype=np.float32)

    def predict(self, x: np.ndarray) -> Prediction:
        # D_e = sqrt(mean_i((F_i - P_e,i)^2))
        # preallocated arrays: 매 프레임 30 x M 임시 배열 생성 방지
        np.subtract(self._P, x, out=self._diff)
        np.square(self._diff, out=self._diff)
        np.mean(self._diff, axis=1, out=self._dist_sq)
        np.sqrt(self._dist_sq, out=self._dist)

        top2 = np.argpartition(self._dist, 1)[:2]
        if self._dist[top2[0]] <= self._dist[top2[1]]:
            best, second = int(top2[0]), int(top2[1])
        else:
            best, second = int(top2[1]), int(top2[0])

        d1 = float(self._dist[best])
        d2 = float(self._dist[second])
        return Prediction(
            emoji=EMOJIS[best],
            index=best,
            distance=d1,
            second_emoji=EMOJIS[second],
            second_distance=d2,
            margin=d2 - d1,
        )
