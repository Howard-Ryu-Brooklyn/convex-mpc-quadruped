"""베지에 곡선 — 스윙 발 궤적 생성.

이 모듈이 따로 존재하는 이유
    같은 곡선을 두 곳에서 각자 구현하고 있었고, 안전성이 서로 달랐다.
        config.compute_bezier                      -> 위상 클립 없음
        swing_trajectory_generator.compute_bezier_with_kinematics -> 클립 있음
    위험한 쪽이 메인 루프에서 쓰이고 있었고, 그것이 결함 6 이다.

    여기서는 bezier_with_derivatives 가 위치 계산을 bezier 에 위임한다.
    두 함수가 다른 위치를 낼 수 있는 경로 자체를 없애기 위해서다.
    규율이 아니라 구조로 막는다.
"""
from __future__ import annotations

import math

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def bezier(s: float, control_points: Sequence[NDArray[np.float64]] | NDArray[np.float64]) -> NDArray[np.float64]:
    """N 차 베지에 곡선 위의 한 점.

    Args:
        s: 궤적 진행률 [0, 1]. 범위 밖은 클립된다.
        control_points: (N+1, 3) 제어점

    Returns:
        (3,) 진행률 s 에서의 위치

    위상 클립이 함수 내부에 있는 이유:
        s > 1 이면 번스타인 다항식이 외삽한다. 3 차 베지에는
        p3 + 3(s-1)(p3-p2) 로 발산하고, 스윙 궤적에서 p3-p2 는
        -clearance * z_hat 이므로 발이 지면 아래로 파고든다.
        호출부는 계속 늘어나고 그중 하나는 반드시 클립을 잊는다.
        방어는 함수 안쪽에 둔다.
    """
    s = np.clip(s, 0.0, 1.0)
    pts = np.array(control_points)
    n = len(pts) - 1

    p = np.zeros(3)
    for i in range(n + 1):
        coeff = math.comb(n, i) * ((1 - s) ** (n - i)) * (s ** i)
        p += coeff * pts[i]
    return p


def bezier_with_derivatives(
    s: float,
    control_points: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    duration_s: float = 0.3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """베지에 곡선의 위치와 시간 미분(속도, 가속도)을 해석적으로 계산한다.

    Args:
        s: 궤적 진행률 [0, 1]. 범위 밖은 클립된다.
        control_points: (N+1, 3) 제어점
        duration_s: 궤적 전체 소요 시간 [s]. s -> t 변환에 쓰인다.

    Returns:
        (p, v, a) — 위치 [m], 속도 [m/s], 가속도 [m/s^2]

    유도
        B'(s)  = n * sum_i b_{i,n-1}(s) * (P_{i+1} - P_i)
        B''(s) = n(n-1) * sum_i b_{i,n-2}(s) * (P_{i+2} - 2 P_{i+1} + P_i)
        연쇄법칙: v = B'(s)/T,  a = B''(s)/T^2

    위치는 bezier() 에 위임한다. 두 함수가 서로 다른 위치를 낼 수 있는
    경로를 구조적으로 없애기 위함이다 (결함 6 재발 방지).
    """
    s = np.clip(s, 0.0, 1.0)
    pts = np.array(control_points)
    n = len(pts) - 1

    p = bezier(s, pts)

    dp_ds = np.zeros(3)
    d2p_ds2 = np.zeros(3)

    if n >= 1:
        d1 = pts[1:] - pts[:-1]                      # 1 계 차분
        for i in range(n):
            coeff = math.comb(n - 1, i) * ((1 - s) ** (n - 1 - i)) * (s ** i)
            dp_ds += n * coeff * d1[i]

        if n >= 2:
            d2 = d1[1:] - d1[:-1]                    # 2 계 차분
            for i in range(n - 1):
                coeff = math.comb(n - 2, i) * ((1 - s) ** (n - 2 - i)) * (s ** i)
                d2p_ds2 += n * (n - 1) * coeff * d2[i]

    ds_dt = 1.0 / duration_s
    return p, dp_ds * ds_dt, d2p_ds2 * (ds_dt ** 2)
