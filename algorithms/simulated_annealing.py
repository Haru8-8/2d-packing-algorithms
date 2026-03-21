"""
simulated_annealing.py
焼きなまし法 (Simulated Annealing) による矩形パッキングの配置順序最適化。

アイデア:
    BL法の解品質は矩形の配置順序に依存する。
    焼きなまし法で「良い順序」を探索することで充填率を改善する。

状態   : 矩形の配置順序（インデックスの順列）
近傍   : ランダムに選んだ2要素をスワップ
目的関数: 使用した高さ（最小化） → 充填率の最大化に対応

参考:
    今堀・山下, "メタヒューリスティクスの数理", 共立出版, 2010.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from algorithms.nfp_bottom_left import bl_method_nfp

# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------
Rect  = tuple[float, float]   # (width, height)
Point = tuple[float, float]   # (x, y)


# ---------------------------------------------------------------------------
# 評価関数
# ---------------------------------------------------------------------------

def evaluate(
    order: list[int],
    rects: list[Rect],
    bin_w: float,
) -> float:
    """
    与えられた配置順序で BL法を実行し、使用した高さを返す。

    Parameters
    ----------
    order  : 配置順序（インデックスのリスト）
    rects  : 全矩形のリスト
    bin_w  : ビンの幅

    Returns
    -------
    使用した高さ（小さいほど良い）
    """
    sorted_rects = [rects[k] for k in order]
    placed: list[tuple[float, float, float, float]] = []

    from algorithms.nfp_bottom_left import find_bl_point
    positions: list[Point] = []

    for (wq, hq) in sorted_rects:
        bl = find_bl_point(wq, hq, placed, bin_w)
        if bl is None:
            return float('inf')
        x, y = bl
        placed.append((x, y, wq, hq))
        positions.append((x, y))

    return max(y + h for (_, h), (_, y) in zip(sorted_rects, positions))


# ---------------------------------------------------------------------------
# 近傍操作
# ---------------------------------------------------------------------------

def swap_two(order: list[int]) -> tuple[list[int], int, int]:
    """
    ランダムに選んだ2要素をスワップした新しい順列を返す。

    Returns
    -------
    (新しい順列, スワップしたインデックスi, j)
    """
    n = len(order)
    i, j = random.sample(range(n), 2)
    new_order = order[:]
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order, i, j


def insert_one(order: list[int]) -> tuple[list[int], int, int]:
    """
    ランダムに選んだ1要素を別の位置に挿入した新しい順列を返す。

    swap より大きな変化を生むため、探索の多様性を高める。

    Returns
    -------
    (新しい順列, 元の位置, 挿入先の位置)
    """
    n = len(order)
    i = random.randrange(n)
    j = random.randrange(n - 1)
    if j >= i:
        j += 1
    new_order = order[:]
    elem = new_order.pop(i)
    new_order.insert(j, elem)
    return new_order, i, j


# ---------------------------------------------------------------------------
# 結果の格納
# ---------------------------------------------------------------------------

@dataclass
class SAResult:
    """焼きなまし法の実行結果。"""
    best_order:     list[int]          # 最良の配置順序
    best_height:    float              # 最良の使用高さ
    best_positions: list[Point]        # 最良配置の座標（入力順）
    best_rects:     list[Rect]         # 最良配置の矩形（配置順）
    initial_height: float              # 初期解の高さ
    elapsed:        float              # 実行時間 (s)
    history:        list[tuple[int, float, float]] = field(default_factory=list)
    # history の各要素: (反復数, 現在の高さ, 最良の高さ)


# ---------------------------------------------------------------------------
# 焼きなまし法の本体
# ---------------------------------------------------------------------------

def simulated_annealing(
    rects: list[Rect],
    bin_w: float,
    *,
    t_start:    float = 5.0,
    t_end:      float = 0.01,
    cooling:    float = 0.995,
    max_iter:   int   = 10_000,
    neighbor:   str   = "swap",
    init_order: list[int] | None = None,
    seed:       int | None = None,
    log_interval: int = 500,
) -> SAResult:
    """
    焼きなまし法で矩形の配置順序を最適化する。

    Parameters
    ----------
    rects       : 配置する矩形のリスト
    bin_w       : ビンの幅
    t_start     : 初期温度
    t_end       : 終了温度
    cooling     : 冷却率（1反復ごとに T *= cooling）
    max_iter    : 最大反復数
    neighbor    : 近傍操作の種類 ("swap" or "insert")
    init_order  : 初期順序（None なら面積降順）
    seed        : 乱数シード（再現性のため）
    log_interval: 何反復ごとに history を記録するか

    Returns
    -------
    SAResult オブジェクト
    """
    if seed is not None:
        random.seed(seed)

    n = len(rects)

    # --- 初期解の生成 ---
    if init_order is not None:
        current_order = init_order[:]
    else:
        # 面積降順（BL法で最も一般的な出発点）
        current_order = sorted(range(n), key=lambda k: rects[k][0] * rects[k][1], reverse=True)

    neighbor_fn = swap_two if neighbor == "swap" else insert_one

    current_height = evaluate(current_order, rects, bin_w)
    best_order     = current_order[:]
    best_height    = current_height
    initial_height = current_height

    history: list[tuple[int, float, float]] = [(0, current_height, best_height)]

    t = t_start
    t_start_time = time.perf_counter()

    for it in range(1, max_iter + 1):
        # --- 近傍解の生成 ---
        new_order, _, _ = neighbor_fn(current_order)
        new_height = evaluate(new_order, rects, bin_w)

        # --- 受理判定 ---
        delta = new_height - current_height
        if delta < 0 or random.random() < math.exp(-delta / t):
            current_order  = new_order
            current_height = new_height

            # 最良解の更新
            if current_height < best_height:
                best_height = current_height
                best_order  = current_order[:]

        # --- 温度の更新 ---
        t = max(t * cooling, t_end)

        # --- ログ ---
        if it % log_interval == 0:
            history.append((it, current_height, best_height))

    elapsed = time.perf_counter() - t_start_time

    # --- 最良解の配置情報を復元 ---
    from algorithms.nfp_bottom_left import find_bl_point
    sorted_rects = [rects[k] for k in best_order]
    placed: list[tuple[float, float, float, float]] = []
    positions_sorted: list[Point] = []

    for (wq, hq) in sorted_rects:
        bl = find_bl_point(wq, hq, placed, bin_w)
        x, y = bl
        placed.append((x, y, wq, hq))
        positions_sorted.append((x, y))

    # 入力順に戻す
    positions: list[Point | None] = [None] * n
    for sorted_idx, orig_idx in enumerate(best_order):
        positions[orig_idx] = positions_sorted[sorted_idx]

    return SAResult(
        best_order     = best_order,
        best_height    = best_height,
        best_positions = positions,
        best_rects     = sorted_rects,
        initial_height = initial_height,
        elapsed        = elapsed,
        history        = history,
    )