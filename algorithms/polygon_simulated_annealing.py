"""
polygon_simulated_annealing.py
焼きなまし法 (Simulated Annealing) による多角形パッキングの
配置順序 × 回転角の組み合わせ最適化。

アイデア:
    多角形BL法の解品質は配置順序と各図形の回転角に依存する。
    焼きなまし法で「良い順序 × 回転角の組み合わせ」を探索することで
    充填率を改善する。

状態   : 配置順序（インデックスの順列）+ 各図形の回転角
近傍   : (1) swap_two   : ランダムに2要素の順序をスワップ
         (2) insert_one : ランダムに1要素を別の位置に挿入
         (3) rotate_one : ランダムに1図形の回転角を変更
目的関数: 使用した高さ（最小化） → 充填率の最大化に対応

参考:
    今堀・山下, "メタヒューリスティクスの数理", 共立出版, 2010.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from algorithms.nfp_polygon import (
    Vertices, Angle,
    make_polygon, rotate_polygon,
    build_nfp_cache, build_ifr_cache,
)
from algorithms.polygon_bl import find_bl_point_polygon

# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------
PlacedItem = tuple[int, Polygon, tuple[float, float], Angle]
# (元インデックス, 図形, 参照点座標, 回転角)


# ---------------------------------------------------------------------------
# 評価関数
# ---------------------------------------------------------------------------

def evaluate(
    order:     list[int],
    thetas:    list[Angle],
    polygons:  list[Polygon],
    nfp_cache: dict,
    ifr_cache: dict,
) -> float:
    """
    与えられた配置順序・回転角で BL法を実行し、使用した高さを返す。

    Parameters
    ----------
    order     : 配置順序（インデックスのリスト）
    thetas    : 各図形の回転角（インデックス順）
    polygons  : 全図形のリスト（make_polygon済み・原点正規化済み）
    nfp_cache : 事前計算済みの NFP キャッシュ
    ifr_cache : 事前計算済みの IFR キャッシュ

    Returns
    -------
    使用した高さ（小さいほど良い）
    """
    placed_items: list[PlacedItem] = []
    max_y = 0.0

    for orig_idx in order:
        theta   = thetas[orig_idx]
        rotated = rotate_polygon(polygons[orig_idx], theta)
        ifr     = ifr_cache.get((orig_idx, theta))

        pos = find_bl_point_polygon(
            poly_moving   = rotated,
            placed_items  = placed_items,
            nfp_cache     = nfp_cache,
            ifr           = ifr,
            moving_idx    = orig_idx,
            moving_theta  = theta,
        )

        if pos is None:
            return float('inf')

        placed_items.append((orig_idx, rotated, pos, theta))
        # 図形の実際の上端 = 参照点y + 図形の高さ
        # rotate_polygon で原点正規化済みのため bounds[1]=0、bounds[3]=高さ
        max_y = max(max_y, pos[1] + (rotated.bounds[3] - rotated.bounds[1]))

    return max_y


# ---------------------------------------------------------------------------
# 近傍操作
# ---------------------------------------------------------------------------

def swap_two(
    order:        list[int],
    thetas:       list[Angle],
    orientations: list[Angle],
) -> tuple[list[int], list[Angle]]:
    """
    ランダムに選んだ2要素の順序をスワップした新しい状態を返す。

    Returns
    -------
    (新しい順列, 回転角リスト（変更なし）)
    """
    n = len(order)
    i, j = random.sample(range(n), 2)
    new_order = order[:]
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order, thetas[:]


def insert_one(
    order:        list[int],
    thetas:       list[Angle],
    orientations: list[Angle],
) -> tuple[list[int], list[Angle]]:
    """
    ランダムに選んだ1要素を別の位置に挿入した新しい状態を返す。

    swap より大きな変化を生むため、探索の多様性を高める。

    Returns
    -------
    (新しい順列, 回転角リスト（変更なし）)
    """
    n = len(order)
    i = random.randrange(n)
    j = random.randrange(n - 1)
    if j >= i:
        j += 1
    new_order = order[:]
    elem = new_order.pop(i)
    new_order.insert(j, elem)
    return new_order, thetas[:]


def rotate_one(
    order:        list[int],
    thetas:       list[Angle],
    orientations: list[Angle],
) -> tuple[list[int], list[Angle]]:
    """
    ランダムに選んだ1図形の回転角を別の角度に変更した新しい状態を返す。

    Returns
    -------
    (順列（変更なし）, 新しい回転角リスト)
    """
    idx      = random.randrange(len(order))
    orig_idx = order[idx]
    current  = thetas[orig_idx]
    choices  = [a for a in orientations if a != current]
    if not choices:
        return order[:], thetas[:]
    new_thetas            = thetas[:]
    new_thetas[orig_idx]  = random.choice(choices)
    return order[:], new_thetas


# ---------------------------------------------------------------------------
# 結果の格納
# ---------------------------------------------------------------------------

@dataclass
class SAResult:
    """焼きなまし法の実行結果。"""
    best_order:     list[int]                  # 最良の配置順序
    best_thetas:    list[Angle]                # 最良の回転角（インデックス順）
    best_height:    float                      # 最良の使用高さ
    best_positions: list[tuple[float, float]]  # 最良配置の座標（入力順）
    best_polys:     list[Polygon]              # 最良配置の図形（入力順）
    initial_height: float                      # 初期解の高さ
    elapsed:        float                      # 実行時間 (s)
    history:        list[tuple[int, float, float]] = field(default_factory=list)
    # history の各要素: (反復数, 現在の高さ, 最良の高さ)


# ---------------------------------------------------------------------------
# 焼きなまし法の本体
# ---------------------------------------------------------------------------

def simulated_annealing_polygon(
    vertices_list: list[Vertices],
    bin_w:         float,
    orientations:  list[Angle] | None = None,
    *,
    t_start:      float = 5.0,
    t_end:        float = 0.01,
    cooling:      float = 0.995,
    max_iter:     int   = 10_000,
    neighbor:     str   = "mixed",
    init_order:   list[int]   | None = None,
    init_thetas:  list[Angle] | None = None,
    seed:         int | None = None,
    log_interval: int = 500,
) -> SAResult:
    """
    焼きなまし法で多角形の配置順序 × 回転角を最適化する。

    Parameters
    ----------
    vertices_list : 配置する多角形の頂点リストのリスト
    bin_w         : ビンの幅
    orientations  : 使用する回転角のリスト（デフォルト: [0, 90, 180, 270]）
    t_start       : 初期温度
    t_end         : 終了温度
    cooling       : 冷却率（1反復ごとに T *= cooling）
    max_iter      : 最大反復数
    neighbor      : 近傍操作の種類
                    "swap"   : 順序の2要素スワップのみ
                    "insert" : 順序の1要素挿入のみ
                    "rotate" : 回転角変更のみ
                    "mixed"  : swap・insert・rotate をランダムに選択
    init_order    : 初期順序（None なら面積降順）
    init_thetas   : 初期回転角（None なら全図形0度）
    seed          : 乱数シード（再現性のため）
    log_interval  : 何反復ごとに history を記録するか

    Returns
    -------
    SAResult オブジェクト
    """
    if seed is not None:
        random.seed(seed)

    if orientations is None:
        orientations = [0, 90, 180, 270]

    # --- 図形の前処理 ---
    polygons = [make_polygon(v) for v in vertices_list]
    n = len(polygons)

    # --- NFP・IFR キャッシュを事前計算（SA全体で1度だけ）---
    nfp_cache = build_nfp_cache(polygons, orientations)
    ifr_cache = build_ifr_cache(polygons, orientations, bin_w)

    # --- 初期解の生成 ---
    if init_order is not None:
        current_order = init_order[:]
    else:
        # 面積降順（BL法で最も一般的な出発点）
        current_order = sorted(range(n), key=lambda k: polygons[k].area, reverse=True)

    if init_thetas is not None:
        current_thetas = init_thetas[:]
    else:
        current_thetas = [0] * n

    # --- 近傍関数の選択 ---
    neighbor_fns = {
        "swap":   [swap_two],
        "insert": [insert_one],
        "rotate": [rotate_one],
        "mixed":  [swap_two, insert_one, rotate_one],
    }
    fns = neighbor_fns.get(neighbor, [swap_two, insert_one, rotate_one])

    # --- 初期評価 ---
    current_height = evaluate(current_order, current_thetas, polygons, nfp_cache, ifr_cache)
    best_order     = current_order[:]
    best_thetas    = current_thetas[:]
    best_height    = current_height
    initial_height = current_height

    history: list[tuple[int, float, float]] = [(0, current_height, best_height)]

    t = t_start
    t_start_time = time.perf_counter()

    for it in range(1, max_iter + 1):
        # --- 近傍解の生成 ---
        fn = random.choice(fns)
        new_order, new_thetas = fn(current_order, current_thetas, orientations)
        new_height = evaluate(new_order, new_thetas, polygons, nfp_cache, ifr_cache)

        # --- 受理判定 ---
        delta = new_height - current_height
        if delta < 0 or random.random() < math.exp(-delta / t):
            current_order  = new_order
            current_thetas = new_thetas
            current_height = new_height

            # 最良解の更新
            if current_height < best_height:
                best_height = current_height
                best_order  = current_order[:]
                best_thetas = current_thetas[:]

        # --- 温度の更新 ---
        t = max(t * cooling, t_end)

        # --- ログ ---
        if it % log_interval == 0:
            history.append((it, current_height, best_height))

    elapsed = time.perf_counter() - t_start_time

    # --- 最良解の配置情報を復元 ---
    placed_items: list[PlacedItem] = []
    positions_sorted:  list[tuple[float, float]] = []
    polys_sorted:      list[Polygon] = []

    for orig_idx in best_order:
        theta   = best_thetas[orig_idx]
        rotated = rotate_polygon(polygons[orig_idx], theta)
        ifr     = ifr_cache.get((orig_idx, theta))

        pos = find_bl_point_polygon(
            poly_moving   = rotated,
            placed_items  = placed_items,
            nfp_cache     = nfp_cache,
            ifr           = ifr,
            moving_idx    = orig_idx,
            moving_theta  = theta,
        )
        placed_items.append((orig_idx, rotated, pos, theta))
        positions_sorted.append(pos)
        polys_sorted.append(rotated)

    # 入力順に戻す
    positions:    list[tuple[float, float] | None] = [None] * n
    placed_polys: list[Polygon | None]             = [None] * n

    for sorted_idx, orig_idx in enumerate(best_order):
        positions[orig_idx]    = positions_sorted[sorted_idx]
        placed_polys[orig_idx] = polys_sorted[sorted_idx]

    return SAResult(
        best_order     = best_order,
        best_thetas    = best_thetas,
        best_height    = best_height,
        best_positions = positions,
        best_polys     = placed_polys,
        initial_height = initial_height,
        elapsed        = elapsed,
        history        = history,
    )