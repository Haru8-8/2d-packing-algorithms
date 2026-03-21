"""
bottom_left.py
2次元矩形パッキング問題に対する Bottom-Left (BL) 法。

参考文献:
    今堀慎治 他, "Python による図形詰込みアルゴリズム入門",
    オペレーションズ・リサーチ, 63(12), pp.762-769, 2018.

実装レベル: 単純実装 O(n^4)
    - bl_candidates : BL実行可能点の候補を列挙  O(n^2)
    - is_feasible   : 候補点に重なりなく置けるか判定  O(n)
    - bl_method     : BL法の本体  O(n^4) 全体

ノート:
    NFP を用いた高速版 (O(n^2 log n)) は nfp_bottom_left.py で実装予定。
    単純版と高速版を同じインターフェースで比較できるよう設計している。
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------
Rect = tuple[float, float]        # (width, height)
Point = tuple[float, float]       # (x, y)


# ---------------------------------------------------------------------------
# 候補点の列挙
# ---------------------------------------------------------------------------

def bl_candidates(
    i: int,
    x: list[float],
    y: list[float],
    w: list[float],
    h: list[float],
) -> list[Point]:
    """
    長方形 i を置く BL実行可能点の「候補」を列挙する。

    BL実行可能点とは、長方形 i を下にも左にも動かせない点のこと。
    そのような点は必ず以下のいずれかで決まる:
        (A) 容器の左下 (0, 0)
        (B) 既配置の長方形 j の右端 x[j]+w[j] × 長方形 k の上端 y[k]+h[k]
        (C) 容器左端 0 × 長方形 j の上端 y[j]+h[j]
        (D) 長方形 j の右端 x[j]+w[j] × 容器底辺 0

    Parameters
    ----------
    i   : これから配置する長方形のインデックス
    x,y : 既配置の長方形の左下座標リスト (長さ i)
    w,h : 全長方形の幅・高さリスト

    Returns
    -------
    候補点のリスト (重複あり、実行可能かは未チェック)
    """
    cand: list[Point] = [(0.0, 0.0)]   # (A)

    for j in range(i):
        for k in range(j):
            # (B) j の右端 × k の上端、および k の右端 × j の上端
            cand.append((x[j] + w[j], y[k] + h[k]))
            cand.append((x[k] + w[k], y[j] + h[j]))

        # (C) 容器左端 × j の上端
        cand.append((0.0, y[j] + h[j]))
        # (D) j の右端 × 容器底辺
        cand.append((x[j] + w[j], 0.0))

    return cand


# ---------------------------------------------------------------------------
# 実行可能性の判定
# ---------------------------------------------------------------------------

def is_feasible(
    i: int,
    p: Point,
    x: list[float],
    y: list[float],
    w: list[float],
    h: list[float],
    bin_w: float,
) -> bool:
    """
    長方形 i を座標 p=(px, py) に置けるかどうかを判定する。

    条件:
        1. ビンの左右にはみ出さない (y 方向の上限はストリップパッキングなので無制限)
        2. 既配置の長方形 0..i-1 のいずれとも重ならない

    Parameters
    ----------
    i      : これから配置する長方形のインデックス
    p      : 配置候補の左下座標
    x,y    : 既配置の長方形の左下座標リスト
    w,h    : 全長方形の幅・高さリスト
    bin_w  : ビンの幅

    Returns
    -------
    True なら配置可能
    """
    px, py = p

    # 座標が負、またはビン右端からはみ出す場合は不可
    if px < 0 or px + w[i] > bin_w:
        return False
    if py < 0:
        return False

    # 既配置の長方形と重なりがないか確認
    for j in range(i):
        # x 方向に重なりがあるか
        x_overlap = max(px, x[j]) < min(px + w[i], x[j] + w[j])
        # y 方向に重なりがあるか
        y_overlap = max(py, y[j]) < min(py + h[i], y[j] + h[j])
        if x_overlap and y_overlap:
            return False

    return True


# ---------------------------------------------------------------------------
# BL法の本体
# ---------------------------------------------------------------------------

def bl_method(
    rects: list[Rect],
    bin_w: float,
    sort_key: str = "area",
) -> tuple[list[Point], list[Rect]]:
    """
    Bottom-Left 法で矩形をストリップ（幅固定・高さ可変）に詰める。

    アルゴリズム:
        長方形を sort_key の順に並べ、各長方形を BL点（最も下で最も左）に配置する。

    Parameters
    ----------
    rects    : [(w0,h0), ...] 配置する矩形のリスト
    bin_w    : ビンの幅
    sort_key : 並べ替えの基準
                "area"   : 面積の降順（デフォルト）
                "width"  : 幅の降順
                "height" : 高さの降順
                "none"   : 並べ替えなし（入力順）

    Returns
    -------
    positions   : [(x0,y0), ...] 各矩形の配置座標（入力順）
    sorted_rects: 実際に配置した順の矩形リスト（可視化用）

    Raises
    ------
    ValueError : ビンに入らない矩形が存在する場合
    """
    # --- 並べ替え ---
    if sort_key == "area":
        order = sorted(range(len(rects)), key=lambda k: rects[k][0] * rects[k][1], reverse=True)
    elif sort_key == "width":
        order = sorted(range(len(rects)), key=lambda k: rects[k][0], reverse=True)
    elif sort_key == "height":
        order = sorted(range(len(rects)), key=lambda k: rects[k][1], reverse=True)
    elif sort_key == "none":
        order = list(range(len(rects)))
    else:
        raise ValueError(f"未知の sort_key: {sort_key!r}")

    sorted_rects = [rects[k] for k in order]

    # ビン幅を超える矩形がないか事前チェック
    for wi, hi in sorted_rects:
        if wi > bin_w:
            raise ValueError(
                f"矩形 (w={wi}, h={hi}) がビン幅 {bin_w} を超えています。"
            )

    w = [r[0] for r in sorted_rects]
    h = [r[1] for r in sorted_rects]
    x: list[float] = []
    y: list[float] = []

    for i in range(len(sorted_rects)):
        # 候補点を列挙
        cands = bl_candidates(i, x, y, w, h)

        # 実行可能な候補点を絞り込む
        feasible = [
            p for p in cands
            if is_feasible(i, p, x, y, w, h, bin_w)
        ]

        if not feasible:
            raise RuntimeError(
                f"長方形 {i} ({w[i]}×{h[i]}) を配置できる場所が見つかりませんでした。"
            )

        # BL点: y が最小、同じなら x が最小
        bl_point = min(feasible, key=lambda p: (p[1], p[0]))
        x.append(bl_point[0])
        y.append(bl_point[1])

    positions_sorted = list(zip(x, y))

    # 入力順の positions に戻す（可視化時に元のインデックスと対応させるため）
    positions = [None] * len(rects)
    for sorted_idx, orig_idx in enumerate(order):
        positions[orig_idx] = positions_sorted[sorted_idx]

    return positions, sorted_rects