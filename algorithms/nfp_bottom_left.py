"""
nfp_bottom_left.py
NFP (No-Fit Polygon) を用いた高速 Bottom-Left 法。

参考文献:
    今堀慎治 他, "Python による図形詰込みアルゴリズム入門",
    オペレーションズ・リサーチ, 63(12), pp.762-769, 2018.

実装レベル: NFP + 走査線 (Find2D-BL)  O(n^2 log n)
    - calc_ifr        : Inner-Fit Rectangle の計算  O(1)
    - calc_nfp        : 矩形同士の NFP の計算  O(1)
    - find_bl_point   : 走査線で BL点を探索  O(n log n)
    - bl_method_nfp   : BL法の本体  O(n^2 log n) 全体

bottom_left.py (単純実装 O(n^4)) と同じインターフェースを持つため、
そのまま差し替えてベンチマーク比較ができる。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------
Rect    = tuple[float, float]                # (width, height)
Point   = tuple[float, float]                # (x, y)
NFPRect = tuple[float, float, float, float]  # (x, y, w, h)


# ---------------------------------------------------------------------------
# ① IFR (Inner-Fit Rectangle) の計算
# ---------------------------------------------------------------------------

def calc_ifr(
    bin_w: float,
    bin_h_max: float,
    wq: float,
    hq: float,
) -> NFPRect:
    """
    ビン内で矩形 Q(wq×hq) の参照点(左下)が取れる範囲を返す。

    IFR とは「Q をビン内側に接するように動かしたときの
    Q の参照点の軌跡の内部領域」。
    矩形ビンの場合は単純な矩形になる。

    Returns
    -------
    (x, y, w, h) : IFR の左下座標とサイズ
    """
    return (0.0, 0.0, bin_w - wq, bin_h_max - hq)


# ---------------------------------------------------------------------------
# ② NFP の計算（矩形同士）
# ---------------------------------------------------------------------------

def calc_nfp(
    px: float, py: float, wp: float, hp: float,
    wq: float, hq: float,
) -> NFPRect:
    """
    固定矩形 P に対する移動矩形 Q の NFP(P, Q) を計算する。

    矩形同士の場合、NFP は必ず矩形になる。
        NFP の左下 = (px - wq, py - hq)
        NFP のサイズ = (wp + wq) × (hp + hq)

    Returns
    -------
    (x, y, w, h) : NFP の左下座標とサイズ
    """
    return (px - wq, py - hq, wp + wq, hp + hq)


# ---------------------------------------------------------------------------
# ③ 走査線による BL点探索 (Find2D-BL)
# ---------------------------------------------------------------------------

def find_bl_point(
    wq: float,
    hq: float,
    placed_rects: list[tuple[float, float, float, float]],
    bin_w: float,
    bin_h_max: float = 1e9,
) -> Point | None:
    """
    走査線アルゴリズムで BL点（最も下で最も左）を探す。

    アルゴリズムの概要:
        1. IFR を計算し、Q の参照点が取れる範囲を得る
        2. 各既配置矩形の NFP を計算
        3. y イベント（NFP の下端・上端）をソートして走査線を動かす
        4. 各 y において NFP に覆われていない最小 x を探す
        5. 最初に見つかった点（y 最小 → x 最小）が BL点

    Parameters
    ----------
    wq, hq        : 配置しようとする矩形 Q のサイズ
    placed_rects  : 既配置矩形のリスト [(x, y, w, h), ...]
    bin_w         : ビンの幅
    bin_h_max     : ビン高さの上限（ストリップパッキングでは大きな値を渡す）

    Returns
    -------
    BL点の座標、または見つからない場合は None
    """
    ifr_x, ifr_y, ifr_w, ifr_h = calc_ifr(bin_w, bin_h_max, wq, hq)

    if ifr_w < -1e-9 or ifr_h < -1e-9:
        return None  # ビンに入らない

    # --- 各既配置矩形の NFP を計算 ---
    nfps: list[NFPRect] = [
        calc_nfp(px, py, pw, ph, wq, hq)
        for (px, py, pw, ph) in placed_rects
    ]

    # --- 走査線に使う y 候補を列挙 ---
    # BL点の y は必ず IFR 下端か各 NFP の下端・上端のいずれか
    y_candidates: list[float] = [ifr_y]
    for (nx, ny, nw, nh) in nfps:
        y_candidates.append(ny)       # NFP 下端
        y_candidates.append(ny + nh)  # NFP 上端

    # IFR の y 範囲内に絞り込んで昇順ソート（重複除去）
    y_upper = ifr_y + ifr_h
    y_candidates = sorted(set(
        y for y in y_candidates
        if ifr_y - 1e-9 <= y <= y_upper + 1e-9
    ))

    # --- 各 y 候補で「NFP に覆われていない最小 x」を探す ---
    for y in y_candidates:
        # この y で x 方向に重なっている NFP の x 区間を収集
        # NFP の境界上（上端・下端と一致）は接触であり重なりではないため除外
        active_x_intervals: list[tuple[float, float]] = [
            (nx, nx + nw)
            for (nx, ny, nw, nh) in nfps
            if ny + 1e-9 < y < ny + nh - 1e-9
        ]

        x = _find_min_x_not_covered(ifr_x, ifr_x + ifr_w, active_x_intervals)
        if x is not None:
            return (x, y)

    return None


def _find_min_x_not_covered(
    x_min: float,
    x_max: float,
    blocked: list[tuple[float, float]],
) -> float | None:
    """
    区間 [x_min, x_max] の中で blocked のいずれにも含まれない最小 x を返す。

    アルゴリズム:
        x = x_min から始め、x が blocked 区間の内部にあれば
        その右端まで飛ばす。これを変化がなくなるまで繰り返す。

    Parameters
    ----------
    x_min, x_max : 探索する x の範囲
    blocked      : 塞がれている x 区間のリスト [(x0, x1), ...]

    Returns
    -------
    条件を満たす最小 x、なければ None
    """
    x = x_min
    improved = True
    while improved:
        improved = False
        for (bx0, bx1) in blocked:
            # x が blocked 区間 (bx0, bx1) の内部にある場合、右端へ飛ばす
            if bx0 + 1e-9 < x < bx1 - 1e-9:
                x = bx1
                improved = True
                break

    return x if x <= x_max + 1e-9 else None


# ---------------------------------------------------------------------------
# BL法の本体（bottom_left.py と同じインターフェース）
# ---------------------------------------------------------------------------

def bl_method_nfp(
    rects: list[Rect],
    bin_w: float,
    sort_key: str = "area",
) -> tuple[list[Point], list[Rect]]:
    """
    NFP + 走査線を用いた高速 Bottom-Left 法。

    bottom_left.py の bl_method() と完全に同じインターフェース。
    内部の BL点探索のみ NFP ベースに差し替えている。

    Parameters
    ----------
    rects    : [(w0,h0), ...] 配置する矩形のリスト
    bin_w    : ビンの幅
    sort_key : 並べ替えの基準（"area" / "width" / "height" / "none"）

    Returns
    -------
    positions    : [(x0,y0), ...] 各矩形の配置座標（入力順）
    sorted_rects : 実際に配置した順の矩形リスト（可視化用）
    """
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

    for wi, hi in sorted_rects:
        if wi > bin_w:
            raise ValueError(f"矩形 (w={wi}, h={hi}) がビン幅 {bin_w} を超えています。")

    placed: list[tuple[float, float, float, float]] = []
    positions_sorted: list[Point] = []

    for (wq, hq) in sorted_rects:
        bl = find_bl_point(wq, hq, placed, bin_w)
        if bl is None:
            raise RuntimeError(f"矩形 ({wq}×{hq}) を配置できる場所が見つかりませんでした。")
        x, y = bl
        placed.append((x, y, wq, hq))
        positions_sorted.append((x, y))

    positions: list[Point | None] = [None] * len(rects)
    for sorted_idx, orig_idx in enumerate(order):
        positions[orig_idx] = positions_sorted[sorted_idx]

    return positions, sorted_rects