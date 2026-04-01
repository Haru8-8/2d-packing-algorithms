"""
nfp_polygon.py
多角形パッキング問題における NFP (No-Fit Polygon) と
IFR (Inner-Fit Rectangle) の計算・キャッシュ管理。

設計方針:
    - 入力図形: 整数座標の頂点リスト（pyclipperとの整合性のため）
    - 内部表現: Shapely Polygon（重なり判定・NFP・IFR計算に使用）
    - NFP/IFR はすべて事前計算してキャッシュ
      → BL法の配置フェーズでは再計算なし

対応する図形:
    - 凸多角形・非凸多角形の両方
    - 回転角: 0, 90, 180, 270 度（離散的）
"""

from __future__ import annotations
import math
import pyclipper
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, Point

# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------
Vertices = list[tuple[int, int]]   # 整数座標の頂点リスト
Angle    = int                     # 回転角 (0, 90, 180, 270)


# ---------------------------------------------------------------------------
# 図形の前処理
# ---------------------------------------------------------------------------

def make_polygon(vertices: Vertices) -> Polygon:
    """
    整数座標の頂点リストから Shapely Polygon を生成する。
    参照点（バウンディングボックスの左下）が原点になるよう正規化する。
    """
    poly = Polygon(vertices)
    min_x, min_y, _, _ = poly.bounds
    return affinity.translate(poly, xoff=-min_x, yoff=-min_y)


def rotate_polygon(poly: Polygon, angle_deg: Angle) -> Polygon:
    """
    図形を参照点（左下）周りに angle_deg 度回転させ、
    バウンディングボックスの左下が原点になるよう正規化する。
    """
    if angle_deg == 0:
        return poly
    rotated = affinity.rotate(poly, angle_deg, origin=(0, 0))
    min_x, min_y, _, _ = rotated.bounds
    return affinity.translate(rotated, xoff=-min_x, yoff=-min_y)


# ---------------------------------------------------------------------------
# IFR (Inner-Fit Rectangle) の計算
# ---------------------------------------------------------------------------

def calc_ifr_polygon(
    bin_w: float,
    bin_h_max: float,
    poly: Polygon,
) -> Polygon | None:
    """
    ビン内で多角形 poly の参照点が取れる範囲（IFR）を Shapely Polygon で返す。

    計算方法:
        1. 図形の底辺（y=0）に接する頂点の中で x が最小の点を座標Aとする
        2. IFR の頂点を以下のように定義する:
               左下: (座標Aのx, 0)
               右下: (bin_w - (max_x - 座標Aのx), 0)
               右上: (bin_w - (max_x - 座標Aのx), bin_h - max_y)
               左上: (座標Aのx, bin_h - max_y)

    この方法により、バウンディングボックスベースより正確な IFR を計算できる。
    直交する辺のみを持つ図形（矩形・L字型・T字型など）で特に有効。

    Parameters
    ----------
    bin_w     : ビンの幅
    bin_h_max : ビンの高さ上限
    poly      : 配置しようとする図形（参照点が原点に正規化済み）

    Returns
    -------
    IFR の Shapely Polygon、または配置不可の場合 None
    """
    _, _, max_x, max_y = poly.bounds

    # 底辺（y=0）に接する頂点の中で x が最小のものを座標A とする
    coords = list(poly.exterior.coords[:-1])  # 末尾の重複点を除く
    bottom_coords = [x for x, y in coords if abs(y) < 1e-9]

    if not bottom_coords:
        # y=0 の頂点がない場合はバウンディングボックスで代用
        ref_x = 0.0
    else:
        ref_x = min(bottom_coords)

    ifr_left  = ref_x
    ifr_right = bin_w - (max_x - ref_x)
    ifr_top   = bin_h_max - max_y

    if ifr_right <= ifr_left or ifr_top <= 0:
        return None

    return Polygon([
        (ifr_left,  0),
        (ifr_right, 0),
        (ifr_right, ifr_top),
        (ifr_left,  ifr_top),
    ])


# ---------------------------------------------------------------------------
# NFP の計算（多角形同士）
# ---------------------------------------------------------------------------

def calc_nfp_polygon(
    poly_fixed:  Polygon,
    poly_moving: Polygon,
) -> Polygon | None:
    """
    固定図形 poly_fixed に対する移動図形 poly_moving の NFP を計算する。

    pyclipper の MinkowskiDiff を使用。
    MinkowskiDiff(A, B) = A ⊕ (-B) がNFPに対応する。

    Parameters
    ----------
    poly_fixed  : 固定図形（参照点が原点に正規化済み）
    poly_moving : 移動図形（参照点が原点に正規化済み）

    Returns
    -------
    NFP の Shapely Polygon、または計算失敗の場合 None
    """
    # Shapely Polygon → 整数座標の頂点リストに変換（pyclipper用）
    coords_fixed  = [(int(x), int(y))
                     for x, y in poly_fixed.exterior.coords[:-1]]
    coords_moving = [(int(x), int(y))
                     for x, y in poly_moving.exterior.coords[:-1]]

    try:
        result = pyclipper.MinkowskiDiff(coords_moving, coords_fixed)
    except Exception:
        return None

    if not result:
        return None

    # 複数の候補が返る場合は面積最大のものを選ぶ
    # MinkowskiDiffは自己交差したポリゴンを返すことがあるため修正が必要。
    # buffer(0)後のexterior.coordsで再生成する方法を使う。
    # （buffer(0)だけだと内部的に辺が残り境界判定が誤ることがある。
    #   exterior.coordsは正しい頂点を返すため、それで再生成すると解消できる。）
    polygons = []
    for coords in result:
        try:
            p = Polygon(coords)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_valid or p.area <= 0:
                continue
            if p.geom_type == 'MultiPolygon':
                p = max(p.geoms, key=lambda g: g.area)
            # exterior.coordsで再生成して内部的な辺を解消する
            p = Polygon(list(p.exterior.coords))
            if not p.is_valid or p.area <= 0:
                continue
            polygons.append(p)
        except Exception:
            continue

    if not polygons:
        return None

    return max(polygons, key=lambda p: p.area)


# ---------------------------------------------------------------------------
# キャッシュの構築
# ---------------------------------------------------------------------------

def build_nfp_cache(
    polygons:    list[Polygon],
    orientations: list[Angle],
) -> dict[tuple[int, Angle, int, Angle], Polygon | None]:
    """
    全図形ペア × 全回転角の組み合わせで NFP を事前計算してキャッシュする。

    Parameters
    ----------
    polygons     : 全図形のリスト（参照点が原点に正規化済み）
    orientations : 使用する回転角のリスト（例: [0, 90, 180, 270]）

    Returns
    -------
    nfp_cache[(i, θ_i, j, θ_j)] = NFP(P_i(θ_i), P_j(θ_j))
    """
    n = len(polygons)
    cache: dict[tuple[int, Angle, int, Angle], Polygon | None] = {}

    for i in range(n):
        for theta_i in orientations:
            poly_i = rotate_polygon(polygons[i], theta_i)
            for j in range(n):
                if i == j:
                    continue
                for theta_j in orientations:
                    poly_j = rotate_polygon(polygons[j], theta_j)
                    cache[(i, theta_i, j, theta_j)] = calc_nfp_polygon(
                        poly_i, poly_j
                    )

    return cache


def build_ifr_cache(
    polygons:    list[Polygon],
    orientations: list[Angle],
    bin_w:       float,
    bin_h_max:   float = 1e6,
) -> dict[tuple[int, Angle], Polygon | None]:
    """
    全図形 × 全回転角の組み合わせで IFR を事前計算してキャッシュする。

    Parameters
    ----------
    polygons     : 全図形のリスト
    orientations : 使用する回転角のリスト
    bin_w        : ビンの幅
    bin_h_max    : ビンの高さ上限

    Returns
    -------
    ifr_cache[(i, θ_i)] = IFR(bin, P_i(θ_i))
    """
    cache: dict[tuple[int, Angle], Polygon | None] = {}

    for i, poly in enumerate(polygons):
        for theta in orientations:
            rotated = rotate_polygon(poly, theta)
            cache[(i, theta)] = calc_ifr_polygon(bin_w, bin_h_max, rotated)

    return cache


# ---------------------------------------------------------------------------
# 重なり判定
# ---------------------------------------------------------------------------

def is_overlapping(
    poly_a: Polygon,
    pos_a:  tuple[float, float],
    poly_b: Polygon,
    pos_b:  tuple[float, float],
) -> bool:
    """
    座標 pos_a に配置された poly_a と、座標 pos_b に配置された poly_b が
    重なっているかを判定する。

    境界上の接触（touches）は重なりとみなさない。
    """
    translated_a = affinity.translate(poly_a, xoff=pos_a[0], yoff=pos_a[1])
    translated_b = affinity.translate(poly_b, xoff=pos_b[0], yoff=pos_b[1])
    return translated_a.overlaps(translated_b)


def is_inside_bin(
    poly: Polygon,
    pos:  tuple[float, float],
    bin_w: float,
    bin_h_max: float,
) -> bool:
    """
    座標 pos に配置された poly がビン内に収まっているかを判定する。
    """
    translated = affinity.translate(poly, xoff=pos[0], yoff=pos[1])
    min_x, min_y, max_x, max_y = translated.bounds
    return (min_x >= 0 and min_y >= 0 and
            max_x <= bin_w and max_y <= bin_h_max)