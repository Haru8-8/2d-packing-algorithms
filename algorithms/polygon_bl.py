"""
polygon_bl.py
多角形パッキング問題に対する Bottom-Left (BL) 法。

参考文献:
    Imamichi et al., "An iterated local search algorithm based on nonlinear
    programming for the irregular strip packing problem",
    Technical Report 2007-009, Kyoto University, 2007.

    今堀慎治 他, "Python による図形詰込みアルゴリズム入門",
    オペレーションズ・リサーチ, 63(12), pp.762-769, 2018.

設計:
    - NFP・IFR を事前計算してキャッシュ（再計算なし）
    - 候補点: IFR の頂点 + 各NFPの頂点 + IFRとNFPの辺の交点
    - BL点: 候補点の中でy最小→x最小かつ全NFPの外部にある点
    - 回転角: 離散的（0, 90, 180, 270度）
"""

from __future__ import annotations

from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely import affinity

from algorithms.nfp_polygon import (
    Vertices, Angle,
    make_polygon, rotate_polygon,
    calc_nfp_polygon, calc_ifr_polygon,
    build_nfp_cache, build_ifr_cache,
)

# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------
PlacedItem = tuple[int, Polygon, tuple[float, float], Angle]
# (元インデックス, 図形, 参照点座標, 回転角)


# ---------------------------------------------------------------------------
# 候補点の列挙
# ---------------------------------------------------------------------------

def _get_edges(poly: Polygon) -> list[LineString]:
    """Polygon の辺をLineStringのリストで返す。"""
    coords = list(poly.exterior.coords)
    edges = []
    for i in range(len(coords) - 1):
        if coords[i] != coords[i + 1]:
            edges.append(LineString([coords[i], coords[i + 1]]))
    return edges


def _line_intersection(line_a: LineString, line_b: LineString) -> tuple[float, float] | None:
    """2線分の交点を返す。交点がない場合は None。"""
    pt = line_a.intersection(line_b)
    if pt.is_empty:
        return None
    if pt.geom_type == 'Point':
        return (pt.x, pt.y)
    return None


def find_bl_point_polygon(
    poly_moving:  Polygon,
    placed_items: list[PlacedItem],
    nfp_cache:    dict,
    ifr:          Polygon | None,
    moving_idx:   int,
    moving_theta: Angle,
    # nfp_cache のキー: (fixed_idx, fixed_theta, moving_idx, moving_theta)
) -> tuple[float, float] | None:
    """
    走査線アルゴリズムで多角形の BL点を探す。

    候補点の生成:
        1. IFR の頂点
        2. 各既配置図形に対する NFP の頂点
        3. IFR の辺と各 NFP の辺の交点

    BL点の選択:
        候補点の中で、IFR の内部かつ全 NFP の外部にある点を実行可能点とし、
        y最小→x最小の点を BL点とする。

    Parameters
    ----------
    poly_moving   : 配置しようとする図形（回転済み・原点正規化済み）
    placed_items  : 既配置図形のリスト [(元インデックス, 図形, 参照点座標, 回転角), ...]
    nfp_cache     : 事前計算済みの NFP キャッシュ（キー: (fixed_idx, fixed_theta, moving_idx, moving_theta)）
    ifr           : 事前計算済みの IFR
    moving_idx    : 配置しようとする図形のインデックス
    moving_theta  : 配置しようとする図形の回転角

    Returns
    -------
    BL点の座標、または見つからない場合は None
    """
    if ifr is None:
        return None

    # --- 配置済み図形の NFP を取得（キャッシュ優先・参照点分だけ平行移動）---
    translated_nfps: list[Polygon] = []
    for fixed_idx, fixed_poly, fixed_pos, fixed_theta in placed_items:
        # まずキャッシュから取得を試みる
        cache_key = (fixed_idx, fixed_theta, moving_idx, moving_theta)
        nfp = nfp_cache.get(cache_key)
        if nfp is None:
            # キャッシュにない場合は直接計算
            nfp = calc_nfp_polygon(fixed_poly, poly_moving)
        if nfp is None:
            continue
        # NFP を固定図形の参照点分だけ平行移動
        nfp_translated = affinity.translate(nfp, xoff=fixed_pos[0], yoff=fixed_pos[1])
        translated_nfps.append(nfp_translated)

    # --- 候補点の列挙 ---
    candidates: list[tuple[float, float]] = []

    # IFR の頂点
    for coord in ifr.exterior.coords[:-1]:
        candidates.append((coord[0], coord[1]))

    # 各 NFP の頂点・IFRとの辺の交点・NFP同士の辺の交点
    ifr_edges = _get_edges(ifr)
    nfp_edges_list = [_get_edges(nfp) for nfp in translated_nfps]

    for i, nfp in enumerate(translated_nfps):
        # NFP の頂点
        for coord in nfp.exterior.coords[:-1]:
            candidates.append((coord[0], coord[1]))

        nfp_edges = nfp_edges_list[i]

        # IFR の辺と NFP の辺の交点
        for ifr_edge in ifr_edges:
            for nfp_edge in nfp_edges:
                pt = _line_intersection(ifr_edge, nfp_edge)
                if pt is not None:
                    candidates.append(pt)

        # NFP 同士の辺の交点
        for j, other_nfp_edges in enumerate(nfp_edges_list):
            if i >= j:
                continue
            for nfp_edge in nfp_edges:
                for other_edge in other_nfp_edges:
                    pt = _line_intersection(nfp_edge, other_edge)
                    if pt is not None:
                        candidates.append(pt)

    if not candidates:
        return None

    # --- 実行可能点の絞り込み ---
    # y最小→x最小の順にソートして探索
    candidates = sorted(set(candidates), key=lambda p: (p[1], p[0]))

    for px, py in candidates:
        pt = Point(px, py)

        # IFR の内部または境界上にあるか
        if not (ifr.contains(pt) or ifr.boundary.contains(pt)):
            continue

        # 全 NFP の外部にあるか（境界上は許可）
        feasible = True
        for nfp in translated_nfps:
            if nfp.contains(pt):
                feasible = False
                break

        if feasible:
            return (px, py)

    return None


# ---------------------------------------------------------------------------
# BL法の本体
# ---------------------------------------------------------------------------

def bl_method_polygon(
    vertices_list: list[Vertices],
    bin_w:         float,
    orientations:  list[Angle] | None = None,
    sort_key:      str = 'area',
) -> tuple[list[tuple[float, float]], list[Angle], list[Polygon]]:
    """
    多角形パッキング問題に対する Bottom-Left 法。

    Parameters
    ----------
    vertices_list : 配置する多角形の頂点リストのリスト
    bin_w         : ビンの幅
    orientations  : 使用する回転角のリスト（デフォルト: [0, 90, 180, 270]）
    sort_key      : 並べ替えの基準（'area' / 'none'）

    Returns
    -------
    positions     : 各図形の参照点座標のリスト（入力順）
    best_thetas   : 各図形の最良回転角のリスト（入力順）
    placed_polys  : 各図形の配置済み Polygon のリスト（入力順）
    """
    if orientations is None:
        orientations = [0, 90, 180, 270]

    # 図形を Shapely Polygon に変換・正規化
    polygons = [make_polygon(v) for v in vertices_list]
    n = len(polygons)

    # 並べ替え
    if sort_key == 'area':
        order = sorted(range(n), key=lambda k: polygons[k].area, reverse=True)
    else:
        order = list(range(n))

    # NFP・IFR キャッシュを事前計算
    nfp_cache = build_nfp_cache(polygons, orientations)
    ifr_cache = build_ifr_cache(polygons, orientations, bin_w)

    placed_items: list[PlacedItem] = []
    positions_sorted:  list[tuple[float, float]] = []
    thetas_sorted:     list[Angle] = []
    polys_sorted:      list[Polygon] = []

    for orig_idx in order:
        poly = polygons[orig_idx]
        best_pos   = None
        best_theta = None
        best_poly  = None

        # 全回転角で試してy最小→x最小のBL点を選ぶ
        for theta in orientations:
            rotated = rotate_polygon(poly, theta)
            ifr = ifr_cache.get((orig_idx, theta))

            pos = find_bl_point_polygon(
                poly_moving   = rotated,
                placed_items  = placed_items,
                nfp_cache     = nfp_cache,
                ifr           = ifr,
                moving_idx    = orig_idx,
                moving_theta  = theta,
            )

            if pos is None:
                continue

            # y最小→x最小で最良点を更新
            if best_pos is None or (pos[1], pos[0]) < (best_pos[1], best_pos[0]):
                best_pos   = pos
                best_theta = theta
                best_poly  = rotated

        if best_pos is None:
            raise RuntimeError(
                f'図形 {orig_idx} を配置できる場所が見つかりませんでした。'
            )

        placed_items.append((orig_idx, best_poly, best_pos, best_theta))
        positions_sorted.append(best_pos)
        thetas_sorted.append(best_theta)
        polys_sorted.append(best_poly)

    # 入力順に戻す
    positions:   list[tuple[float, float] | None] = [None] * n
    thetas:      list[Angle | None]               = [None] * n
    placed_polys: list[Polygon | None]            = [None] * n

    for sorted_idx, orig_idx in enumerate(order):
        positions[orig_idx]    = positions_sorted[sorted_idx]
        thetas[orig_idx]       = thetas_sorted[sorted_idx]
        placed_polys[orig_idx] = polys_sorted[sorted_idx]

    return positions, thetas, placed_polys