"""
visualizer.py
パッキング結果の可視化ユーティリティ。
matplotlib を使って矩形の配置を描画する。
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np

def _setup_japanese_font() -> None:
    """環境に応じて日本語フォントを自動設定する。"""
    import matplotlib.font_manager as fm
    import platform

    candidates = {
        "Darwin":  ["Hiragino Sans", "Hiragino Maru Gothic Pro"],
        "Windows": ["Yu Gothic", "Meiryo", "MS Gothic"],
        "Linux":   ["Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic"],
    }
    os_name = platform.system()
    available = {f.name for f in fm.fontManager.ttflist}

    for font in candidates.get(os_name, []):
        if font in available:
            matplotlib.rcParams['font.family'] = font
            return

    # 見つからなかった場合は sans-serif にフォールバック
    matplotlib.rcParams['font.family'] = 'sans-serif'

_setup_japanese_font()


def plot_packing(
    rects: list[tuple[float, float]],
    positions: list[tuple[float, float]],
    bin_w: float,
    bin_h: float | None = None,
    title: str = "2D Bin Packing",
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    矩形のパッキング結果を描画する。

    Parameters
    ----------
    rects     : [(w0,h0), (w1,h1), ...] 各矩形の (幅, 高さ)
    positions : [(x0,y0), (x1,y1), ...] 各矩形の左下座標
    bin_w     : ビンの幅
    bin_h     : ビンの高さ（None の場合は配置済み矩形の最大高さを使用）
    title     : グラフタイトル
    ax        : 既存の Axes に描画する場合は渡す
    show      : True なら plt.show() を呼ぶ
    """
    if bin_h is None:
        bin_h = max(y + h for (_, h), (_, y) in zip(rects, positions))
        bin_h *= 1.05  # 少し余白

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6 * bin_h / bin_w))

    # ビン外枠
    bin_rect = patches.Rectangle(
        (0, 0), bin_w, bin_h,
        linewidth=2, edgecolor="black", facecolor="whitesmoke"
    )
    ax.add_patch(bin_rect)

    # 色マップ（アイテム数に応じて自動割り当て）
    colors = cm.tab20(np.linspace(0, 1, max(len(rects), 1)))

    for i, ((w, h), (x, y)) in enumerate(zip(rects, positions)):
        rect_patch = patches.Rectangle(
            (x, y), w, h,
            linewidth=1,
            edgecolor="white",
            facecolor=colors[i % len(colors)],
            alpha=0.85,
        )
        ax.add_patch(rect_patch)

        # アイテム番号をラベル表示
        ax.text(
            x + w / 2, y + h / 2, str(i),
            ha="center", va="center",
            fontsize=max(6, min(12, int(w * 2))),
            color="white", fontweight="bold",
        )

    # 充填率を計算してタイトルに表示
    total_area = sum(w * h for w, h in rects)
    bin_area = bin_w * bin_h
    fill_rate = total_area / bin_area * 100

    ax.set_xlim(0, bin_w)
    ax.set_ylim(0, bin_h)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n充填率: {fill_rate:.1f}%  (アイテム数: {len(rects)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_comparison(
    results: list[dict],
    bin_w: float,
    bin_h: float | None = None,
) -> None:
    """
    複数アルゴリズムの結果を横並びで比較描画する。

    Parameters
    ----------
    results : [
        {"label": str, "rects": [...], "positions": [...], "time": float},
        ...
    ]
    bin_w   : ビンの幅
    bin_h   : ビンの高さ（None の場合は自動）
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        label = result["label"]
        elapsed = result.get("time", None)
        title = f"{label}"
        if elapsed is not None:
            title += f"\n実行時間: {elapsed:.4f}s"
        plot_packing(
            result["rects"],
            result["positions"],
            bin_w,
            bin_h=bin_h,
            title=title,
            ax=ax,
            show=False,
        )

    plt.tight_layout()
    plt.show()


def plot_polygon_packing(
    polygons:  list,
    positions: list[tuple[float, float]],
    bin_w:     float,
    bin_h:     float | None = None,
    title:     str = "2D Polygon Packing",
    ax=None,
    show:      bool = True,
):
    """
    多角形のパッキング結果を描画する。

    Parameters
    ----------
    polygons  : Shapely Polygon のリスト（回転済み・原点正規化済み）
    positions : 各図形の参照点座標のリスト
    bin_w     : ビンの幅
    bin_h     : ビンの高さ（None の場合は自動）
    title     : グラフタイトル
    ax        : 既存の Axes に描画する場合は渡す
    show      : True なら plt.show() を呼ぶ
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.collections import PatchCollection
    from shapely import affinity
    import numpy as np

    if bin_h is None:
        used_h = max(
            pos[1] + poly.bounds[3]
            for poly, pos in zip(polygons, positions)
        )
        bin_h = used_h * 1.05

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6 * bin_h / bin_w))

    # ビン外枠
    bin_rect = patches.Rectangle(
        (0, 0), bin_w, bin_h,
        linewidth=2, edgecolor='black', facecolor='whitesmoke'
    )
    ax.add_patch(bin_rect)

    colors = cm.tab20(np.linspace(0, 1, max(len(polygons), 1)))

    for i, (poly, pos) in enumerate(zip(polygons, positions)):
        translated = affinity.translate(poly, xoff=pos[0], yoff=pos[1])

        coords = np.array(translated.exterior.coords)
        path = Path(coords)
        patch = PathPatch(
            path,
            facecolor=colors[i % len(colors)],
            edgecolor='white',
            alpha=0.85,
            linewidth=1,
        )
        ax.add_patch(patch)

        # ラベル（重心に表示）
        cx, cy = translated.centroid.x, translated.centroid.y
        ax.text(
            cx, cy, str(i),
            ha='center', va='center',
            fontsize=9, color='white', fontweight='bold',
        )

    total_area = sum(p.area for p in polygons)
    bin_area   = bin_w * bin_h
    fill_rate  = total_area / bin_area * 100

    ax.set_xlim(0, bin_w)
    ax.set_ylim(0, bin_h)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n充填率: {fill_rate:.1f}%  (アイテム数: {len(polygons)})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if show:
        plt.tight_layout()
        plt.show()

    return ax