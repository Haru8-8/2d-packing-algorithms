"""
app.py
2次元パッキング アルゴリズム比較デモ（Streamlit）

タブ1: 矩形パッキング
    - BL法（単純版）O(n^4)
    - BL法（NFP版）O(n^2 log n)
    - 焼きなまし法

タブ2: 多角形パッキング
    - 多角形BL法（回転あり・非凸対応）
    - 多角形BL法 + 焼きなまし法（配置順序 × 回転角の最適化）
"""

import time
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
import streamlit as st

from algorithms.bottom_left                    import bl_method
from algorithms.nfp_bottom_left               import bl_method_nfp
from algorithms.simulated_annealing           import simulated_annealing
from algorithms.polygon_bl                    import bl_method_polygon
from algorithms.polygon_simulated_annealing   import simulated_annealing_polygon
from algorithms.nfp_polygon                   import make_polygon
from utils.visualizer                         import plot_polygon_packing

# ---------------------------------------------------------------------------
# フォント設定
# ---------------------------------------------------------------------------
def _setup_japanese_font():
    import glob
    from matplotlib import font_manager
    import matplotlib as mpl

    font_manager._load_fontmanager(try_read_cache=False)

    for font in font_manager.fontManager.ttflist:
        if 'Noto' in font.name and 'CJK' in font.name:
            mpl.rcParams['font.family'] = font.name
            mpl.rcParams['font.sans-serif'] = [font.name]
            return

    patterns = [
        '/usr/share/fonts/**/Noto*CJK*.ttc',
        '/usr/share/fonts/**/Noto*CJK*.otf',
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            font_manager.fontManager.addfont(files[0])
            prop = font_manager.FontProperties(fname=files[0])
            mpl.rcParams['font.family'] = prop.get_name()
            mpl.rcParams['font.sans-serif'] = [prop.get_name()]
            return

    # Mac環境
    candidates = ["Hiragino Sans", "Hiragino Maru Gothic Pro"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            mpl.rcParams['font.family'] = font
            return

_setup_japanese_font()

# ---------------------------------------------------------------------------
# ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="2D Packing Demo",
    page_icon="📦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------

def make_rects(n, w_min, w_max, h_min, h_max, seed):
    rng = np.random.default_rng(seed=seed)
    return [
        (float(rng.integers(w_min, w_max + 1)),
         float(rng.integers(h_min, h_max + 1)))
        for _ in range(n)
    ]


def fill_rate_rect(rects, positions, bin_w):
    used_h = max(y + h for (_, h), (_, y) in zip(rects, positions))
    return sum(w * h for w, h in rects) / (bin_w * used_h) * 100


def fill_rate_polygon(polygons, positions, bin_w):
    used_h = max(pos[1] + poly.bounds[3] for poly, pos in zip(polygons, positions))
    return sum(p.area for p in polygons) / (bin_w * used_h) * 100


def plot_rect_result(rects, positions, bin_w, title, fill, elapsed=None):
    used_h = max(y + h for (_, h), (_, y) in zip(rects, positions))
    fig, ax = plt.subplots(figsize=(5, 5 * used_h / bin_w))

    ax.add_patch(patches.Rectangle(
        (0, 0), bin_w, used_h * 1.02,
        lw=2, ec="black", fc="whitesmoke"
    ))
    colors = cm.tab20(np.linspace(0, 1, max(len(rects), 1)))

    for i, ((w, h), (x, y)) in enumerate(zip(rects, positions)):
        ax.add_patch(patches.Rectangle(
            (x, y), w, h,
            lw=1, ec="white",
            fc=colors[i % len(colors)],
            alpha=0.85,
        ))
        ax.text(
            x + w / 2, y + h / 2, str(i),
            ha="center", va="center",
            fontsize=max(6, min(10, int(w * 1.5))),
            color="white", fontweight="bold",
        )

    subtitle = f"充填率: {fill:.1f}%"
    if elapsed is not None:
        subtitle += f"  |  実行時間: {elapsed:.3f}s"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlim(0, bin_w)
    ax.set_ylim(0, used_h * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    return fig


def plot_convergence(history, best_height):
    iters        = [h[0] for h in history]
    heights_cur  = [h[1] for h in history]
    heights_best = [h[2] for h in history]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(iters, heights_cur,  alpha=0.5, color="#888780", label="現在の解")
    ax.plot(iters, heights_best, color="#1D9E75", lw=2, label="最良解")
    ax.axhline(y=best_height, color="#E24B4A", lw=1, ls="--",
               label=f"最良値 {best_height:.1f}")
    ax.set_xlabel("反復数")
    ax.set_ylabel("使用した高さ")
    ax.set_title("焼きなまし法：収束曲線")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bar_comparison(labels, fills, times):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    colors = ["#B5D4F4", "#9FE1CB", "#F5C4B3"]

    axes[0].bar(labels, fills,
                color=colors[:len(labels)],
                edgecolor=["#185FA5", "#0F6E56", "#993C1D"][:len(labels)])
    axes[0].set_ylabel("充填率 (%)")
    axes[0].set_title("充填率の比較")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(fills):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(labels, times,
                color=colors[:len(labels)],
                edgecolor=["#185FA5", "#0F6E56", "#993C1D"][:len(labels)])
    axes[1].set_ylabel("実行時間 (s)")
    axes[1].set_title("実行時間の比較")
    for i, v in enumerate(times):
        axes[1].text(i, v + max(times) * 0.01, f"{v:.3f}s", ha="center", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# メインタイトル
# ---------------------------------------------------------------------------
st.title("📦 2D パッキング：アルゴリズム比較デモ")
st.markdown("矩形・多角形を幅固定のビンに詰め込む **2次元ストリップパッキング問題** を、複数のアルゴリズムで解いて比較します。")

tab_rect, tab_poly = st.tabs(["🔲 矩形パッキング", "🔷 多角形パッキング"])

# ===========================================================================
# タブ1: 矩形パッキング
# ===========================================================================
with tab_rect:
    with st.expander("📖 アルゴリズムの説明"):
        st.markdown("""
| 手法 | 計算量 | 概要 |
|------|--------|------|
| **BL法（単純版）** | O(n⁴) | 候補点を全列挙し、最も左下に配置する貪欲法 |
| **BL法（NFP版）** | O(n²logn) | No-Fit Polygon + 走査線で高速化した BL法 |
| **焼きなまし法** | — | NFP版BL法 + 配置順序を焼きなまし法で最適化 |
""")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("⚙️ パラメータ設定")
        st.markdown("**問題設定**")
        n_rects = st.slider("矩形数 n", 5, 50, 15, 5, key="rect_n")
        bin_w_r = st.slider("ビン幅 W", 10, 60, 30, 5, key="rect_bw")
        w_min   = st.slider("幅の最小値", 1, 10, 2, key="rect_wmin")
        w_max   = st.slider("幅の最大値", 2, 15, 8, key="rect_wmax")
        h_min   = st.slider("高さの最小値", 1, 10, 2, key="rect_hmin")
        h_max   = st.slider("高さの最大値", 2, 15, 8, key="rect_hmax")
        seed_r  = st.number_input("乱数シード", 0, 999, 42, key="rect_seed")

        st.markdown("**手法の選択**")
        use_simple = st.checkbox("BL法（単純版）", value=False,
                                  help="O(n⁴)。n が大きいと時間がかかります。")
        use_nfp    = st.checkbox("BL法（NFP版）",  value=True)
        use_sa     = st.checkbox("焼きなまし法",    value=True)

        if use_sa:
            st.markdown("**焼きなまし法のパラメータ**")
            max_iter = st.select_slider("反復数", [1000,2000,5000,10000,20000], 5000, key="sa_iter")
            cooling  = st.select_slider("冷却率", [0.990,0.993,0.995,0.997,0.999], 0.995, key="sa_cool")
            t_start  = st.slider("初期温度", 1.0, 20.0, 5.0, 1.0, key="sa_temp")
            neighbor = st.radio("近傍操作", ["swap", "insert"], key="sa_nb")

        run_rect = st.button("▶ 実行", type="primary", use_container_width=True, key="run_rect")

    with col_right:
        if not run_rect:
            st.info("👈 パラメータを設定して「▶ 実行」を押してください。")
        else:
            if not (use_simple or use_nfp or use_sa):
                st.warning("手法を1つ以上選択してください。")
            else:
                if use_simple and n_rects > 30:
                    st.warning(f"⚠️ 単純版BL法は n={n_rects} だと時間がかかる場合があります。")

                rects = make_rects(n_rects, w_min, w_max, h_min, h_max, seed_r)
                results_r = {}
                sa_result = None

                with st.spinner("計算中..."):
                    if use_simple:
                        t0 = time.perf_counter()
                        pos, _ = bl_method(rects, bin_w_r, sort_key="area")
                        elapsed = time.perf_counter() - t0
                        results_r["BL法（単純版）"] = {
                            "positions": pos, "fill": fill_rate_rect(rects, pos, bin_w_r), "elapsed": elapsed
                        }

                    if use_nfp:
                        t0 = time.perf_counter()
                        pos, _ = bl_method_nfp(rects, bin_w_r, sort_key="area")
                        elapsed = time.perf_counter() - t0
                        results_r["BL法（NFP版）"] = {
                            "positions": pos, "fill": fill_rate_rect(rects, pos, bin_w_r), "elapsed": elapsed
                        }

                    if use_sa:
                        sa_result = simulated_annealing(
                            rects, bin_w_r,
                            t_start=t_start, t_end=0.01,
                            cooling=cooling, max_iter=max_iter,
                            neighbor=neighbor, seed=int(seed_r),
                            log_interval=max(1, max_iter // 50),
                        )
                        results_r["焼きなまし法"] = {
                            "positions": sa_result.best_positions,
                            "fill": fill_rate_rect(rects, sa_result.best_positions, bin_w_r),
                            "elapsed": sa_result.elapsed,
                        }

                st.subheader("📊 配置結果")
                cols = st.columns(len(results_r))
                for col, (label, res) in zip(cols, results_r.items()):
                    with col:
                        fig = plot_rect_result(rects, res["positions"], bin_w_r,
                                               label, res["fill"], res["elapsed"])
                        st.pyplot(fig)
                        plt.close(fig)

                st.subheader("📈 比較グラフ")
                labels = list(results_r.keys())
                fills  = [results_r[l]["fill"]    for l in labels]
                times  = [results_r[l]["elapsed"] for l in labels]
                fig_bar = plot_bar_comparison(labels, fills, times)
                st.pyplot(fig_bar)
                plt.close(fig_bar)

                if use_sa and sa_result is not None:
                    st.subheader("🌡️ 焼きなまし法：収束の様子")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig_conv = plot_convergence(sa_result.history, sa_result.best_height)
                        st.pyplot(fig_conv)
                        plt.close(fig_conv)
                    with col2:
                        init_fill = fill_rate_rect(
                            rects, bl_method_nfp(rects, bin_w_r, sort_key="area")[0], bin_w_r
                        )
                        best_fill = results_r["焼きなまし法"]["fill"]
                        st.metric("初期充填率", f"{init_fill:.1f}%")
                        st.metric("最良充填率", f"{best_fill:.1f}%",
                                  delta=f"{best_fill - init_fill:+.1f}%")

                st.subheader("📋 数値サマリー")
                st.dataframe({
                    "手法": labels,
                    "充填率 (%)": [f"{f:.1f}" for f in fills],
                    "実行時間 (s)": [f"{t:.4f}" for t in times],
                }, use_container_width=True)

# ===========================================================================
# タブ2: 多角形パッキング
# ===========================================================================
with tab_poly:
    with st.expander("📖 アルゴリズムの説明"):
        st.markdown("""
| 手法 | 概要 |
|------|------|
| **多角形BL法** | NFP + IFR を用いた多角形対応の Bottom-Left 法。回転角を離散的に試して最良点を選択。 |
| **焼きなまし法** | 多角形BL法 + 配置順序 × 回転角の組み合わせを焼きなまし法で最適化。 |

**近傍操作（焼きなまし法）**
- `swap`   : 配置順序の2要素をスワップ
- `insert` : 配置順序の1要素を別の位置に挿入
- `rotate` : 1図形の回転角をランダムに変更
- `mixed`  : 上記3つをランダムに選択（推奨）
""")

    # ランダム生成用のベース図形
    BASE_SHAPES = [
        [(0,0),(20,0),(20,10),(10,10),(10,20),(0,20)],   # L字型
        [(0,0),(30,0),(30,10),(20,10),(20,20),(10,20),(10,10),(0,10)],  # U字型
        [(0,0),(15,0),(15,15),(0,15)],                   # 正方形
        [(0,0),(30,0),(30,8),(0,8)],                     # 横長矩形
        [(0,0),(8,0),(8,30),(0,30)],                     # 縦長矩形
        [(0,0),(20,0),(20,20),(10,20),(10,10),(0,10)],   # 逆L字型
        [(0,0),(25,0),(25,10),(0,10)],                   # 横長矩形2
        [(0,0),(10,0),(10,25),(0,25)],                   # 縦長矩形2
        [(0,0),(15,0),(15,20),(0,20)],                   # 縦長矩形3
        [(0,0),(20,0),(20,15),(0,15)],                   # 矩形
    ]

    # プリセット図形の定義
    PRESETS = {
        "end0tknr例題（L字・三角形）": {
            "vertices_list": [
                [(0,0),(20,0),(20,50),(0,50)],
                [(0,0),(40,0),(40,50),(20,50),(20,30),(0,30)],
                [(0,0),(20,0),(20,20),(0,20)],
                [(0,0),(40,0),(40,30)],
                [(0,0),(40,30),(40,50),(0,50)],
            ],
            "bin_w": 120,
        },
        "非凸図形セット（L字・U字）": {
            "vertices_list": [
                [(0,0),(20,0),(20,10),(10,10),(10,20),(0,20)],
                [(0,0),(30,0),(30,10),(20,10),(20,20),(10,20),(10,10),(0,10)],
                [(0,0),(10,0),(10,10),(0,10)],
                [(0,0),(20,0),(20,5),(0,5)],
                [(0,0),(5,0),(5,20),(0,20)],
            ],
            "bin_w": 60,
        },
        "縦横比が異なる矩形（回転の効果が出やすい）": {
            "vertices_list": [
                [(0,0),(40,0),(40,10),(0,10)],
                [(0,0),(5,0),(5,30),(0,30)],
                [(0,0),(30,0),(30,5),(0,5)],
                [(0,0),(20,0),(20,15),(0,15)],
                [(0,0),(8,0),(8,25),(0,25)],
            ],
            "bin_w": 50,
        },
        "ランダム生成（大量図形）": {
            "vertices_list": None,  # 動的に生成
            "bin_w": 100,
        },
    }

    col_left2, col_right2 = st.columns([1, 3])

    with col_left2:
        st.subheader("⚙️ パラメータ設定")

        preset_name = st.selectbox("プリセット図形", list(PRESETS.keys()))
        preset = PRESETS[preset_name]

        bin_w_p = st.slider("ビン幅 W", 30, 200,
                            preset["bin_w"], 10, key="poly_bw")

        if preset_name == "ランダム生成（大量図形）":
            n_poly = st.slider("図形数 n", 10, 50, 20, 5, key="poly_n")
            seed_p = st.number_input("乱数シード", 0, 999, 42, key="poly_seed")
        else:
            n_poly = None
            seed_p = None

        st.markdown("**回転角の設定**")
        st.caption("回転なし（0度のみ）と比較します")
        use_0   = st.checkbox("0度",   value=True,  key="rot0")
        use_90  = st.checkbox("90度",  value=True,  key="rot90")
        use_180 = st.checkbox("180度", value=False, key="rot180")
        use_270 = st.checkbox("270度", value=False, key="rot270")

        st.markdown("**手法の選択**")
        use_poly_sa = st.checkbox("焼きなまし法", value=True, key="poly_use_sa")

        if use_poly_sa:
            st.markdown("**焼きなまし法のパラメータ**")
            poly_max_iter = st.select_slider(
                "反復数", [500, 1000, 2000, 5000, 10000], 2000, key="poly_sa_iter",
                help="多角形はBL法の評価コストが高いため、矩形より少ない反復数を推奨します。"
            )
            poly_cooling  = st.select_slider(
                "冷却率", [0.990, 0.993, 0.995, 0.997, 0.999], 0.995, key="poly_sa_cool"
            )
            poly_t_start  = st.slider("初期温度", 1.0, 20.0, 5.0, 1.0, key="poly_sa_temp")
            poly_neighbor = st.radio(
                "近傍操作", ["mixed", "swap", "insert", "rotate"], key="poly_sa_nb"
            )

        run_poly = st.button("▶ 実行", type="primary",
                             use_container_width=True, key="run_poly")

    with col_right2:
        if not run_poly:
            st.info("👈 パラメータを設定して「▶ 実行」を押してください。")
        else:
            orientations = []
            if use_0:   orientations.append(0)
            if use_90:  orientations.append(90)
            if use_180: orientations.append(180)
            if use_270: orientations.append(270)

            if not orientations:
                st.warning("回転角を1つ以上選択してください。")
                st.stop()

            # ランダム生成の場合はベース図形からランダムに選ぶ
            if preset["vertices_list"] is None:
                import random
                rng_p = random.Random(int(seed_p))
                vertices_list = [
                    rng_p.choice(BASE_SHAPES) for _ in range(n_poly)
                ]
            else:
                vertices_list = preset["vertices_list"]

            # 焼きなまし法の計算時間警告
            if use_poly_sa:
                n_shapes = len(vertices_list)
                n_rot    = len(orientations)
                # 実測値（25図形・4回転・1000反復で約3000秒）をもとに目安を算出
                # 評価1回あたり約3秒 @ n=25, rot=4 → n²×rot に比例して粗く推定
                sec_per_eval = 3.0 * (n_shapes / 25) ** 2 * (n_rot / 4)
                est_sec = sec_per_eval * poly_max_iter
                if est_sec >= 60:
                    est_str = f"約 {est_sec/60:.0f} 分" if est_sec < 3600 else f"約 {est_sec/3600:.1f} 時間"
                    st.warning(
                        f"⚠️ 焼きなまし法の推定実行時間: **{est_str}**（図形数 {n_shapes}、"
                        f"回転角 {n_rot} 種、反復数 {poly_max_iter}）。"
                        f"図形数10以下・反復数1000以下での実行を推奨します。"
                    )
                elif est_sec >= 10:
                    st.info(
                        f"ℹ️ 焼きなまし法の推定実行時間: **約 {est_sec:.0f} 秒**。"
                        f"しばらくお待ちください。"
                    )

            with st.spinner("NFPを計算中...（図形数・回転角数によって時間がかかります）"):
                # 回転なし（BL法）
                t0 = time.perf_counter()
                pos_no, thetas_no, polys_no = bl_method_polygon(
                    vertices_list, bin_w_p,
                    orientations=[0], sort_key='area',
                )
                t_no = time.perf_counter() - t0

                # 回転あり（BL法）
                t0 = time.perf_counter()
                pos_rot, thetas_rot, polys_rot = bl_method_polygon(
                    vertices_list, bin_w_p,
                    orientations=orientations, sort_key='area',
                )
                t_rot = time.perf_counter() - t0

                # 焼きなまし法
                sa_poly_result = None
                if use_poly_sa:
                    sa_poly_result = simulated_annealing_polygon(
                        vertices_list, bin_w_p,
                        orientations=orientations,
                        t_start=poly_t_start, t_end=0.01,
                        cooling=poly_cooling, max_iter=poly_max_iter,
                        neighbor=poly_neighbor,
                        seed=int(seed_p) if seed_p is not None else 42,
                        log_interval=max(1, poly_max_iter // 50),
                    )

            fill_no  = fill_rate_polygon(polys_no,  pos_no,  bin_w_p)
            fill_rot = fill_rate_polygon(polys_rot, pos_rot, bin_w_p)

            # ------------------------------------------------------------------
            # 配置結果の表示
            # ------------------------------------------------------------------
            st.subheader("📊 配置結果の比較")

            n_cols = 3 if use_poly_sa else 2
            cols_res = st.columns(n_cols)

            with cols_res[0]:
                st.markdown("**回転なし（BL法）**")
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                plot_polygon_packing(
                    polys_no, pos_no, bin_w_p,
                    title=f"回転なし\n充填率: {fill_no:.1f}%  |  {t_no:.3f}s",
                    ax=ax1, show=False,
                )
                st.pyplot(fig1)
                plt.close(fig1)

            with cols_res[1]:
                st.markdown(f"**回転あり（BL法）{orientations}度**")
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                plot_polygon_packing(
                    polys_rot, pos_rot, bin_w_p,
                    title=f"回転あり {orientations}\n充填率: {fill_rot:.1f}%  |  {t_rot:.3f}s",
                    ax=ax2, show=False,
                )
                st.pyplot(fig2)
                plt.close(fig2)

            if use_poly_sa and sa_poly_result is not None:
                fill_sa = fill_rate_polygon(
                    sa_poly_result.best_polys, sa_poly_result.best_positions, bin_w_p
                )
                with cols_res[2]:
                    st.markdown("**焼きなまし法**")
                    fig3, ax3 = plt.subplots(figsize=(5, 4))
                    plot_polygon_packing(
                        sa_poly_result.best_polys, sa_poly_result.best_positions, bin_w_p,
                        title=f"焼きなまし法\n充填率: {fill_sa:.1f}%  |  {sa_poly_result.elapsed:.3f}s",
                        ax=ax3, show=False,
                    )
                    st.pyplot(fig3)
                    plt.close(fig3)

            # ------------------------------------------------------------------
            # 比較グラフ・メトリクス
            # ------------------------------------------------------------------
            st.subheader("📈 充填率の比較")

            if use_poly_sa and sa_poly_result is not None:
                labels_p = ["BL法（回転なし）", "BL法（回転あり）", "焼きなまし法"]
                fills_p  = [fill_no, fill_rot, fill_sa]
                times_p  = [t_no, t_rot, sa_poly_result.elapsed]
            else:
                labels_p = ["BL法（回転なし）", "BL法（回転あり）"]
                fills_p  = [fill_no, fill_rot]
                times_p  = [t_no, t_rot]

            fig_bar_p = plot_bar_comparison(labels_p, fills_p, times_p)
            st.pyplot(fig_bar_p)
            plt.close(fig_bar_p)

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("BL法（回転なし）", f"{fill_no:.1f}%")
            with col_m2:
                st.metric("BL法（回転あり）", f"{fill_rot:.1f}%",
                          delta=f"{fill_rot - fill_no:+.1f}%")
            with col_m3:
                if use_poly_sa and sa_poly_result is not None:
                    st.metric("焼きなまし法", f"{fill_sa:.1f}%",
                              delta=f"{fill_sa - fill_rot:+.1f}% vs BL回転あり")

            # ------------------------------------------------------------------
            # 焼きなまし法：収束の様子
            # ------------------------------------------------------------------
            if use_poly_sa and sa_poly_result is not None:
                st.subheader("🌡️ 焼きなまし法：収束の様子")
                col_conv1, col_conv2 = st.columns([2, 1])
                with col_conv1:
                    fig_conv = plot_convergence(
                        sa_poly_result.history, sa_poly_result.best_height
                    )
                    st.pyplot(fig_conv)
                    plt.close(fig_conv)
                with col_conv2:
                    st.metric("初期充填率（BL法）", f"{fill_rot:.1f}%")
                    st.metric("最良充填率（SA）",   f"{fill_sa:.1f}%",
                              delta=f"{fill_sa - fill_rot:+.1f}%")
                    st.metric("実行時間", f"{sa_poly_result.elapsed:.2f}s")

                    st.markdown("**最良解の回転角**")
                    theta_df = {
                        "図形": [f"図形{i}" for i in range(len(vertices_list))],
                        "回転角 (度)": [sa_poly_result.best_thetas[i]
                                       for i in range(len(vertices_list))],
                    }
                    st.dataframe(theta_df, use_container_width=True)

            # ------------------------------------------------------------------
            # 数値サマリー
            # ------------------------------------------------------------------
            st.subheader("📋 数値サマリー")
            st.dataframe({
                "手法":        labels_p,
                "充填率 (%)":  [f"{f:.1f}" for f in fills_p],
                "実行時間 (s)": [f"{t:.4f}" for t in times_p],
            }, use_container_width=True)