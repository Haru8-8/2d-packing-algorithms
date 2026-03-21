"""
app.py
2次元矩形パッキング アルゴリズム比較デモ（Streamlit）

手法:
    1. BL法（単純版）   O(n^4)
    2. BL法（NFP版）    O(n^2 log n)
    3. 焼きなまし法     NFP版BL法 + 配置順序の最適化
"""

import time
import matplotlib
matplotlib.use("Agg")  # Streamlit では非インタラクティブバックエンドを使用
import japanize_matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
import streamlit as st

from algorithms.bottom_left         import bl_method
from algorithms.nfp_bottom_left     import bl_method_nfp
from algorithms.simulated_annealing import simulated_annealing

# ---------------------------------------------------------------------------
# ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="2D Packing Demo",
    page_icon="📦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

def make_rects(n: int, w_min: int, w_max: int, h_min: int, h_max: int, seed: int):
    rng = np.random.default_rng(seed=seed)
    return [
        (float(rng.integers(w_min, w_max + 1)),
         float(rng.integers(h_min, h_max + 1)))
        for _ in range(n)
    ]


def fill_rate(rects, positions, bin_w) -> float:
    used_h = max(y + h for (_, h), (_, y) in zip(rects, positions))
    total  = sum(w * h for w, h in rects)
    return total / (bin_w * used_h) * 100


def plot_result(rects, positions, bin_w, title, fill, elapsed=None):
    """配置結果を matplotlib Figure で返す。"""
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


def plot_convergence(history, initial_height, best_height):
    """焼きなまし法の収束曲線を返す。"""
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
    """充填率と実行時間の棒グラフを返す。"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    colors = ["#B5D4F4", "#9FE1CB", "#F5C4B3"]

    axes[0].bar(labels, fills, color=colors[:len(labels)],
                edgecolor=["#185FA5", "#0F6E56", "#993C1D"][:len(labels)])
    axes[0].set_ylabel("充填率 (%)")
    axes[0].set_title("充填率の比較")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(fills):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(labels, times, color=colors[:len(labels)],
                edgecolor=["#185FA5", "#0F6E56", "#993C1D"][:len(labels)])
    axes[1].set_ylabel("実行時間 (s)")
    axes[1].set_title("実行時間の比較")
    for i, v in enumerate(times):
        axes[1].text(i, v + max(times) * 0.01, f"{v:.3f}s", ha="center", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# サイドバー：パラメータ設定
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ パラメータ設定")

st.sidebar.subheader("問題設定")
n_rects = st.sidebar.slider("矩形数 n", min_value=5, max_value=50, value=15, step=5)
bin_w   = st.sidebar.slider("ビン幅 W", min_value=10, max_value=60, value=30, step=5)
w_min   = st.sidebar.slider("幅の最小値", min_value=1, max_value=10, value=2)
w_max   = st.sidebar.slider("幅の最大値", min_value=2, max_value=15, value=8)
h_min   = st.sidebar.slider("高さの最小値", min_value=1, max_value=10, value=2)
h_max   = st.sidebar.slider("高さの最大値", min_value=2, max_value=15, value=8)
seed    = st.sidebar.number_input("乱数シード", min_value=0, max_value=999, value=42)

st.sidebar.subheader("手法の選択")
use_simple = st.sidebar.checkbox("BL法（単純版）", value=False,
                                  help="O(n⁴)。n が大きいと時間がかかります。")
use_nfp    = st.sidebar.checkbox("BL法（NFP版）",  value=True,
                                  help="O(n²logn)。高速で単純版と同じ結果。")
use_sa     = st.sidebar.checkbox("焼きなまし法",    value=True,
                                  help="NFP版BL法 + 配置順序の最適化。")

if use_sa:
    st.sidebar.subheader("焼きなまし法のパラメータ")
    max_iter = st.sidebar.select_slider(
        "反復数", options=[1000, 2000, 5000, 10000, 20000], value=5000
    )
    cooling  = st.sidebar.select_slider(
        "冷却率", options=[0.990, 0.993, 0.995, 0.997, 0.999], value=0.995
    )
    t_start  = st.sidebar.slider("初期温度", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
    neighbor = st.sidebar.radio("近傍操作", ["swap", "insert"], index=0)

run_btn = st.sidebar.button("▶ 実行", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# メインエリア
# ---------------------------------------------------------------------------
st.title("📦 2D 矩形パッキング：アルゴリズム比較デモ")
st.markdown("""
矩形をビン（幅固定・高さ可変）に詰め込む **2次元ストリップパッキング問題** を、
3つのアルゴリズムで解いて比較します。
""")

with st.expander("📖 アルゴリズムの説明"):
    st.markdown("""
| 手法 | 計算量 | 概要 |
|------|--------|------|
| **BL法（単純版）** | O(n⁴) | 候補点を全列挙し、最も左下に配置する貪欲法 |
| **BL法（NFP版）** | O(n²logn) | No-Fit Polygon + 走査線で高速化した BL法 |
| **焼きなまし法** | — | NFP版BL法 + 配置順序を焼きなまし法で最適化 |

**参考文献:**
- 今堀慎治 他, "Python による図形詰込みアルゴリズム入門", OR 学会誌, 2018.
- 今堀・今堀・柳浦, "3次元パッキングに対する効率的な bottom-left", 数理解析研究所, 2011.
- Imamichi et al., "An iterated local search algorithm based on nonlinear programming
  for the irregular strip packing problem", Technical Report, Kyoto University, 2007.
""")

if not run_btn:
    st.info("👈 サイドバーでパラメータを設定して「▶ 実行」を押してください。")
    st.stop()

if not (use_simple or use_nfp or use_sa):
    st.warning("手法を1つ以上選択してください。")
    st.stop()

if use_simple and n_rects > 30:
    st.warning(f"⚠️ 単純版BL法は n={n_rects} だと時間がかかる場合があります（O(n⁴)）。")

# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------
rects = make_rects(n_rects, w_min, w_max, h_min, h_max, seed)

results   = {}  # label -> {positions, fill, elapsed}
sa_result = None

with st.spinner("計算中..."):
    if use_simple:
        t0 = time.perf_counter()
        pos, _ = bl_method(rects, bin_w, sort_key="area")
        elapsed = time.perf_counter() - t0
        results["BL法（単純版）"] = {
            "positions": pos,
            "fill": fill_rate(rects, pos, bin_w),
            "elapsed": elapsed,
        }

    if use_nfp:
        t0 = time.perf_counter()
        pos, _ = bl_method_nfp(rects, bin_w, sort_key="area")
        elapsed = time.perf_counter() - t0
        results["BL法（NFP版）"] = {
            "positions": pos,
            "fill": fill_rate(rects, pos, bin_w),
            "elapsed": elapsed,
        }

    if use_sa:
        sa_result = simulated_annealing(
            rects, bin_w,
            t_start=t_start, t_end=0.01,
            cooling=cooling,
            max_iter=max_iter,
            neighbor=neighbor,
            seed=int(seed),
            log_interval=max(1, max_iter // 50),
        )
        results["焼きなまし法"] = {
            "positions": sa_result.best_positions,
            "fill": fill_rate(rects, sa_result.best_positions, bin_w),
            "elapsed": sa_result.elapsed,
        }

# ---------------------------------------------------------------------------
# 結果表示：配置図
# ---------------------------------------------------------------------------
st.subheader("📊 配置結果")

cols = st.columns(len(results))
for col, (label, res) in zip(cols, results.items()):
    with col:
        fig = plot_result(
            rects, res["positions"], bin_w,
            label, res["fill"], res["elapsed"]
        )
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------------------
# 結果表示：比較グラフ
# ---------------------------------------------------------------------------
st.subheader("📈 比較グラフ")

labels = list(results.keys())
fills  = [results[l]["fill"]    for l in labels]
times  = [results[l]["elapsed"] for l in labels]

fig_bar = plot_bar_comparison(labels, fills, times)
st.pyplot(fig_bar)
plt.close(fig_bar)

# ---------------------------------------------------------------------------
# 焼きなまし法：収束曲線
# ---------------------------------------------------------------------------
if use_sa and sa_result is not None:
    st.subheader("🌡️ 焼きなまし法：収束の様子")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_conv = plot_convergence(
            sa_result.history,
            sa_result.initial_height,
            sa_result.best_height,
        )
        st.pyplot(fig_conv)
        plt.close(fig_conv)
    with col2:
        st.metric("初期解の高さ", f"{sa_result.initial_height:.1f}")
        st.metric("最良解の高さ", f"{sa_result.best_height:.1f}",
                  delta=f"{sa_result.best_height - sa_result.initial_height:.1f}")
        init_fill = fill_rate(rects,
                              bl_method_nfp(rects, bin_w, sort_key="area")[0],
                              bin_w)
        best_fill = results["焼きなまし法"]["fill"]
        st.metric("初期充填率", f"{init_fill:.1f}%")
        st.metric("最良充填率", f"{best_fill:.1f}%",
                  delta=f"{best_fill - init_fill:+.1f}%")

# ---------------------------------------------------------------------------
# 数値サマリー
# ---------------------------------------------------------------------------
st.subheader("📋 数値サマリー")
summary_data = {
    "手法": labels,
    "充填率 (%)": [f"{f:.1f}" for f in fills],
    "実行時間 (s)": [f"{t:.4f}" for t in times],
}
st.dataframe(summary_data, use_container_width=True)