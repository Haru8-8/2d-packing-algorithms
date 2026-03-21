# 2D Packing Algorithms

2次元矩形パッキング問題に対する複数アルゴリズムの実装・比較ポートフォリオ。

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://2d-packing-algorithms-r3lvpgajmvk2earrnhl3dk.streamlit.app/)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## デモ

Streamlit Cloud でインタラクティブデモを公開しています。  
→ **https://2d-packing-algorithms-r3lvpgajmvk2earrnhl3dk.streamlit.app/**

## 実装手法

| 手法 | 計算量 | ファイル |
|------|--------|---------|
| BL法（単純版） | O(n⁴) | `algorithms/bottom_left.py` |
| BL法（NFP版） | O(n²logn) | `algorithms/nfp_bottom_left.py` |
| 焼きなまし法 | — | `algorithms/simulated_annealing.py` |

## ファイル構成

```
packing/
├── app.py                           # Streamlit デモアプリ
├── requirements.txt                 # 依存ライブラリ
├── algorithms/
│   ├── bottom_left.py               # BL法（単純版 O(n^4)）
│   ├── nfp_bottom_left.py           # BL法（NFP版 O(n^2 log n)）
│   └── simulated_annealing.py       # 焼きなまし法
├── utils/
│   └── visualizer.py                # matplotlib 可視化ユーティリティ
└── notebooks/
    ├── 01_bottom_left.ipynb         # BL法の復習・動作確認
    ├── 02_nfp_bottom_left.ipynb     # NFP実装・速度比較ベンチマーク
    └── 03_simulated_annealing.ipynb # 焼きなまし法・収束確認
```

## ローカルで実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ベンチマーク結果

### 単純版 vs NFP版（速度比較）

| n | 単純版 (s) | NFP版 (s) | 高速化倍率 |
|---|-----------|----------|-----------|
| 20 | 0.0100 | 0.0003 | 35x |
| 50 | 0.2928 | 0.0023 | 125x |
| 100 | 4.2966 | 0.0175 | 246x |

### BL法 vs 焼きなまし法（充填率比較）

10問の平均: BL法 **86.1%** → 焼きなまし法 **94.9%**（平均 **+8.8%** 改善）

## ライセンス

[MIT License](LICENSE)

## 参考文献

- 今堀慎治 他, "Python による図形詰込みアルゴリズム入門", オペレーションズ・リサーチ, 63(12), pp.762-769, 2018.
- 川島大貴 他, "3次元パッキングに対する効率的な bottom-left", 数理解析研究所講究録, 1726, pp.50-61, 2011.
- Imamichi et al., "An iterated local search algorithm based on nonlinear programming for the irregular strip packing problem", Technical Report 2007-009, Kyoto University, 2007.