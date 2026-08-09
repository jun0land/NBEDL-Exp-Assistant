"""분석 방법 공용 모듈: 이상치 판정(여러 방법) · 강건 축약 · 방법/유의수준 컨트롤 · 박스플롯.

이상치 판정에 절대적 정답은 없으므로 성격이 다른 방법을 여러 개 제공한다. 강건 평균(app.py),
제외 목록(origin_charts), 박스플롯이 모두 이 모듈의 outlier_mask 를 써서 기준을 통일한다.

방법:
  none  : 제거 안 함
  iqr   : Tukey 박스플롯 (Q1-1.5·IQR ~ Q3+1.5·IQR 밖)
  zscore: 표준점수 |z| > 3 (평균·표준편차)
  mad   : 수정된 Z-점수 |Mi| > 3.5 (중앙값·MAD, 소표본에 강건)
  esd   : 일반화 극단 스튜던트화 편차(Rosner), 유의수준 α

판정은 '같은 공정 조건 반복 측정' 그룹 안에서 이루어지며, 그룹 크기 3 미만이면 판정하지 않는다.
그룹의 값이 전부 이상치로 나오면(=기준이 무의미) 아무것도 제거하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MIN_GROUP = 3  # 이 개수 미만이면 이상치 판정 안 함 (통계적으로 무의미)

# (key, 화면 라벨). 순서가 셀렉트박스 순서.
OUTLIER_METHODS = [
    ("iqr", "IQR / 박스플롯 (1.5×IQR)"),
    ("zscore", "Z-점수 (|z| > 3)"),
    ("mad", "수정된 Z-점수 · MAD (> 3.5)"),
    ("esd", "ESD · 유의수준 α (Rosner)"),
    ("none", "제거 안 함"),
]
_METHOD_KEYS = [m[0] for m in OUTLIER_METHODS]
_METHOD_LABEL = dict(OUTLIER_METHODS)


def method_label(method):
    return _METHOD_LABEL.get(method, method)


def method_uses_alpha(method):
    """유의수준 α 가 의미 있는 방법인지 (현재 ESD 만)."""
    return method == "esd"


# ---------------------------------------------------------------------------
# 개별 방법 (유한값만 담긴 1D 배열을 받아, 같은 길이의 bool 이상치 마스크 반환)
# ---------------------------------------------------------------------------
def _iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (x < lo) | (x > hi)


def _zscore(x):
    sd = x.std(ddof=1)
    if sd == 0:
        return np.zeros(len(x), bool)
    return np.abs((x - x.mean()) / sd) > 3.0


def _mad(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros(len(x), bool)
    mi = 0.6745 * (x - med) / mad
    return np.abs(mi) > 3.5


def _esd(x, alpha):
    """일반화 ESD (Rosner). 최대 n//2 개까지 후보로 검정해 이상치 개수를 정한다."""
    from scipy.stats import t as tdist

    n = len(x)
    max_out = max(1, n // 2)
    active = np.ones(n, bool)
    order, R, L = [], [], []
    for _ in range(max_out):
        nn = int(active.sum())
        if nn < 3:
            break
        vals = x[active]
        sd = vals.std(ddof=1)
        if sd == 0:
            break
        m = vals.mean()
        dev = np.where(active, np.abs(x - m), -1.0)
        j = int(np.argmax(dev))
        R.append(abs(x[j] - m) / sd)
        p = 1.0 - alpha / (2.0 * nn)
        tc = tdist.ppf(p, nn - 2)
        L.append((nn - 1) * tc / np.sqrt((nn - 2 + tc ** 2) * nn))
        order.append(j)
        active[j] = False
    n_out = 0
    for i in range(len(R)):
        if R[i] > L[i]:
            n_out = i + 1
    mask = np.zeros(n, bool)
    for k in range(n_out):
        mask[order[k]] = True
    return mask


def outlier_mask(values, method="iqr", alpha=0.05):
    """values(1D, NaN 허용) 에서 이상치 위치를 True 로 표시한 bool 배열 반환.

    NaN 은 절대 이상치로 보지 않는다. 유한값이 MIN_GROUP 미만이면 전부 False.
    방법이 그룹 전체를 이상치로 지목하면(무의미) 전부 False 로 되돌린다.
    """
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(len(x), bool)
    if method == "none":
        return mask
    finite = np.isfinite(x)
    xf = x[finite]
    if len(xf) < MIN_GROUP:
        return mask

    if method == "iqr":
        sub = _iqr(xf)
    elif method == "zscore":
        sub = _zscore(xf)
    elif method == "mad":
        sub = _mad(xf)
    elif method == "esd":
        sub = _esd(xf, float(alpha))
    else:
        sub = np.zeros(len(xf), bool)

    if sub.all():  # 전부 이상치면 판정 무의미 → 아무것도 제거하지 않음
        sub = np.zeros(len(xf), bool)
    mask[finite] = sub
    return mask


def robust_reduce(values, method="iqr", alpha=0.05):
    """이상치를 뺀 값 리스트를 반환. 남는 게 없으면 원본 유지. 강건 평균 계산용."""
    vals = list(values)
    mask = outlier_mask(vals, method, alpha)
    kept = [v for v, m in zip(vals, mask) if not m]
    return kept if kept else vals


# ---------------------------------------------------------------------------
# 방법/유의수준 선택 UI
# ---------------------------------------------------------------------------
_ALPHA_PRESETS = {"0.05 (기본)": 0.05, "0.01 (엄격)": 0.01, "직접 입력": None}


def render_method_controls(method, alpha, *, key_prefix="outlier"):
    """이상치 방법 + α 선택 위젯. (method, alpha) 를 반환 — 호출부가 세션/파일에 저장한다."""
    c1, c2 = st.columns([1.4, 1])
    idx = _METHOD_KEYS.index(method) if method in _METHOD_KEYS else 0
    sel = c1.selectbox("이상치 판정 방법", _METHOD_KEYS, index=idx,
                       format_func=method_label, key=f"{key_prefix}_method")
    out_alpha = float(alpha)
    if method_uses_alpha(sel):
        preset_keys = list(_ALPHA_PRESETS)
        # 현재 α 가 프리셋과 다르면 '직접 입력'을 기본 선택
        cur = "직접 입력"
        for k, v in _ALPHA_PRESETS.items():
            if v is not None and abs(v - float(alpha)) < 1e-12:
                cur = k
                break
        pk = c2.selectbox("유의수준 α", preset_keys, index=preset_keys.index(cur),
                          key=f"{key_prefix}_alpha_preset")
        if _ALPHA_PRESETS[pk] is None:
            out_alpha = c2.number_input("α 직접 입력", min_value=0.0001, max_value=0.5,
                                        value=float(alpha), step=0.005, format="%.4f",
                                        key=f"{key_prefix}_alpha_custom")
        else:
            out_alpha = _ALPHA_PRESETS[pk]
        st.caption("α 가 클수록 이상치를 더 많이 제거해 학습에 쓰이는 데이터가 줄어듭니다. "
                   "α 는 ESD 에만 적용되며, 다른 방법은 관례 상수(1.5×IQR · z=3 · MAD 3.5)를 씁니다.")
    else:
        st.caption("현재 방법은 유의수준 대신 관례 상수를 사용합니다 "
                   "(IQR 1.5× · Z-점수 3 · 수정 Z-점수 3.5). 판정은 같은 조건 3회 이상 반복 그룹에서만.")
    return sel, out_alpha


# ---------------------------------------------------------------------------
# 박스플롯 (반복 측정 분포 + 이상치)
# ---------------------------------------------------------------------------
def render_boxplots(df, config_vars, target_vars, method, alpha, *, key_prefix="box"):
    """선택한 공정 변수로 그룹을 나눠, 목표별 값 분포를 박스플롯으로. 현재 방법이 이상치로
    지목한 점은 빨강, 나머지는 검정으로 겹쳐 그린다."""
    cvars = [v["Name"] for v in config_vars if v.get("Name") and v["Name"] in df.columns]
    targets = [tv for tv in target_vars if tv.get("Name") and tv["Name"] in df.columns]
    if df.empty or not cvars or not targets:
        st.info("데이터·공정 변수·목표 지표가 있어야 박스플롯을 그립니다.")
        return

    gvar = st.selectbox("그룹 기준 (X축) 공정 변수", cvars, key=f"{key_prefix}_gvar")
    st.caption(f"같은 '{gvar}' 값끼리 묶어 분포를 봅니다. 빨간 점 = 현재 방법"
               f"({method_label(method)})이 이상치로 판정한 값.")

    for tv in targets:
        tn = tv["Name"]
        unit = f" ({tv['Unit']})" if tv.get("Unit") else ""
        y_all = pd.to_numeric(df[tn], errors="coerce")
        gser = df[gvar].astype(str)

        fig = go.Figure()
        fig.add_trace(go.Box(x=gser, y=y_all, name=tn, boxpoints=False,
                             line=dict(color="#1a1a1a"), fillcolor="rgba(237,84,43,0.12)"))
        # 그룹별로 이상치 마스크를 계산해 점을 겹쳐 찍는다
        for gval, g in df.groupby(gvar):
            yv = pd.to_numeric(g[tn], errors="coerce").to_numpy(dtype=float)
            m = outlier_mask(yv, method, alpha)
            colors = ["#FF0000" if o else "#333333" for o in m]
            fig.add_trace(go.Scatter(
                x=[str(gval)] * len(yv), y=yv, mode="markers",
                marker=dict(color=colors, size=8, line=dict(width=0)),
                showlegend=False, hovertemplate=f"{gvar}={gval}<br>{tn}: %{{y:.6g}}<extra></extra>",
            ))
        fig.update_layout(
            template="simple_white", plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Myriad Pro, Pretendard, sans-serif", color="black"),
            title=dict(text=f"{tn}{unit}", font=dict(size=16)),
            xaxis=dict(title=gvar, showline=True, linecolor="black", mirror=True, ticks="inside"),
            yaxis=dict(title=tn, showline=True, linecolor="black", mirror=True, ticks="inside"),
            showlegend=False, height=360, margin=dict(l=60, r=20, t=40, b=50),
        )
        st.plotly_chart(fig, use_container_width=True,
                        key=f"{key_prefix}_{tn}", config={"displaylogo": False})
