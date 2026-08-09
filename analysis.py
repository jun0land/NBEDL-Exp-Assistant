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

# (key, 화면 라벨). 순서가 셀렉트박스 순서. "auto"는 데이터에 맞춰 아래 방법 중 하나를 자동 선택.
OUTLIER_METHODS = [
    ("auto", "자동 추천 (데이터에 맞춤)"),
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


def group_sizes(df, feature_cols):
    """완전히 같은 조건(모든 공정 변수 일치) 그룹들의 크기 리스트."""
    feats = [c for c in feature_cols if c in df.columns]
    if df is None or df.empty or not feats:
        return []
    return df.groupby(feats, dropna=False).size().tolist()


def recommend_method(df, feature_cols):
    """반복(완전 동일 조건) 그룹 크기에 맞춰 이상치 방법을 추천한다. (method_key, reason) 반환.

    공정 변수가 많을수록 조건이 잘게 쪼개져 반복 수가 작아지므로, 변수 개수를 간접적으로 반영하되
    실제 반복 그룹 크기를 직접 보고 정하는 게 정확하다.
      - 판정 가능한(n>=3) 그룹이 없으면 → 없음
      - 대표 반복 수 n<=4 (소표본): IQR은 거의 못 잡고 MAD는 동일값 있으면 실패 → ESD
      - n 5~9 (소·중): 유의수준으로 조절되는 ESD
      - n>=10 (충분): 사분위가 안정적인 IQR
    """
    sizes = group_sizes(df, feature_cols)
    judge = [s for s in sizes if s >= MIN_GROUP]
    if not judge:
        return "none", ("완전히 같은 조건을 3회 이상 반복한 그룹이 없어 통계적 이상치 판정이 불가합니다. "
                        "같은 조건을 3회 이상 반복하면 판정할 수 있습니다.")
    n_typ = int(np.median(judge))
    if n_typ <= 4:
        return "esd", (f"조건당 반복이 대개 {n_typ}회인 소표본입니다. 이 구간에선 IQR은 거의 못 잡고"
                       f"(삼반복이면 사분위가 튀는 값 쪽으로 끌려감), 동일값이 있으면 MAD도 0이 되어 실패합니다. "
                       f"소표본용 정식 검정인 ESD를 추천합니다.")
    if n_typ <= 9:
        return "esd", (f"조건당 반복이 대개 {n_typ}회입니다. 소·중 표본에 적합하고 유의수준(α)으로 민감도를 "
                       f"조절할 수 있는 ESD를 추천합니다.")
    return "iqr", (f"조건당 반복이 대개 {n_typ}회로 충분합니다. 사분위(박스플롯) 기준이 안정적으로 동작하는 "
                   f"구간이라 가장 널리 쓰는 IQR을 추천합니다.")


def resolve_method(method, df, feature_cols):
    """'auto'면 데이터 기반 추천 방법(concrete)으로 바꿔 반환. 그 외엔 그대로."""
    if method == "auto":
        return recommend_method(df, feature_cols)[0]
    return method


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


def render_method_controls(method, alpha, df, feature_cols, *, key_prefix="outlier"):
    """이상치 방법 + α 선택 위젯. 현재 데이터에 맞는 추천을 함께 보여준다.
    (method, alpha) 를 반환 — method 는 사용자가 고른 값('auto' 가능), 호출부가 세션/파일에 저장한다.
    실제 적용 방법은 resolve_method(method, df, feature_cols) 로 확정한다."""
    c1, c2 = st.columns([1.4, 1])
    idx = _METHOD_KEYS.index(method) if method in _METHOD_KEYS else 0
    sel = c1.selectbox("이상치 판정 방법", _METHOD_KEYS, index=idx,
                       format_func=method_label, key=f"{key_prefix}_method")

    rec, reason = recommend_method(df, feature_cols)
    effective = rec if sel == "auto" else sel   # α 컨트롤 노출 판단에 실제 적용 방법을 쓴다
    if sel == "auto":
        st.caption(f"🔍 **자동 추천 적용 → {method_label(rec)}**. {reason}")
    else:
        st.caption(f"💡 이 데이터엔 **{method_label(rec)}** 추천 — {reason}")

    out_alpha = float(alpha)
    if method_uses_alpha(effective):
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
# 이상치 추천 & 사용자 검토 (자동 제거 대신)
# ---------------------------------------------------------------------------
def recommended_outlier_rows(df, feature_cols, target_cols, method, alpha):
    """완전히 같은 조건(모든 공정 변수 일치) 그룹별로 각 목표에서 이상치로 '추천'되는 행을 찾는다.
    학습_적용 상태와 무관하게 전체 데이터로 판정(추천이 안정적이도록). {row_index: [목표명,...]} 반환."""
    reasons = {}
    if df is None or df.empty:
        return reasons
    feats = [c for c in feature_cols if c in df.columns]
    if not feats:
        return reasons
    for _, g in df.groupby(feats, dropna=False):
        for t in target_cols:
            if t not in g.columns:
                continue
            y = pd.to_numeric(g[t], errors="coerce")
            m = outlier_mask(y.tolist(), method, alpha)
            for idx, flag in zip(g.index, m):
                if flag:
                    reasons.setdefault(idx, []).append(t)
    return reasons


def render_outlier_review(feature_cols, target_cols, method, alpha, *, key_prefix="orev"):
    """이상치를 자동 제거하지 않고 '추천'만 한 뒤, 사용자가 학습 제외 여부를 직접 정하게 한다.
    st.session_state.df_data 의 '학습_적용' 을 직접 토글한다(기본은 모두 포함)."""
    df = st.session_state.df_data
    if df is None or df.empty:
        st.info("데이터가 없습니다.")
        return
    reasons = recommended_outlier_rows(df, feature_cols, target_cols, method, alpha)
    st.caption(
        f"현재 방법({method_label(method)})이 '완전히 같은 조건' 그룹 안에서 이상치로 **추천**한 데이터입니다. "
        "알고리즘만으로 이상치를 100% 확신할 수 없으니, 실제로 학습에서 뺄지는 직접 정하세요. "
        "**'학습 적용' 체크를 해제하면 그 행이 AI 학습에서 제외**됩니다. 기본은 모두 포함입니다.")
    if not reasons:
        st.success("추천되는 이상치가 없습니다. (완전히 같은 조건을 3회 이상 반복한 그룹에서만 판정)")
        return

    idxs = [i for i in df.index if i in reasons]
    excluded_now = int((df.loc[idxs, "학습_적용"] != True).sum()) if "학습_적용" in df.columns else 0  # noqa: E712
    st.caption(f"추천 이상치 **{len(idxs)}건** · 현재 제외됨 {excluded_now}건")

    b1, b2 = st.columns(2)
    if b1.button("☐ 추천 전부 제외 (학습에서 빼기)", key=f"{key_prefix}_excl", use_container_width=True):
        st.session_state.df_data.loc[idxs, "학습_적용"] = False
        st.rerun()
    if b2.button("☑ 추천 전부 포함 (되돌리기)", key=f"{key_prefix}_incl", use_container_width=True):
        st.session_state.df_data.loc[idxs, "학습_적용"] = True
        st.rerun()

    view = df.loc[idxs].copy()
    view["이상치 사유"] = [", ".join(reasons[i]) for i in idxs]
    show = (["학습_적용"] if "학습_적용" in view.columns else []) + ["이상치 사유"]
    if "샘플명" in view.columns:
        show.append("샘플명")
    show += [c for c in feature_cols if c in view.columns]
    flagged = [t for t in target_cols if any(t in reasons[i] for i in idxs)]
    show += [c for c in flagged if c in view.columns]
    disabled = [c for c in show if c != "학습_적용"]
    edited = st.data_editor(
        view[show], hide_index=True, use_container_width=True, disabled=disabled,
        column_config={"학습_적용": st.column_config.CheckboxColumn("학습 적용")},
        key=f"{key_prefix}_editor")
    if "학습_적용" in edited.columns:
        st.session_state.df_data.loc[edited.index, "학습_적용"] = edited["학습_적용"].values


def render_applied_exclusions(feature_cols, target_cols, *, key_prefix="applied"):
    """실제로 학습에서 제외된(학습_적용==False) 데이터 목록을 아코디언(익스팬더)으로 보여준다.
    데이터 진단에서 정한 제외가 AI 계산에 반영됐는지 확인하는 용도."""
    df = st.session_state.df_data
    if df is None or df.empty or "학습_적용" not in df.columns:
        return
    excl = df[df["학습_적용"] != True]  # noqa: E712
    with st.expander(f"🚫 학습에서 제외하고 계산한 데이터 {len(excl)}건", expanded=False):
        if excl.empty:
            st.caption("제외된 데이터가 없습니다. 전체 유효 데이터로 계산합니다.")
            return
        st.caption("데이터베이스 관리에서 직접 끄거나, 데이터 진단의 '이상치 검토'에서 제외한 행입니다. "
                   "이 목록이 실제 AI 계산에서 빠졌습니다.")
        show = (["샘플명"] if "샘플명" in excl.columns else [])
        show += [c for c in feature_cols if c in excl.columns]
        show += [c for c in target_cols if c in excl.columns]
        st.dataframe(excl[show] if show else excl, use_container_width=True,
                     hide_index=True, key=f"{key_prefix}_df")


# ---------------------------------------------------------------------------
# 박스플롯 (반복 측정 분포 + 이상치)
# ---------------------------------------------------------------------------
def _cond_label(name, n):
    """조건 그룹 라벨: 모든 공정변수 값 조합 + 반복 수 n."""
    vals = name if isinstance(name, tuple) else (name,)
    return "·".join(str(v) for v in vals) + f" (n={n})"


def render_boxplots(df, config_vars, target_vars, method, alpha, *, key_prefix="box"):
    """완전히 같은 조건(모든 공정 변수 일치)끼리 묶어 목표별 분포를 박스플롯으로 그린다.
    각 박스 = 한 조건의 반복값들. 빨간 점 = 현재 방법이 그 조건 안에서 이상치로 판정한 값
    (= AI 학습에서 실제로 제외되는 값)이라, 박스플롯이 AI가 보는 그대로가 된다.

    이상치 판정은 항상 '완전 동일 조건' 그룹 단위 — X축을 변수 하나로만 묶던 이전 방식과 달리
    조건을 섞지 않는다."""
    cvars = [v["Name"] for v in config_vars if v.get("Name") and v["Name"] in df.columns]
    targets = [tv for tv in target_vars if tv.get("Name") and tv["Name"] in df.columns]
    if df.empty or not cvars or not targets:
        st.info("데이터·공정 변수·목표 지표가 있어야 박스플롯을 그립니다.")
        return

    tnames = [tv["Name"] for tv in targets]
    sel_targets = st.multiselect(
        "표시할 목표 지표", tnames, default=tnames[:min(3, len(tnames))],
        key=f"{key_prefix}_targets",
        help="목표가 많으면 보고 싶은 것만 고르세요. 한 번에 너무 많이 그리면 느려집니다.")
    if not sel_targets:
        st.info("표시할 목표 지표를 하나 이상 선택하세요.")
        return

    st.caption(f"'완전히 같은 조건(모든 공정 변수 일치)'끼리 묶은 박스입니다 · "
               f"빨간 점 = 현재 방법({method_label(method)})이 이상치로 **추천**한 값"
               f"(실제 제외는 '이상치 검토'에서 직접 결정) · n = 그 조건의 반복 수(3 미만은 판정 불가).")

    grouped = list(df.groupby(cvars, dropna=False))

    for tn in sel_targets:
        tvu = next((t for t in targets if t["Name"] == tn), {})
        unit = f" ({tvu.get('Unit')})" if tvu.get("Unit") else ""
        fig = go.Figure()
        n_cond = 0
        for name, g in grouped:
            yv = pd.to_numeric(g[tn], errors="coerce").to_numpy(dtype=float)
            yv = yv[np.isfinite(yv)]
            if len(yv) == 0:
                continue
            n_cond += 1
            lab = _cond_label(name, len(yv))
            m = outlier_mask(yv, method, alpha)
            colors = ["#FF0000" if o else "#333333" for o in m]
            fig.add_trace(go.Box(x=[lab] * len(yv), y=yv, name=lab, boxpoints=False,
                                 line=dict(color="#1a1a1a"), fillcolor="rgba(237,84,43,0.10)",
                                 showlegend=False))
            fig.add_trace(go.Scatter(
                x=[lab] * len(yv), y=yv, mode="markers",
                marker=dict(color=colors, size=8, line=dict(width=0)),
                showlegend=False, hovertemplate=f"{lab}<br>{tn}: %{{y:.6g}}<extra></extra>"))
        if n_cond == 0:
            continue
        fig.update_layout(
            template="simple_white", plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Myriad Pro, Pretendard, sans-serif", color="black"),
            title=dict(text=f"{tn}{unit}  ·  조건 {n_cond}개", font=dict(size=15)),
            xaxis=dict(title="공정 조건 (모든 변수 일치)", showline=True, linecolor="black",
                       mirror=True, ticks="inside", tickangle=-40, automargin=True),
            yaxis=dict(title=tn, showline=True, linecolor="black", mirror=True, ticks="inside"),
            showlegend=False, height=420, margin=dict(l=60, r=20, t=44, b=90),
        )
        st.plotly_chart(fig, use_container_width=True,
                        key=f"{key_prefix}_{tn}", config={"displaylogo": False})
