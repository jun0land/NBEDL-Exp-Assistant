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

# ------------------------------------------------------------------
# 목표 지표의 최적화 방향 (app.py · data_manage.py 공용)
#   Maximize / Minimize 외에 "Target"(특정 값에 맞추기)을 지원한다.
#   예: 잔류 응력 0에 맞추기, 두께를 정확히 300nm로 맞추기.
#   내부적으로 Target은 "목표값과의 거리를 최소화"로 환산해 처리한다.
# ------------------------------------------------------------------
DIRECTION_OPTIONS = ["Maximize", "Minimize", "Target"]
DIRECTION_LABELS = {          # 저장값은 영문 유지(엑셀 호환), 화면 표시만 한글
    "Maximize": "최대화",
    "Minimize": "최소화",
    "Target": "특정값 맞추기",
}


def target_direction(tv):
    """저장된 Direction 값을 3종 중 하나로 정규화한다.
    구버전 파일이나 오염된 값은 Maximize로 폴백."""
    d = str(tv.get("Direction", "Maximize"))
    for opt in DIRECTION_OPTIONS:
        if opt in d:
            return opt
    return "Maximize"


def target_value_of(tv):
    """Target 모드에서 맞출 목표값. 미지정/파싱 실패 시 0.0."""
    try:
        v = float(tv.get("Target_Value", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(v) else v


def direction_label(tv):
    """화면 표시용 방향 문자열. Target이면 맞출 값까지 같이 보여준다."""
    d = target_direction(tv)
    return f"Target={target_value_of(tv):g}" if d == "Target" else d


def direction_arrow(tv):
    """요약표용 짧은 방향 기호."""
    d = target_direction(tv)
    return {"Maximize": "↑", "Minimize": "↓", "Target": "◎"}[d]


def best_so_far(series, direction, target_val=0.0):
    """수렴 곡선용 '현재까지의 최선값' 누적 시리즈.
    Maximize/Minimize는 누적 최대/최소, Target은 목표값에 가장 가까운 값을 추적한다."""
    s = pd.to_numeric(series, errors="coerce")
    if direction == "Maximize":
        return s.expanding().max()
    if direction == "Minimize":
        return s.expanding().min()
    best, out = None, []
    for v in s:
        if pd.notna(v) and (best is None or abs(v - target_val) < abs(best - target_val)):
            best = v
        out.append(best)
    return pd.Series(out, index=s.index)


def desirability(col, tv):
    """목표 하나를 0~1(최선=1)로 정규화한 desirability 점수.
    Maximize는 클수록, Minimize는 작을수록, Target은 목표값에 가까울수록 1에 가깝다."""
    d = target_direction(tv)
    if d == "Target":
        dev = (col - target_value_of(tv)).abs()
        dmax = dev.max()
        if not np.isfinite(dmax) or dmax == 0:
            return pd.Series(1.0, index=col.index)  # 전부 목표값과 같은 거리(=완전 일치) → 최선
        return 1.0 - dev / dmax
    lo, hi = col.min(), col.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=col.index)      # 상수 → 중립
    norm = (col - lo) / (hi - lo)
    return 1.0 - norm if d == "Minimize" else norm


def desirability_scored(col, tv, floor=None, pad=0.1):
    """종합점수 전용 desirability. 그래프 정규화용 desirability() 와 두 가지가 다르다.

    1) **0점 기준을 데이터 최솟값에 두지 않는다.** min-max 로 펴면 꼴찌 조건은 실제
       차이가 5 % 든 500 % 든 무조건 0점이 된다. 0점이 하나라도 있으면 기하평균이
       통째로 0이 되어 순위를 매길 수 없다. 그래서 '최소 허용값'이 있으면 그것을,
       없으면 관측 범위를 pad 만큼 넓힌 지점을 0점으로 삼는다. 이는 하이퍼볼륨
       기준점(ref_point)을 잡는 방식과 같은 철학이다.
    2) **허용값 밖은 0 으로 자른다.** 최소 허용값에 미달한 값이 음수 점수로 남아
       다른 목표의 점수를 갉아먹지 않게 한다.
    """
    d = target_direction(tv)
    col = pd.to_numeric(col, errors="coerce")
    lo, hi = col.min(), col.max()
    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(np.nan, index=col.index)
    rng = hi - lo
    if rng == 0:
        rng = abs(hi) if hi != 0 else 1.0
    if d == "Target":
        dev = (col - target_value_of(tv)).abs()
        dmax = dev.max()
        if not np.isfinite(dmax) or dmax == 0:
            return pd.Series(1.0, index=col.index)
        return (1.0 - dev / (dmax * (1.0 + pad))).clip(lower=0.0, upper=1.0)
    if d == "Maximize":
        zero = float(floor) if floor is not None else (lo - pad * rng)
        top = max(hi, zero + 1e-12)
        return ((col - zero) / (top - zero)).clip(lower=0.0, upper=1.0)
    zero = float(floor) if floor is not None else (hi + pad * rng)   # Minimize: 이 값이 0점
    bottom = min(lo, zero - 1e-12)
    return ((zero - col) / (zero - bottom)).clip(lower=0.0, upper=1.0)


# 종합점수를 만드는 방식. 여러 목표를 하나의 수로 합칠 때 무엇을 '좋다'고 볼지 정한다.
COMPOSITE_MODES = [
    ("geometric", "기하평균 (권장 · 모든 목표가 함께 성립해야 함)"),
    ("arithmetic", "산술평균 (관대 · 한 목표의 우수함이 다른 목표의 부진을 상쇄)"),
    ("minimum", "최솟값 (가장 엄격 · 가장 부진한 목표가 곧 점수)"),
]
_COMPOSITE_LABEL = dict(COMPOSITE_MODES)


def composite_label(mode):
    return _COMPOSITE_LABEL.get(mode, mode)


def composite_score(S, w, mode="geometric", eps=0.01):
    """조건 x 목표 desirability 행렬 S 와 가중치 w 를 하나의 종합점수로 합친다.

    산술평균은 목표들이 서로를 대신할 수 있다고 가정한다. 그래서 한 모드에서 1위를
    쓸어 담고 다른 모드에서 꼴찌인 조건이, 양쪽에서 2위인 조건을 이긴다. 두 모드가
    **함께** 성립해야 하는 문제에서는 이 가정이 틀렸다. 기하평균은 한 목표가 0에
    가까우면 전체가 0에 가까워지므로 상충을 실제로 벌준다(Derringer-Suich 의 원래
    정의도 기하평균이다). NaN 목표는 그 행의 계산에서 빠진다.
    """
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    present = ~np.isnan(S)
    wm = np.where(present, w[np.newaxis, :], 0.0)
    wsum = wm.sum(axis=1)
    if mode == "minimum":
        masked = np.where(present, S, np.inf)
        out = masked.min(axis=1)
        return np.where(np.isfinite(out), out, np.nan)
    if mode == "geometric":
        Sc = np.where(present, np.clip(S, eps, 1.0), 1.0)
        ln = np.where(present, np.log(Sc), 0.0)
        return np.where(wsum > 0, np.exp((ln * wm).sum(axis=1) / np.where(wsum > 0, wsum, 1.0)), np.nan)
    num = np.where(present, np.nan_to_num(S) * wm, 0.0).sum(axis=1)
    return np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)


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

    # 기본 선택은 등록 순서가 아니라 '이상치가 실제로 있는 목표'부터 채운다 — 안 그러면
    # 처음 3개가 우연히 이상치 없는 목표일 때 정작 봐야 할 목표가 안 보인다.
    grouped_for_count = list(df.groupby(cvars, dropna=False))

    def _outlier_count(tn):
        total = 0
        for _, g in grouped_for_count:
            yv = pd.to_numeric(g[tn], errors="coerce").to_numpy(dtype=float)
            yv = yv[np.isfinite(yv)]
            if len(yv):
                total += int(outlier_mask(yv, method, alpha).sum())
        return total

    outlier_counts = {tn: _outlier_count(tn) for tn in tnames}
    default_order = sorted(tnames, key=lambda tn: (-outlier_counts[tn], tnames.index(tn)))
    default_sel = default_order[:min(3, len(tnames))]

    sel_targets = st.multiselect(
        "표시할 목표 지표", tnames, default=default_sel,
        key=f"{key_prefix}_targets",
        help="목표가 많으면 보고 싶은 것만 고르세요. 기본값은 현재 방법으로 이상치가 "
             "많이 잡힌 목표부터 우선 채웁니다(한 번에 너무 많이 그리면 느려집니다).")
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
