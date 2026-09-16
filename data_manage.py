"""tab2 데이터베이스 관리: 검색·변수별 필터·전체선택/해제·일괄삭제·요약통계 + 넓은 편집표.

st.session_state.df_data 를 유일한 진실로 두고, 필터된 뷰에서 편집한 내용을 원본에
인덱스로 되써서 숨은(필터로 가려진) 행이 유실되지 않게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analysis import desirability, direction_arrow

EDITOR_HEIGHT = 560


def _distinct_values(series):
    vals = series.dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


def render_target_selectors(target_vars, key_prefix, per_row=2):
    """목표별 [체크박스=포함][좁은 가중치칸]을 컴팩트 그리드로 렌더한다(한 줄에 per_row개).
    체크 해제 시 그 목표는 제외되고 가중치칸은 비활성. (선택된 이름 리스트, {이름:가중치}) 반환.
    요약 통계·파레토 후보가 같은 모양을 쓰도록 공용."""
    tvs = [tv for tv in target_vars if tv.get("Name")]
    incl, weights = {}, {}

    def _arrow(tv):
        d = tv.get("Direction", "Maximize")
        return "↑" if "Maximize" in d else ("↓" if "Minimize" in d else "◎")

    for i in range(0, len(tvs), per_row):
        row = tvs[i:i + per_row]
        specs = []
        for _ in row:
            specs += [1.7, 0.55]
        cols = st.columns(specs, vertical_alignment="center")
        for j, tv in enumerate(row):
            name = tv["Name"]
            incl[name] = cols[2 * j].checkbox(f"{name} {_arrow(tv)}", value=True, key=f"{key_prefix}_incl_{name}")
            weights[name] = cols[2 * j + 1].number_input(
                "w", min_value=0.0, value=1.0, step=0.1, key=f"{key_prefix}_w_{name}",
                label_visibility="collapsed", disabled=not incl[name])
    sel = [tv["Name"] for tv in tvs if incl.get(tv["Name"])]
    return sel, weights


def _render_summary(df, config_vars, target_vars, cfg_names):
    """요약 통계 + 목표별 '최적 조건'. 최고 raw 값은 이상치일 수 있으므로 학습 적용
    데이터(수동 제외분 반영) 기준으로, 목표 방향(Max/Min)에 맞는 최적 조건을 보여준다."""
    with st.expander("📊 요약 통계 · 종합 최적 조건", expanded=False):
        valid = df[df["학습_적용"] == True] if "학습_적용" in df.columns else df  # noqa: E712
        st.caption(f"학습 적용 {len(valid)}행 / 전체 {len(df)}행 기준")

        if cfg_names:
            rows = []
            for n in cfg_names:
                s = pd.to_numeric(valid[n], errors="coerce").dropna()
                if len(s):
                    rows.append({"공정 변수": n, "min": s.min(), "max": s.max(),
                                 "평균": round(float(s.mean()), 4)})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if valid.empty:
            st.caption("학습 적용된 데이터가 없어 최적 조건을 계산할 수 없습니다.")
            return

        tgts = [tv for tv in target_vars
                if tv.get("Name") and tv["Name"] in valid.columns
                and pd.to_numeric(valid[tv["Name"]], errors="coerce").notna().any()]
        if not tgts:
            st.caption("유효한 목표 지표 데이터가 없어 최적 조건을 계산할 수 없습니다.")
            return

        # 여러 목표의 trade-off 를 함께 고려한 '종합 최적 조건' 하나를 고른다.
        # 각 목표를 방향(최대화/최소화)에 맞춰 0~1 로 정규화(최선=1)한 desirability 점수를 만들고,
        # 목표 점수의 평균이 가장 높은 실험 조건을 뽑는다 — 한 목표만 뛰어나고 나머지가 나쁜 조건은
        # 평균이 낮아 밀리고, 전체적으로 균형 잡힌(파레토상 타협점에 가까운) 조건이 선택된다.
        score_df = pd.DataFrame(index=valid.index)
        for tv in tgts:
            col = pd.to_numeric(valid[tv["Name"]], errors="coerce")
            # 방향별 desirability (최대화=클수록, 최소화=작을수록, 특정값=목표값에 가까울수록 1)
            score_df[tv["Name"]] = desirability(col, tv)

        # 목표별 '포함 여부 + 가중치'를 한 줄에: [체크박스=이름/방향] [가중치칸]. 체크 해제한 목표는
        # 종합 최적 조건 계산에서 빠지고, 그 목표의 가중치칸은 비활성화된다.
        st.caption("종합 최적 조건에 포함할 목표와 가중치 (체크 해제 시 제외 · 기본 가중치 1)")
        sel_names, weights = render_target_selectors(tgts, "sum")

        sel = [tv for tv in tgts if tv["Name"] in sel_names]
        if not sel:
            st.caption("최소 1개 이상의 목표를 선택하세요.")
            return

        # 가중 평균 desirability (선택 목표만). 행마다 값이 있는 목표들만으로 정규화(NaN 목표는 건너뜀).
        cols = [tv["Name"] for tv in sel]
        w = np.array([weights[c] for c in cols], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(cols))  # 전부 0 이면 동일 가중으로 폴백
        S = score_df[cols].to_numpy(dtype=float)
        present = ~np.isnan(S)
        wsum = np.where(present, w[np.newaxis, :], 0.0).sum(axis=1)
        num = np.where(present, np.nan_to_num(S) * w[np.newaxis, :], 0.0).sum(axis=1)
        comp_vals = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)
        composite = pd.Series(comp_vals, index=valid.index)
        if composite.dropna().empty:
            st.caption("종합 점수를 계산할 수 없습니다.")
            return
        best_idx = composite.idxmax()
        custom_w = any(abs(weights[c] - weights[cols[0]]) > 1e-9 for c in cols)

        cond = " · ".join(f"{n}={valid.loc[best_idx, n]}" for n in cfg_names)
        sample = valid.loc[best_idx, "샘플명"] if "샘플명" in valid.columns else ""
        excl_note = f" (제외 {len(tgts) - len(sel)}개)" if len(sel) < len(tgts) else ""
        target_names = ", ".join(
            f"{tv['Name']}({direction_arrow(tv)}"
            + (f"×{weights[tv['Name']]:g}" if custom_w else "") + ")" for tv in sel)
        w_note = " · 가중치 적용됨" if custom_w else ""
        st.markdown(f"**🎯 종합 최적 조건** — 선택 목표{excl_note}({target_names})의 방향·trade-off 를 함께 고려한 균형점{w_note}")
        if cond:
            line = f"- 조건: **{cond}**"
            if isinstance(sample, str) and sample.strip():
                line += f"  ·  샘플: {sample}"
            st.markdown(line)
        vals = []
        for tv in sel:
            v = pd.to_numeric(valid[tv["Name"]], errors="coerce").loc[best_idx]
            unit = f" {tv['Unit']}" if tv.get("Unit") else ""
            vals.append(f"{tv['Name']}{direction_arrow(tv)} {v:.6g}{unit}")
        st.markdown(f"- 그 조건의 목표값: {' · '.join(vals)}  ·  종합점수 **{composite.loc[best_idx]:.3f}** / 1")
        st.caption("각 목표를 방향에 맞춰 0~1(최선=1)로 정규화한 desirability 의 평균이 최대인 실험 조건입니다. "
                   "여러 목표가 상충할 때 한쪽으로 치우치지 않은 균형 조건을 고릅니다.")


def render_summary(config_vars, target_vars):
    """요약 통계 · 종합 최적 조건만 따로 렌더한다(데이터 진단 탭에서 호출). 세션의 df 를 읽는다."""
    df = st.session_state.df_data
    if df is None or df.empty:
        st.info("아직 입력된 데이터가 없습니다. '신규 실험 입력' 탭에서 데이터를 추가하세요.")
        return
    cfg_names = [v["Name"] for v in config_vars if v.get("Name") and v["Name"] in df.columns]
    _render_summary(df, config_vars, target_vars, cfg_names)


def render_data_manager(config_vars, target_vars, passive_vars):
    df = st.session_state.df_data
    if df is None or df.empty:
        st.info("아직 입력된 데이터가 없습니다. '신규 실험 입력' 탭에서 데이터를 추가하세요.")
        return

    cfg_names = [v["Name"] for v in config_vars if v.get("Name") and v["Name"] in df.columns]

    # ---------- 검색 ----------
    query = st.text_input(
        "🔍 표에서 검색",
        key="dm_search",
        placeholder="예: A-12 (샘플명) · 500 (값) · Toluene (옵션) — 입력한 글자가 든 행만 남습니다",
        help="표의 어느 칸에든(샘플명·공정 조건값·결과값) 입력한 글자가 들어간 행만 걸러 보여줍니다. "
             "일부만 입력해도 되고, 비워 두면 전체가 표시됩니다. 특정 값을 빠르게 찾을 때 쓰세요.",
    )

    # ---------- 변수별 멀티셀렉트 필터 ----------
    filters = {}
    if cfg_names:
        with st.expander("🔎 공정 변수별 필터", expanded=bool(any(
                st.session_state.get(f"dm_filt_{n}") for n in cfg_names))):
            fcols = st.columns(min(len(cfg_names), 4))
            for i, n in enumerate(cfg_names):
                sel = fcols[i % len(fcols)].multiselect(n, _distinct_values(df[n]), key=f"dm_filt_{n}")
                if sel:
                    filters[n] = set(sel)

    # ---------- 마스크 ----------
    mask = pd.Series(True, index=df.index)
    if query and query.strip():
        q = query.strip().lower()
        mask &= df.apply(lambda row: q in " ".join(str(x) for x in row.values).lower(), axis=1)
    for n, sel in filters.items():
        mask &= df[n].isin(sel)
    filtered_idx = df.index[mask]
    fdf = df.loc[filtered_idx]

    # ---------- 액션 버튼 ----------
    b1, b2, b3, b4 = st.columns([1.3, 1.1, 0.8, 1.1])
    if b1.button("☑ 보이는 행 전체 학습 적용", use_container_width=True):
        st.session_state.df_data.loc[filtered_idx, "학습_적용"] = True
        st.rerun()
    if b2.button("☐ 보이는 행 전체 해제", use_container_width=True):
        st.session_state.df_data.loc[filtered_idx, "학습_적용"] = False
        st.rerun()
    confirm_del = b3.checkbox("삭제 확인", key="dm_del_confirm",
                             help="실수 방지: 체크해야 삭제 버튼이 활성화됩니다.")
    if b4.button("🗑 보이는 행 삭제", use_container_width=True, disabled=not confirm_del):
        st.session_state.df_data = df.drop(index=filtered_idx).reset_index(drop=True)
        st.session_state.dm_del_confirm = False
        st.rerun()

    note = "  (필터 적용됨)" if len(fdf) != len(df) else ""
    c_note, c_h = st.columns([2, 1], vertical_alignment="center")
    c_note.caption(f"표시 {len(fdf)}행 / 전체 {len(df)}행{note}")
    # 표 높이는 화면 크기에 따라 조절 — 노트북은 기본값(560), 큰 모니터(4K 등)는 높여서 활용.
    table_h = c_h.slider("표 높이(px)", 300, 1400, EDITOR_HEIGHT, 20, key="dm_height",
                        help="큰 모니터에서는 높여서 화면을 더 활용하세요.")

    # ---------- 편집표 (넓게) ----------
    # st.form 으로 감싼다. 폼 안에서는 위젯을 건드려도 리런이 일어나지 않으므로, 체크박스를
    # 여러 개 바꾼 뒤 '적용'을 한 번만 누르면 된다. (폼이 없으면 체크 한 번마다 전체 앱이
    # 다시 그려져 — 분석 탭의 무거운 계산까지 — 한참 기다리게 된다.)
    st.caption("체크를 여러 개 바꾼 뒤 아래 **적용** 버튼을 한 번만 누르세요. 누르기 전에는 계산에 반영되지 않습니다.")
    with st.form("dm_editor_form", border=False):
        edited = st.data_editor(
            fdf, use_container_width=True, hide_index=True, height=table_h,
            column_config={"학습_적용": st.column_config.CheckboxColumn("학습 적용")},
        )
        applied = st.form_submit_button("✅ 변경 사항 적용", type="primary", use_container_width=True)

    # 편집 내용을 원본에 인덱스로 되쓰기 (필터로 가려진 행은 그대로 보존)
    if applied:
        if not edited.equals(fdf):
            df.loc[edited.index, edited.columns] = edited
            st.session_state.df_data = df
            st.rerun()
        else:
            st.info("바뀐 내용이 없습니다.")
