"""tab2 데이터베이스 관리: 검색·변수별 필터·전체선택/해제·일괄삭제·요약통계 + 넓은 편집표.

st.session_state.df_data 를 유일한 진실로 두고, 필터된 뷰에서 편집한 내용을 원본에
인덱스로 되써서 숨은(필터로 가려진) 행이 유실되지 않게 한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

EDITOR_HEIGHT = 560


def _distinct_values(series):
    vals = series.dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


def _render_summary(df, config_vars, target_vars, cfg_names):
    """요약 통계 + 목표별 '최적 조건'. 최고 raw 값은 이상치일 수 있으므로 학습 적용
    데이터(수동 제외분 반영) 기준으로, 목표 방향(Max/Min)에 맞는 최적 조건을 보여준다."""
    with st.expander("📊 요약 통계 · 목표별 최적 조건", expanded=False):
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
        for tv in target_vars:
            tn = tv.get("Name")
            if not tn or tn not in valid.columns:
                continue
            col = pd.to_numeric(valid[tn], errors="coerce")
            if col.dropna().empty:
                continue
            direction = tv.get("Direction", "Maximize")
            is_max = "Maximize" in direction
            idx = col.idxmax() if is_max else col.idxmin()
            best = col.loc[idx]
            unit = f" {tv['Unit']}" if tv.get("Unit") else ""
            cond = " · ".join(f"{n}={valid.loc[idx, n]}" for n in cfg_names)
            sample = valid.loc[idx, "샘플명"] if "샘플명" in valid.columns else ""
            arrow = "최대" if is_max else "최소"
            line = f"**🎯 {tn}** ({arrow}) 최적 **{best:.6g}{unit}**"
            if cond:
                line += f"  ·  조건: {cond}"
            if isinstance(sample, str) and sample.strip():
                line += f"  ·  샘플: {sample}"
            st.markdown(line)


def render_data_manager(config_vars, target_vars, passive_vars):
    df = st.session_state.df_data
    if df is None or df.empty:
        st.info("아직 입력된 데이터가 없습니다. '신규 실험 입력' 탭에서 데이터를 추가하세요.")
        return

    cfg_names = [v["Name"] for v in config_vars if v.get("Name") and v["Name"] in df.columns]

    _render_summary(df, config_vars, target_vars, cfg_names)

    # ---------- 검색 ----------
    query = st.text_input("🔍 검색", key="dm_search", placeholder="샘플명·값 부분일치로 행 좁히기",
                          label_visibility="collapsed")

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
    edited = st.data_editor(
        fdf, use_container_width=True, hide_index=True, height=table_h,
        column_config={"학습_적용": st.column_config.CheckboxColumn("학습 적용")},
    )
    # 편집 내용을 원본에 인덱스로 되쓰기 (필터로 가려진 행은 그대로 보존)
    if not edited.equals(fdf):
        df.loc[edited.index, edited.columns] = edited
        st.session_state.df_data = df
