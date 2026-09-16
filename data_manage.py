"""tab2 데이터베이스 관리: 검색·변수별 필터·전체선택/해제·일괄삭제·요약통계 + 넓은 편집표.

st.session_state.df_data 를 유일한 진실로 두고, 필터된 뷰에서 편집한 내용을 원본에
인덱스로 되써서 숨은(필터로 가려진) 행이 유실되지 않게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analysis import (desirability, desirability_scored, direction_arrow, direction_label,
                      target_direction, COMPOSITE_MODES, composite_score)

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
    """공정 변수별 범위·평균 요약. 종합 최적 조건은 AI 계산 탭의
    render_composite_optimum 으로 옮겨 갔다(조건 단위 평가 + 최소 허용값 적용)."""
    with st.expander("📊 요약 통계", expanded=False):
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

        st.caption(
            "**종합 최적 조건**은 이제 **AI 계산 탭 아래쪽**에서 보여줍니다. "
            "거기서는 시료 하나를 집어 주는 대신 같은 공정 조건의 반복 시료를 묶어 **조건 자체**를 평가하고, "
            "AI 계산에 넣은 **최소 허용값**을 그대로 적용합니다."
        )


def render_summary(config_vars, target_vars):
    """요약 통계만 따로 렌더한다(데이터 진단 탭에서 호출). 세션의 df 를 읽는다."""
    df = st.session_state.df_data
    if df is None or df.empty:
        st.info("아직 입력된 데이터가 없습니다. '신규 실험 입력' 탭에서 데이터를 추가하세요.")
        return
    cfg_names = [v["Name"] for v in config_vars if v.get("Name") and v["Name"] in df.columns]
    _render_summary(df, config_vars, target_vars, cfg_names)



# ---------------------------------------------------------------------------
# 조건 단위 '종합 최적 조건'
# ---------------------------------------------------------------------------
# 시료 한 개를 집어 추천하면 그 시료의 우연한 편차까지 추천에 섞인다. 공정에서 실제로
# 고를 수 있는 것은 '조건'이지 '그 시료'가 아니므로, 같은 공정 조건의 반복 시료를 하나로
# 묶어 조건 자체를 평가한다. 대표값은 이상치 한 개에 흔들리지 않도록 중앙값을 쓴다.

def _condition_table(valid, cfg_names, sel_tvs):
    """같은 공정 조건끼리 묶어 목표별 중앙값과 반복 수(n)를 담은 표를 만든다."""
    names = [tv["Name"] for tv in sel_tvs]
    work = valid.copy()
    for n in names:
        work[n] = pd.to_numeric(work[n], errors="coerce")
    g = work.groupby(cfg_names, dropna=False, sort=True)
    med = g[names].median()
    cnt = g[names].count()
    tbl = med.copy()
    tbl["n"] = g.size()
    tbl["n_min"] = cnt.min(axis=1)
    return tbl.reset_index()


def _floor_verdict(tbl, sel_tvs, floors):
    """조건별 '최소 허용값' 충족 여부와, 미달한 목표 이름들을 돌려준다.
    최대화 목표는 이 값 이상이어야 하고, 최소화 목표는 이 값 이하여야 한다."""
    ok = pd.Series(True, index=tbl.index)
    why = pd.Series([[] for _ in range(len(tbl))], index=tbl.index)
    for tv in sel_tvs:
        f = floors.get(tv["Name"])
        if f is None:
            continue
        d = target_direction(tv)
        col = pd.to_numeric(tbl[tv["Name"]], errors="coerce")
        if d == "Maximize":
            bad = (col < f)
        elif d == "Minimize":
            bad = (col > f)
        else:
            continue
        bad = bad.fillna(False)
        ok &= ~bad
        for i in tbl.index[bad]:
            why.loc[i] = why.loc[i] + [tv["Name"]]
    return ok, why


def render_composite_optimum(config_vars, target_vars, key_prefix="mobo", floors=None):
    """지금까지 쌓인 데이터만으로 고른 '종합 최적 조건'을 조건 단위로 보여준다.

    AI 후보가 '다음에 해볼 만한 미지의 지점'이라면, 이쪽은 '이미 실험해 본 것 중 현재
    가장 균형이 좋은 조건'이다. 둘을 나란히 두면 후보가 지금 최선보다 나아질 여지가
    있는지 가늠할 수 있어 AI 결과 옆에 둔다. 목표 선택·가중치·최소 허용값은 AI 계산에
    쓴 것(key_prefix 위젯)을 그대로 읽어 같은 기준으로 평가한다."""
    df = st.session_state.get("df_data")
    if df is None or df.empty:
        return
    cfg_names = [v["Name"] for v in config_vars if v.get("Name") and v["Name"] in df.columns]
    if not cfg_names:
        return
    valid = df[df["학습_적용"] == True] if "학습_적용" in df.columns else df  # noqa: E712
    if valid.empty:
        return

    # AI 계산과 동일한 목표 선택·가중치를 세션 위젯에서 읽는다(탭을 안 열었으면 기본값).
    sel_tvs, weights = [], {}
    for tv in target_vars:
        name = tv.get("Name")
        if not name or name not in valid.columns:
            continue
        if not pd.to_numeric(valid[name], errors="coerce").notna().any():
            continue
        if not st.session_state.get(f"{key_prefix}_incl_{name}", True):
            continue
        sel_tvs.append(tv)
        try:
            weights[name] = float(st.session_state.get(f"{key_prefix}_w_{name}", 1.0))
        except (TypeError, ValueError):
            weights[name] = 1.0
    if not sel_tvs:
        return

    if floors is None:
        floors = {}
        for tv in sel_tvs:
            raw = str(st.session_state.get(f"mobo_floor_{tv['Name']}", "") or "").strip()
            if raw:
                try:
                    floors[tv["Name"]] = float(raw)
                except ValueError:
                    pass

    tbl = _condition_table(valid, cfg_names, sel_tvs)
    if tbl.empty:
        return
    ok, why = _floor_verdict(tbl, sel_tvs, floors)
    passed = tbl[ok]
    if passed.empty:
        st.warning(
            "최소 허용값을 모든 목표에서 동시에 넘는 **조건**이 하나도 없습니다. "
            "값을 낮추거나 일부를 비워 두세요."
        )
        return

    # 종합 방식: 목표들을 하나의 수로 어떻게 합칠지. 산술평균은 한 목표의 우수함이 다른
    # 목표의 부진을 상쇄하므로, 두 모드가 함께 성립해야 하는 문제에서는 상충을 벌주지
    # 못한다. 기본값을 기하평균으로 둔다.
    mode = st.radio(
        "종합 방식", [m[0] for m in COMPOSITE_MODES],
        format_func=lambda k: dict(COMPOSITE_MODES)[k],
        key=f"{key_prefix}_copt_mode", horizontal=False,
        help="여러 목표를 하나의 점수로 합치는 방법입니다. 목표들이 서로를 대신할 수 있으면 "
             "산술평균이 맞지만, 모든 목표가 동시에 성립해야 하면 기하평균이나 최솟값이 맞습니다.")

    # desirability 는 통과한 조건들 사이에서만 0~1 로 매긴다 — 탈락한(사실상 죽은)
    # 조건이 척도의 양 끝을 차지해 살아 있는 조건들의 점수를 뭉개지 않게 하기 위함이다.
    # 0점 기준은 최소 허용값(있으면) 또는 관측 범위를 조금 넓힌 지점으로 잡는다.
    score = pd.DataFrame(index=passed.index)
    for tv in sel_tvs:
        score[tv["Name"]] = desirability_scored(
            passed[tv["Name"]], tv, floor=floors.get(tv["Name"]))
    cols = [tv["Name"] for tv in sel_tvs]
    w = np.array([weights[c] for c in cols], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(cols))
    comp = pd.Series(composite_score(score[cols].to_numpy(dtype=float), w, mode=mode),
                     index=passed.index)
    if comp.dropna().empty:
        return

    ranked = passed.copy()
    ranked["종합점수"] = comp
    ranked = ranked.sort_values("종합점수", ascending=False)
    best = ranked.iloc[0]

    custom_w = any(abs(weights[c] - weights[cols[0]]) > 1e-9 for c in cols)
    tgt_txt = ", ".join(
        f"{tv['Name']}({direction_arrow(tv)}" + (f"×{weights[tv['Name']]:g}" if custom_w else "") + ")"
        for tv in sel_tvs)
    sel_names = {tv["Name"] for tv in sel_tvs}
    excluded = [tv.get("Name") for tv in target_vars
                if tv.get("Name") and tv["Name"] not in sel_names and tv["Name"] in valid.columns]
    floor_txt = " · ".join(f"{k} {v:g}" for k, v in floors.items()) if floors else "적용 안 함"

    st.caption(
        f"평가 대상 목표: {tgt_txt}"
        + (f"  ·  제외 {len(excluded)}개" if excluded else "")
        + f"  ·  최소 허용값: {floor_txt}"
    )

    # ---- 조건 하나 ----
    cond_txt = " · ".join(f"{n} = **{best[n]}**" for n in cfg_names)
    st.markdown(
        f"🥇 **지금까지 최선의 조건** — {cond_txt}  \n"
        f"　반복 {int(best['n'])}회 · 종합점수 **{best['종합점수']:.3f}** / 1"
    )
    if int(best["n_min"]) < 3:
        st.caption("⚠️ 이 조건의 반복 수가 3회 미만이라 중앙값이 아직 흔들립니다. 순위를 그대로 믿지 마세요.")

    # ---- 조건 범위 ----
    # 점수 1~2위가 소수점 둘째 자리에서 갈리는 정도면 그 차이는 실험 편차 안이다.
    # 그래서 '한 점'이 아니라 '비슷한 점수의 조건들이 이루는 구간'을 함께 제시한다.
    margin = st.slider(
        "권장 범위로 묶을 종합점수 차이", 0.0, 0.30, 0.05, 0.01,
        key=f"{key_prefix}_copt_margin",
        help="1위와 이 값 이내로 붙어 있는 조건들을 '사실상 동급'으로 보고 한 구간으로 묶습니다. "
             "0 으로 두면 1위 조건 하나만 남습니다.")
    band = ranked[ranked["종합점수"] >= best["종합점수"] - margin]
    if len(band) > 1:
        parts = []
        for n in cfg_names:
            s = pd.to_numeric(band[n], errors="coerce")
            if s.notna().all() and s.min() != s.max():
                parts.append(f"{n} **{s.min():g} ~ {s.max():g}**")
            else:
                vs = sorted({str(v) for v in band[n]})
                parts.append(f"{n} **{', '.join(vs)}**")
        st.markdown(
            "🎯 **권장 조건 범위** — " + " · ".join(parts)
            + f"  \n　동급 조건 {len(band)}개 · 종합점수 {band['종합점수'].min():.3f}~{band['종합점수'].max():.3f}"
            f" · 반복 합계 {int(band['n'].sum())}회"
        )
    else:
        st.markdown("🎯 **권장 조건 범위** — 1위와 동급인 조건이 없어 위 조건 하나로 좁혀집니다.")

    # ---- 순위표 ----
    show = ranked.head(8).copy()
    show["종합점수"] = show["종합점수"].round(3)
    show = show.drop(columns=["n_min"]).rename(columns={"n": "반복 수"})
    show.insert(0, "순위", range(1, len(show) + 1))
    show = show[["순위", "종합점수"] + cfg_names + ["반복 수"] + cols]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption(
        "목표값은 같은 조건의 반복 시료를 **중앙값**으로 묶은 대표값입니다(평균은 튀는 시료 하나에 끌려갑니다). "
        f"종합점수는 각 목표를 방향에 맞춰 0~1(최선=1)로 매긴 desirability 를 **{dict(COMPOSITE_MODES)[mode].split(' (')[0]}**으로 합친 값입니다. "
        "0점 기준은 최소 허용값(설정했으면) 또는 관측 범위를 조금 넓힌 지점이라, 꼴찌 조건이 무조건 0점이 되지 않습니다."
    )
    if mode == "arithmetic":
        st.warning(
            "⚠️ 산술평균은 **한 목표의 우수함이 다른 목표의 부진을 상쇄**합니다. "
            "그래서 한쪽에서 1위를 쓸어 담고 다른 쪽에서 꼴찌인 조건이, 양쪽에서 2위인 조건을 이길 수 있습니다. "
            "모든 목표가 **동시에** 성립해야 하는 문제라면 기하평균이나 최솟값을 쓰세요."
        )
    if len(cols) >= 4:
        st.caption(
            "💡 서로 강하게 상관된 목표를 여러 개 넣으면 그 축이 그만큼 여러 표를 행사합니다"
            "(예: 인접 파장의 같은 바이어스 응답도). 물리적으로 독립인 목표만 남기는 편이 순위가 안정적입니다."
        )

    n_drop = int((~ok).sum())
    if n_drop:
        with st.expander(f"🚧 최소 허용값에 걸려 빠진 조건 {n_drop}개"):
            drop = tbl[~ok].copy()
            drop["미달 목표"] = [", ".join(why.loc[i]) for i in drop.index]
            st.dataframe(drop.drop(columns=["n_min"]).rename(columns={"n": "반복 수"}),
                         use_container_width=True, hide_index=True)
            st.caption("데이터를 지운 것이 아니라 이 평가에서만 빠졌습니다. GP 학습에는 그대로 쓰입니다.")

    st.caption(
        "**조건 하나**는 다음 실험을 당장 어디서 찍을지 정할 때 쓰고, **조건 범위**는 그 조건을 "
        "공정으로 고정해도 되는지 판단할 때 씁니다. 범위가 넓게 잡힌다면 그만큼 이 데이터로는 "
        "그 안에서 우열을 가릴 수 없다는 뜻이므로, 좁히려면 같은 조건을 여러 배치에 걸쳐 반복해야 합니다."
    )


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
