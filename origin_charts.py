"""공정 변수별 Origin 스타일 Plotly 그래프 + 제외 데이터 표시 + 내보내기(PNG/JPG/CSV).

photodetector-app 의 오리진 형식을 그대로 이식했다:
- 그래프 크기·축 스타일: pd_app/figure.py (10x8인치=960x768px, simple_white, 검정 mirror 축,
  inside 틱, exponentformat="E", Myriad Pro 폰트)
- 팔레트: pd_app/constants.ORIGIN_COLORS (24색 순서 그대로)
- 클라이언트 사이드 이미지 내보내기: pd_app/ui/summary.py (Plotly CDN + downloadImage, scale=3=300dpi)

Myriad Pro 는 pd 앱과 동일하게 폰트 파일을 번들하지 않고 이름만 지정한다 — 클라이언트에
설치돼 있으면 사용되고, 없으면 Pretendard 로 폴백한다.
"""

from __future__ import annotations

import base64
import functools
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---- Myriad Pro 번들 폰트 ----
# pd 앱은 폰트를 이름만 지정해 클라이언트 설치본에 의존했다. 여기서는 static/fonts 에
# 폰트를 번들해 @font-face 로 로드한다 → 폰트 미설치 PC나 내보낸 이미지에서도 동일하게 나온다.
_FONT_PATH = os.path.join(os.path.dirname(__file__), "static", "fonts", "MyriadPro-Regular.otf")
_FONT_STATIC_URL = "app/static/fonts/MyriadPro-Regular.otf"


@functools.lru_cache(maxsize=1)
def _font_b64():
    try:
        with open(_FONT_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except OSError:
        return ""


def _fontface_css(*, embed):
    """@font-face 규칙 문자열. 폰트가 없으면 "".
    embed=True  -> base64 data URI (내보내기 iframe 용, 완전 자립)
    embed=False -> static URL (화면 표시용, 브라우저 캐시)
    """
    head = "@font-face{font-family:'Myriad Pro';font-style:normal;font-weight:400;font-display:swap;src:"
    if embed:
        b = _font_b64()
        if not b:
            return ""
        return head + f"url(data:font/otf;base64,{b}) format('opentype');}}"
    if not os.path.exists(_FONT_PATH):
        return ""
    return head + f"url('{_FONT_STATIC_URL}') format('opentype');}}"

# ---- pd_app/figure.py 규격 그대로 ----
FIG_DPI = 96
PAGE_W_IN = 10.0
PAGE_H_IN = 8.0
FIG_W = int(round(PAGE_W_IN * FIG_DPI))   # 960
FIG_H = int(round(PAGE_H_IN * FIG_DPI))   # 768
_AXIS_LINEWIDTH = 1.5
_TICKLEN_MAJOR = 6
_TICKLEN_MINOR = 3
FONT_FAMILY = "Myriad Pro"

# pd_app/constants.ORIGIN_COLORS 값 순서 그대로 (OriginLab 24색)
ORIGIN_COLORS = [
    "#000000", "#FF0000", "#00FF00", "#0000FF", "#00FFFF", "#FF00FF",
    "#FFFF00", "#808000", "#000080", "#800080", "#800000", "#008000",
    "#008080", "#0000A0", "#FF8000", "#8000FF", "#FF0080", "#FFFFFF",
    "#C0C0C0", "#808080", "#FFFF80", "#80FFFF", "#FF80FF", "#404040",
]

# 흰 배경에서 안 보이는 흰색은 목표 기본색에서 건너뛴다 (사용자가 색선택기로 바꾸는 건 자유).
_DEFAULT_TRACE_COLORS = [c for c in ORIGIN_COLORS if c != "#FFFFFF"]


def origin_color(i: int) -> str:
    return _DEFAULT_TRACE_COLORS[i % len(_DEFAULT_TRACE_COLORS)]


def _normalize(series) -> pd.Series:
    """0~1 정규화. 상수 컬럼(max==min) 또는 데이터 없음은 0.5."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


# =====================================================================
# 제외된 데이터
# =====================================================================
def compute_excluded_rows(df_data, feature_cols, target_cols, config_vars, include_range):
    """AI 학습에서 빠지는 행 + '제외 사유' 컬럼을 붙인 DataFrame 반환. 없으면 빈 DataFrame.

    사유 3종:
      - '학습 제외 (수동)'   : 학습_적용 체크 해제
      - '이상치 (목표명)'    : 같은 공정 조건 반복측정(3개 이상) 중 IQR(1.5) 밖 값 (목표별)
      - '범위 밖 (변수명)'   : 공정 변수 설정 Min~Max 밖 / 옵션에 없는 값 (단일 목표 경로에서만)
    """
    if df_data is None or df_data.empty:
        return pd.DataFrame()

    reasons = {idx: [] for idx in df_data.index}

    if "학습_적용" in df_data.columns:
        manual_mask = df_data["학습_적용"] != True  # noqa: E712
    else:
        manual_mask = pd.Series(False, index=df_data.index)
    for idx in df_data.index[manual_mask]:
        reasons[idx].append("학습 제외 (수동)")

    valid = df_data[~manual_mask]

    # IQR 이상치 (목표별)
    feats = [c for c in feature_cols if c in valid.columns]
    if feats and len(valid):
        for _, group in valid.groupby(feats, dropna=False):
            for t_col in target_cols:
                if t_col not in group.columns:
                    continue
                y = pd.to_numeric(group[t_col], errors="coerce")
                yv = y.dropna()
                if len(yv) >= 3:
                    q1, q3 = np.percentile(yv, [25, 75])
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    out = group.index[(y < lo) | (y > hi)]
                    # 전부 이상치면 process_robust_data 가 원본을 유지하므로 제외로 치지 않는다.
                    if 0 < len(out) < len(yv):
                        for idx in out:
                            reasons[idx].append(f"이상치 ({t_col})")

    # 범위 밖 (단일 목표 경로에서만 실제 제외됨)
    if include_range:
        for idx in valid.index:
            for var in config_vars:
                name = var.get("Name")
                if not name or name not in df_data.columns:
                    continue
                val = df_data.at[idx, name]
                if "Categorical" in var.get("Type", ""):
                    opts = [o.strip() for o in str(var.get("Options", "")).split(",") if o.strip()]
                    if opts and str(val) not in opts:
                        reasons[idx].append(f"범위 밖 ({name})")
                else:
                    try:
                        v, lo, hi = float(val), float(var.get("Min")), float(var.get("Max"))
                        if not (lo <= v <= hi):
                            reasons[idx].append(f"범위 밖 ({name})")
                    except (TypeError, ValueError):
                        pass

    excluded = [idx for idx in df_data.index if reasons[idx]]
    if not excluded:
        return pd.DataFrame()
    out_df = df_data.loc[excluded].copy()
    out_df.insert(0, "제외 사유", [", ".join(reasons[idx]) for idx in excluded])
    return out_df


def render_excluded_expander(df_data, feature_cols, target_cols, config_vars, include_range, key):
    """AI 계산 영역에 '제외된 데이터 N건' 익스팬더를 그린다."""
    exc = compute_excluded_rows(df_data, feature_cols, target_cols, config_vars, include_range)
    n = len(exc)
    with st.expander(f"🚫 제외된 데이터 {n}건 (AI 학습에서 빠짐)", expanded=False):
        st.caption(
            "제외 사유는 3가지입니다 — "
            "**학습 제외(수동)**: 데이터베이스 관리 탭에서 '학습 적용'을 끈 행 · "
            "**이상치**: 완전히 같은 공정 조건을 3번 이상 반복 측정한 그룹 안에서 IQR(1.5배) 밖으로 튀는 값 "
            "(반복 2개 이하는 판정 안 함, 5개 이상부터 잘 잡힘) · "
            "**범위 밖**: 공정 변수 설정 범위(최소~최대)를 벗어난 값(단일 목표 계산에만 적용)."
        )
        if n == 0:
            st.caption("현재 제외된 데이터가 없습니다. 모든 유효 데이터가 AI 학습에 사용됩니다.")
            return
        show_cols = ["제외 사유"]
        if "샘플명" in exc.columns:
            show_cols.append("샘플명")
        show_cols += [c for c in feature_cols if c in exc.columns]
        show_cols += [c for c in target_cols if c in exc.columns]
        st.dataframe(exc[show_cols], use_container_width=True, hide_index=True, key=f"exc_df_{key}")


# =====================================================================
# Origin 스타일 figure
# =====================================================================
def build_variable_figure(df, var, target_vars, style):
    """공정 변수 var(dict) 1개에 대한 Origin 스타일 figure.

    df          : 학습 적용된 유효 데이터
    target_vars : [{Name, Unit, Direction, ...}] (df 에 컬럼이 있는 것만 넘어온다)
    style       : {x_title, y_title, title_font_size, tick_font_size, line_width,
                   show_markers, colors: {target_name: hex}}
    """
    var_name = var["Name"]
    is_cat = "Categorical" in var.get("Type", "")
    x_raw = df[var_name]
    order = np.arange(len(x_raw)) if is_cat else np.argsort(
        pd.to_numeric(x_raw, errors="coerce").values, kind="stable")

    fig = go.Figure()
    for i, tv in enumerate(target_vars):
        tname = tv["Name"]
        if tname not in df.columns:
            continue
        y_norm = _normalize(df[tname]).values[order]
        y_orig = pd.to_numeric(df[tname], errors="coerce").values[order]
        xs = x_raw.values[order]
        color = style["colors"].get(tname, origin_color(i))
        unit = f" {tv['Unit']}" if tv.get("Unit") else ""
        fig.add_trace(go.Scatter(
            x=xs, y=y_norm,
            mode="lines+markers" if style["show_markers"] else "lines",
            name=tname,
            line=dict(color=color, width=float(style["line_width"])),
            marker=dict(color=color, size=8),
            customdata=y_orig,
            hovertemplate=(f"{var_name}: %{{x}}<br>{tname}: %{{customdata:.6g}}{unit}"
                           f"<br>정규화: %{{y:.3f}}<extra></extra>"),
        ))

    tick_font = dict(family=FONT_FAMILY, size=style["tick_font_size"], color="black")
    title_font = dict(family=FONT_FAMILY, size=style["title_font_size"], color="black")
    axis_common = dict(
        showline=True, linecolor="black", linewidth=_AXIS_LINEWIDTH, mirror=True,
        ticks="inside", tickwidth=_AXIS_LINEWIDTH, tickcolor="black", ticklen=_TICKLEN_MAJOR,
        showgrid=False, zeroline=False, tickfont=tick_font,
    )
    x_kw = dict(axis_common)
    x_kw["title"] = dict(text=style["x_title"], font=title_font)
    if is_cat:
        x_kw["type"] = "category"
    else:
        x_kw["exponentformat"] = "E"
        x_kw["showexponent"] = "all"

    y_kw = dict(axis_common)
    y_kw["title"] = dict(text=style["y_title"], font=title_font)
    y_kw["range"] = [-0.05, 1.05]

    fig.update_layout(
        template="simple_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=FONT_FAMILY, color="black"),
        xaxis=x_kw, yaxis=y_kw,
        showlegend=True,
        legend=dict(font=dict(family=FONT_FAMILY, size=max(6, int(style["tick_font_size"] * 0.7)))),
        margin=dict(l=90, r=30, t=30, b=85),
        width=FIG_W, height=FIG_H,
    )
    return fig


# =====================================================================
# 내보내기 (pd_app/ui/summary.py 방식 그대로)
# =====================================================================
_BTN_STYLE = (
    "width:100%; height:38px; margin:0; padding:0; "
    "background-color:rgb(255, 255, 255); color:rgb(49, 51, 63); "
    "border:1px solid rgba(49, 51, 63, 0.2); border-radius:0.5rem; "
    "cursor:pointer; font-size:14px; font-weight:400; "
    "font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; "
    "display:inline-flex; align-items:center; justify-content:center; "
    "transition: border-color 0.15s ease, color 0.15s ease; box-sizing:border-box;"
)
_BTN_HOVER = "this.style.borderColor='#ed542b'; this.style.color='#ed542b';"
_BTN_LEAVE = "this.style.borderColor='rgba(49, 51, 63, 0.2)'; this.style.color='rgb(49, 51, 63)';"


def _fig_json(fig, *, transparent):
    f = go.Figure(fig)
    if transparent:
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    else:
        f.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    return f.to_json().replace("</script>", "<\\/script>")


def _export_image_button(fig, *, fmt, transparent, filename, label, btn_id):
    """클라이언트 사이드 Plotly 로 960x768 를 scale=3(=300dpi) 이미지로 내려받는 버튼."""
    fig_json = _fig_json(fig, transparent=transparent)
    # 내보낸 이미지에 Myriad Pro 를 고정하려면 iframe 안에서도 폰트가 로드돼 있어야 한다
    # (iframe 은 앱 CSS 를 상속하지 않으므로 base64 로 임베드하고, downloadImage 전에 로드 완료를 기다린다).
    fontface = _fontface_css(embed=True)
    html = f"""
    <html>
    <head>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>{fontface}</style>
    </head>
    <body style="margin:0; padding:0; background:transparent; overflow:hidden;">
    <button id="{btn_id}" style="{_BTN_STYLE}" onmouseover="{_BTN_HOVER}" onmouseout="{_BTN_LEAVE}">{label}</button>
    <script>
    document.getElementById('{btn_id}').addEventListener('click', function() {{
        if (typeof Plotly === 'undefined') {{
            alert('이미지 생성 엔진 로딩 중입니다. 1~2초 뒤 다시 클릭해주세요.');
            return;
        }}
        var d = document.createElement('div');
        d.style.position = 'absolute'; d.style.left = '-9999px';
        d.style.width = '960px'; d.style.height = '768px';
        document.body.appendChild(d);
        var figData = {fig_json};
        var doDownload = function() {{
            Plotly.downloadImage(d, {{format: '{fmt}', width: 960, height: 768, scale: 3, filename: '{filename}'}}).then(function() {{
                document.body.removeChild(d);
            }});
        }};
        Plotly.newPlot(d, figData.data, figData.layout).then(function() {{
            // Myriad Pro 로드 완료를 기다린 뒤 렌더 (미설치 PC에서도 임베드 폰트로 그려지게)
            if (document.fonts && document.fonts.load) {{
                document.fonts.load("30px 'Myriad Pro'").then(function() {{ return document.fonts.ready; }})
                    .then(doDownload, doDownload);
            }} else {{
                doDownload();
            }}
        }});
    }});
    </script>
    </body>
    </html>
    """
    components.html(html, height=44)


def _variable_csv(df, var, target_vars):
    name = var["Name"]
    out = pd.DataFrame({name: df[name].values})
    for tv in target_vars:
        tn = tv["Name"]
        if tn in df.columns:
            out[tn] = pd.to_numeric(df[tn], errors="coerce").values
            out[f"{tn}_정규화"] = _normalize(df[tn]).values
    return out.to_csv(index=False).encode("utf-8-sig")


# =====================================================================
# 섹션 렌더
# =====================================================================
def render_variable_charts(df_valid, config_vars, target_vars, key_prefix="vardist"):
    """공정 변수별 Origin 스타일 그래프 + 그래프별 PNG/JPG/CSV 내보내기."""
    cvars = [v for v in config_vars if v.get("Name") and v["Name"] in df_valid.columns]
    targets = [tv for tv in target_vars if tv.get("Name") and tv["Name"] in df_valid.columns]

    if df_valid.empty or not cvars or not targets:
        st.info("학습 적용된 데이터와 목표 지표가 있어야 그래프가 그려집니다.")
        return

    # 화면 표시용 Myriad Pro @font-face (static URL). 그래프가 이 family 를 참조한다.
    _ff = _fontface_css(embed=False)
    if _ff:
        st.markdown(f"<style>{_ff}</style>", unsafe_allow_html=True)

    # 공통 스타일 컨트롤 (전 그래프 공통 1세트)
    with st.expander("🎨 그래프 스타일 (전체 공통)", expanded=False):
        cS = st.columns(4)
        title_fs = cS[0].number_input("제목 글씨 크기", 6, 50, 30, key=f"{key_prefix}_title_fs")
        tick_fs = cS[1].number_input("눈금 글씨 크기", 6, 50, 30, key=f"{key_prefix}_tick_fs")
        line_w = cS[2].number_input("선 두께", 0.5, 10.0, 2.0, step=0.5, key=f"{key_prefix}_lw")
        show_markers = cS[3].checkbox("마커 표시", True, key=f"{key_prefix}_mk")
        y_title = st.text_input("Y축 제목 (공통)", "정규화 목표값 (0–1)", key=f"{key_prefix}_ytitle")
        st.caption("목표별 선 색상 (Origin 팔레트 기본)")
        ccols = st.columns(min(len(targets), 6))
        colors = {}
        for i, tv in enumerate(targets):
            colors[tv["Name"]] = ccols[i % len(ccols)].color_picker(
                tv["Name"], origin_color(i), key=f"{key_prefix}_col_{i}")

    for vi, var in enumerate(cvars):
        vname = var["Name"]
        unit = f" ({var['Unit']})" if var.get("Unit") else ""
        x_title = st.text_input(f"X축 제목 — {vname}", f"{vname}{unit}", key=f"{key_prefix}_xt_{vi}")
        style = dict(x_title=x_title, y_title=y_title, title_font_size=title_fs,
                     tick_font_size=tick_fs, line_width=line_w, show_markers=show_markers, colors=colors)
        fig = build_variable_figure(df_valid, var, targets, style)
        st.plotly_chart(fig, width="content", config={
            "displaylogo": False, "responsive": False,
            "toImageButtonOptions": {"format": "png", "width": FIG_W, "height": FIG_H,
                                     "scale": 3, "filename": vname},
        })
        e1, e2, e3 = st.columns(3)
        with e1:
            _export_image_button(fig, fmt="png", transparent=True, filename=f"{vname}_dist",
                                 label="🖼️ PNG (투명)", btn_id=f"{key_prefix}_png_{vi}")
        with e2:
            _export_image_button(fig, fmt="jpeg", transparent=False, filename=f"{vname}_dist",
                                 label="📷 JPG (흰 배경)", btn_id=f"{key_prefix}_jpg_{vi}")
        with e3:
            st.download_button("📊 CSV 다운로드", data=_variable_csv(df_valid, var, targets),
                               file_name=f"{vname}_dist.csv", mime="text/csv",
                               use_container_width=True, key=f"{key_prefix}_csv_{vi}")
        st.divider()
