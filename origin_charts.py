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
import json
import os
import re

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


# 목표 지표별로 서로 다른 마커 모양을 순환시킨다 (전부 원형이면 겹칠 때 구분이 안 됨).
_MARKER_SYMBOLS = [
    "circle", "square", "diamond", "triangle-up", "cross",
    "x", "triangle-down", "star", "hexagon", "pentagon",
]


def marker_symbol(i: int) -> str:
    return _MARKER_SYMBOLS[i % len(_MARKER_SYMBOLS)]


# =====================================================================
# Origin 24색 팔레트 색 피커 (FET-studio/fet_app/ui/color_picker.py 그대로 이식)
#
# 포팅 출처: photodetector-app/pd_app/ui/panel_traces.py 의 _color_dialog /
# _color_control 을 FET-studio 가 재사용 컴포넌트로 다듬은 버전. 스와치 그리드,
# 정사각형 트리거 버튼, 커스텀 색을 HTML5 <input type="color"> + JS 동기화로
# 다루는 우회법까지 그대로 가져왔다.
#
# 왜 커스텀 색에 st.color_picker 를 안 쓰는가: 누르면 브라우저 네이티브 팝업이
# 뜨는데, 그 클릭이 st.dialog 모달의 "바깥 클릭 시 닫기" 판정에 걸려 색을
# 고르려는 순간 다이얼로그가 꺼진다. 그래서 네이티브 <input type="color"> 를
# components.html 이 만드는 iframe 안에 두어 부모 문서의 클릭 감지에서 격리시키고,
# 그 oninput 에서 JS 로 옆 st.text_input(hex 칸) 값을 네이티브 setter 로 갱신 +
# input 이벤트를 dispatch 해 Streamlit 에 알린다.
# =====================================================================
ORIGIN_PALETTE = {
    "Black": "#000000", "Red": "#FF0000", "Green": "#00FF00", "Blue": "#0000FF",
    "Cyan": "#00FFFF", "Magenta": "#FF00FF", "Yellow": "#FFFF00",
    "Dark Yellow": "#808000", "Navy": "#000080", "Purple": "#800080",
    "Wine": "#800000", "Olive": "#008000", "Dark Cyan": "#008080",
    "Royal": "#0000A0", "Orange": "#FF8000", "Violet": "#8000FF",
    "Pink": "#FF0080", "White": "#FFFFFF", "LT Gray": "#C0C0C0",
    "Gray": "#808080", "LT Yellow": "#FFFF80", "LT Cyan": "#80FFFF",
    "LT Magenta": "#FF80FF", "Dark Gray": "#404040",
}

_CP_CUSTOM = "Custom"
_CP_GRID_COLS = 8
_CP_SWATCH_SIZE = "32px"
_CP_TARGET = "_origin_color_picker_target"
_CP_UNSAFE = re.compile(r"\W+")
_CP_BRIGHT_CUTOFF = 150.0


def cp_normalize_hex(hex_color, default: str = "#000000") -> str:
    """'ff8000' / '#f80' / '#FF8000' -> '#FF8000'. 못 읽으면 default."""
    h = str(hex_color).strip().upper().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6 or any(c not in "0123456789ABCDEF" for c in h):
        return default
    return "#" + h


def cp_color_name(hex_color) -> str:
    """hex -> Origin 팔레트 이름 역조회. 팔레트에 없으면 "Custom"."""
    h = cp_normalize_hex(hex_color, default="")
    if not h:
        return _CP_CUSTOM
    return next((k for k, v in ORIGIN_PALETTE.items()
                 if cp_normalize_hex(v) == h), _CP_CUSTOM)


def cp_color_caption(hex_color) -> str:
    name = cp_color_name(hex_color)
    return cp_normalize_hex(hex_color) if name == _CP_CUSTOM else name


def _cp_contrast_text(hex_color) -> str:
    h = cp_normalize_hex(hex_color).lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > _CP_BRIGHT_CUTOFF else "#FFFFFF"


def _cp_slug(key: str) -> str:
    return _CP_UNSAFE.sub("_", str(key))


def _cp_trigger_key(key: str) -> str:
    return f"origincp_trig_{_cp_slug(key)}"


def _cp_custom_key(key: str) -> str:
    return f"origincp_custom_{_cp_slug(key)}"


def _cp_swatch_css(class_key: str, hex_color: str, *, size: str | None = None) -> str:
    """.st-key-<key> 컨벤션으로 그 버튼 하나만 색 스와치처럼 칠한다."""
    box = (f"width: {size} !important; height: {size} !important; "
           f"min-width: {size} !important; min-height: {size} !important; "
           f"padding: 0 !important; display: block !important; "
           f"margin: 0 auto !important; ") if size else ""
    return (
        f"<style>"
        f".st-key-{class_key} button {{"
        f" background: {hex_color} !important;"
        f" color: {_cp_contrast_text(hex_color)} !important;"
        f" border: 1px solid rgba(0,0,0,0.35) !important;"
        f" border-radius: 6px !important; {box}"
        f"}}"
        f".st-key-{class_key} button:hover {{"
        f" border: 2px solid #000 !important;"
        f" color: {_cp_contrast_text(hex_color)} !important;"
        f"}}"
        f".st-key-{class_key} button p {{"
        f" color: {_cp_contrast_text(hex_color)} !important;"
        f"}}"
        f"</style>"
    )


def _cp_native_color_input_html(current: str, hex_key: str) -> str:
    """iframe 안에 띄울 HTML5 <input type="color">. oninput 에서 부모 문서의
    hex 텍스트 인풋을 네이티브 setter 로 갱신 + input 이벤트를 dispatch한다."""
    return (
        f'<div style="display:flex; align-items:center; justify-content:center;">'
        f'<input type="color" value="{cp_normalize_hex(current)}" '
        f'style="width:38px; height:38px; padding:0; border:1px solid #ccc; '
        f'border-radius:6px; cursor:pointer;" '
        f'oninput="'
        f"var hexInput = window.parent.document.querySelector('.st-key-{hex_key} input');"
        f'if (hexInput) {{'
        f"var nativeSetter = Object.getOwnPropertyDescriptor("
        f"window.HTMLInputElement.prototype, 'value').set;"
        f'nativeSetter.call(hexInput, this.value);'
        f"hexInput.dispatchEvent(new Event('input', {{ bubbles: true }}));"
        f'}}">'
        f'</div>'
    )


@st.dialog("🎨 색 선택")
def _origin_palette_dialog() -> None:
    """Origin 24색 그리드 + 커스텀 색 폴백. 편집 대상은 세션 상태에서 읽는다."""
    tgt = st.session_state.get(_CP_TARGET)
    if not tgt:
        st.caption("편집할 색 대상을 찾지 못했습니다.")
        return
    target, field = tgt["target"], tgt["field"]
    label, slug = tgt["label"], tgt["slug"]
    current = cp_normalize_hex(target.get(field, "#000000"))

    st.markdown(f"**{label}** · 현재 `{cp_color_caption(current)}`")
    st.caption("Origin 24색 팔레트")

    items = list(ORIGIN_PALETTE.items())
    for start in range(0, len(items), _CP_GRID_COLS):
        cols = st.columns(_CP_GRID_COLS)
        for offset, (name, hexv) in enumerate(items[start:start + _CP_GRID_COLS]):
            with cols[offset]:
                btn_key = f"origincp_sw_{slug}_{start + offset}"
                st.markdown(_cp_swatch_css(btn_key, hexv, size=_CP_SWATCH_SIZE),
                            unsafe_allow_html=True)
                if st.button(" ", key=btn_key, help=name, use_container_width=False):
                    target[field] = hexv
                    st.rerun()

    st.divider()
    st.caption("커스텀 색상")
    hex_key = _cp_custom_key(tgt["key"])
    c_pick, c_hex, c_apply = st.columns([1, 2, 2], vertical_alignment="bottom")
    with c_pick:
        components.html(_cp_native_color_input_html(current, hex_key), height=56)
    with c_hex:
        typed = st.text_input("Hex 코드", value=current, key=hex_key,
                              max_chars=7, label_visibility="collapsed")
    with c_apply:
        if st.button("설정 적용", key=f"origincp_apply_{slug}", type="primary",
                     use_container_width=True):
            target[field] = cp_normalize_hex(typed, default=current)
            st.rerun()


def origin_color_picker(label: str, target: dict, field: str, *,
                        key: str, default: str = "#000000") -> str:
    """색 하나를 고르는 트리거 버튼(정사각형 스와치). 고른 색은 target[field] 에 바로 들어간다.
    반환값은 지금 적용돼 있는 색(#RRGGBB). target 은 세션 상태에 보관된 dict 여야
    리런 사이에 값이 유지된다(예: st.session_state 안의 {목표명: hex} dict)."""
    current = cp_normalize_hex(target.get(field, default), default=default)
    slug = _cp_slug(key)
    btn_key = _cp_trigger_key(key)

    st.caption(label)
    st.markdown(_cp_swatch_css(btn_key, current, size=_CP_SWATCH_SIZE), unsafe_allow_html=True)
    if st.button(" ", key=btn_key, use_container_width=False,
                 help=f"{label} — 현재 {cp_color_caption(current)} · 클릭해서 Origin 팔레트에서 고르기"):
        st.session_state.pop(_cp_custom_key(key), None)
        st.session_state[_CP_TARGET] = {"target": target, "field": field,
                                        "label": label, "slug": slug, "key": key}
        _origin_palette_dialog()
    return current


Y_TITLE_NORM = "정규화 목표값 (0–1)"
Y_TITLE_RAW = "목표값 (원본 단위)"


def _auto_default_text_input(label, default, key, *, auto_values=(), **kwargs):
    """기본값이 다른 설정에 따라 바뀌는 text_input.

    key 가 붙은 위젯은 세션에 저장된 값이 value= 인자보다 우선이라, 기본값을 다시
    계산해도 화면에 찍힌 값은 그대로 남는다 — 정규화를 꺼도 Y축 제목이 계속
    "정규화 목표값"으로 보이던 원인이 이것이다.

    그래서 '자동으로 넣어준 기본값'을 따로 기억해 두고, 사용자가 손대지 않았을 때
    (= 현재 값이 그 기본값 그대로일 때)만 새 기본값으로 갈아끼운다. 직접 고쳐 쓴
    제목은 덮어쓰지 않는다.

    auto_values 는 '자동 기본값이 될 수 있는 값들'이다. 이 함수가 생기기 전부터
    열려 있던 세션에는 기억용 마커가 없어서(prev_auto=None) 예전 코드가 넣어둔
    값을 사용자가 직접 고친 것으로 오해하고 그대로 두게 된다. 그 값이 자동 기본값
    후보 중 하나라면 사용자가 고친 게 아니므로 갱신 대상으로 본다.
    (로컬 개발 중 코드만 리로드돼 세션이 살아남는 경우에도 같은 상황이 된다.)
    """
    auto_key = f"{key}__auto"
    prev_auto = st.session_state.get(auto_key)
    if prev_auto != default:
        cur = st.session_state.get(key)
        untouched = (key not in st.session_state
                     or cur == prev_auto
                     or (prev_auto is None and cur in auto_values))
        if untouched:
            st.session_state[key] = default
        st.session_state[auto_key] = default
    return st.text_input(label, default, key=key, **kwargs)


def _normalize(series) -> pd.Series:
    """0~1 정규화. 상수 컬럼(max==min) 또는 데이터 없음은 0.5."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


# =====================================================================
# Origin 스타일 figure
# =====================================================================
def build_variable_figure(df, var, target_vars, style, scale=1.0):
    """공정 변수 var(dict) 1개에 대한 Origin 스타일 figure.

    df          : 학습 적용된 유효 데이터
    target_vars : [{Name, Unit, Direction, ...}] (df 에 컬럼이 있는 것만 넘어온다)
    style       : {x_title, y_title, title_font_size, tick_font_size, line_width,
                   show_markers, colors: {target_name: hex}, normalize, show_trendline,
                   trendline_opacity, trend_degree}
    scale       : figure 전체(크기·폰트·선·마커·여백)를 비율대로 축소/확대. 화면 표시용
                  축소에 쓰고, 내보내기는 항상 scale=1.0(=960x768 원본)로 만든다. pd 앱의
                  px_scale 과 같은 개념 — 모든 치수를 함께 곱해 비율을 유지한다.
    """
    var_name = var["Name"]
    is_cat = "Categorical" in var.get("Type", "")
    x_raw = df[var_name]
    order = np.arange(len(x_raw)) if is_cat else np.argsort(
        pd.to_numeric(x_raw, errors="coerce").values, kind="stable")
    normalize = style.get("normalize", True)
    trend_degree = int(style.get("trend_degree", 1))
    s = float(scale)

    fig = go.Figure()
    for i, tv in enumerate(target_vars):
        tname = tv["Name"]
        if tname not in df.columns:
            continue
        y_norm = _normalize(df[tname]).values[order]
        y_orig = pd.to_numeric(df[tname], errors="coerce").values[order]
        y_plot = y_norm if normalize else y_orig
        xs = x_raw.values[order]
        color = style["colors"].get(tname, origin_color(i))
        unit = f" {tv['Unit']}" if tv.get("Unit") else ""

        # 포인트만 찍는다 (점끼리 잇지 않음) — 목표별로 마커 모양도 다르게 순환.
        if style["show_markers"]:
            if normalize:
                hover = (f"{var_name}: %{{x}}<br>{tname}: %{{customdata:.6g}}{unit}"
                         f"<br>정규화: %{{y:.3f}}<extra></extra>")
            else:
                hover = f"{var_name}: %{{x}}<br>{tname}: %{{y:.6g}}{unit}<extra></extra>"
            fig.add_trace(go.Scatter(
                x=xs, y=y_plot,
                mode="markers",
                name=tname,
                marker=dict(color=color, size=10 * s, symbol=marker_symbol(i)),
                customdata=y_orig,
                hovertemplate=hover,
                legendgroup=tname,
            ))

        # 추세선(다항 회귀, 기본 1차=선형)은 포인트와 별개 트레이스로, 기본 40% 투명도로 그린다.
        # 범주형 x축은 회귀선이 의미가 없어 건너뛴다.
        if style.get("show_trendline", True) and not is_cat:
            xs_num = pd.to_numeric(pd.Series(xs), errors="coerce").values
            fit_mask = np.isfinite(xs_num) & np.isfinite(y_plot)
            # 차수+1개 이상의 점이 있어야 그 차수로 과적합 없이 피팅할 수 있다.
            if fit_mask.sum() >= trend_degree + 1 and np.ptp(xs_num[fit_mask]) > 0:
                coeffs = np.polyfit(xs_num[fit_mask], y_plot[fit_mask], trend_degree)
                # 1차보다 높은 차수는 두 점을 직선으로 잇는 걸로는 곡률이 안 보이므로
                # 구간을 촘촘히 나눠 곡선으로 그린다.
                x_line = np.linspace(xs_num[fit_mask].min(), xs_num[fit_mask].max(), 100)
                y_line = np.polyval(coeffs, x_line)
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines",
                    name=f"{tname} 추세선",
                    line=dict(color=color, width=float(style["line_width"]) * s),
                    opacity=float(style.get("trendline_opacity", 0.4)),
                    legendgroup=tname,
                    showlegend=False,
                    hoverinfo="skip",
                ))

    tick_font = dict(family=FONT_FAMILY, size=style["tick_font_size"] * s, color="black")
    title_font = dict(family=FONT_FAMILY, size=style["title_font_size"] * s, color="black")
    axis_common = dict(
        showline=True, linecolor="black", linewidth=_AXIS_LINEWIDTH * s, mirror=True,
        ticks="inside", tickwidth=_AXIS_LINEWIDTH * s, tickcolor="black", ticklen=_TICKLEN_MAJOR * s,
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
    if normalize:
        # 정규화(0~1) 모드에서만 축 범위를 고정한다. 원본값은 목표마다 단위/스케일이
        # 달라 고정 범위가 의미 없으므로 Plotly 자동 범위에 맡긴다.
        y_kw["range"] = [-0.05, 1.05]
    else:
        y_kw["exponentformat"] = "E"
        y_kw["showexponent"] = "all"

    fig.update_layout(
        template="simple_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=FONT_FAMILY, color="black"),
        xaxis=x_kw, yaxis=y_kw,
        showlegend=True,
        legend=dict(font=dict(family=FONT_FAMILY, size=max(6, style["tick_font_size"] * 0.7 * s))),
        margin=dict(l=90 * s, r=30 * s, t=30 * s, b=85 * s),
        width=int(FIG_W * s), height=int(FIG_H * s),
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
    # filename 은 사용자가 입력한 공정 변수명에서 온다 — JS 문자열에 그대로 넣으면 따옴표/특수문자로
    # 내보내기가 깨지거나(예: 변수명에 ') XSS 가 될 수 있으므로 json.dumps 로 안전한 JS 리터럴로 만든다.
    fmt_js = json.dumps(fmt)
    filename_js = json.dumps(filename)
    btn_id_js = json.dumps(btn_id)
    html = f"""
    <html>
    <head>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>{fontface}</style>
    </head>
    <body style="margin:0; padding:0; background:transparent; overflow:hidden;">
    <button id="{btn_id}" style="{_BTN_STYLE}" onmouseover="{_BTN_HOVER}" onmouseout="{_BTN_LEAVE}">{label}</button>
    <script>
    document.getElementById({btn_id_js}).addEventListener('click', function() {{
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
            Plotly.downloadImage(d, {{format: {fmt_js}, width: 960, height: 768, scale: 3, filename: {filename_js}}}).then(function() {{
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

    # 등록된 목표가 많으면 그래프 한 장에 다 겹쳐 그리기보다 필요한 것만 골라 보는 게
    # 낫다 — 기본은 전체 표시(기존 동작 유지), 여기서 줄이면 그래프·범례·색상칸에 전부 반영.
    all_target_names = [tv["Name"] for tv in targets]
    sel_target_names = st.multiselect(
        "표시할 목표 지표", all_target_names, default=all_target_names,
        key=f"{key_prefix}_sel_targets",
        help="계산에 포함했는지와 무관하게, 이 그래프에 겹쳐 그릴 목표만 고릅니다.",
    )
    targets = [tv for tv in targets if tv["Name"] in sel_target_names]
    if not targets:
        st.info("표시할 목표 지표를 선택하세요.")
        return

    # 공통 스타일 컨트롤 (전 그래프 공통 1세트)
    with st.expander("🎨 그래프 스타일 (전체 공통)", expanded=False):
        cS = st.columns(4)
        title_fs = cS[0].number_input("제목 글씨 크기", 6, 50, 30, key=f"{key_prefix}_title_fs")
        tick_fs = cS[1].number_input("눈금 글씨 크기", 6, 50, 30, key=f"{key_prefix}_tick_fs")
        line_w = cS[2].number_input("추세선 두께", 0.5, 10.0, 2.0, step=0.5, key=f"{key_prefix}_lw")
        show_markers = cS[3].checkbox("포인트 표시", True, key=f"{key_prefix}_mk")
        cT = st.columns(4)
        show_trendline = cT[0].checkbox("추세선 표시", True, key=f"{key_prefix}_trend")
        trendline_opacity = cT[1].slider("추세선 투명도", 0, 100, 40, key=f"{key_prefix}_trend_op") / 100.0
        trend_degree = cT[2].number_input("추세선 차수", 1, 6, 1, key=f"{key_prefix}_trend_deg",
                                          help="1=선형, 2 이상은 다항 회귀 곡선")
        normalize = cT[3].checkbox("Y값 정규화 (0~1)", True, key=f"{key_prefix}_norm",
                                   help="끄면 목표별 원본 단위 그대로 표시합니다 (목표마다 스케일이 달라도 그대로 겹쳐 그림).")

        disp_scale = st.slider(
            "🔍 그래프 표시 배율 (%)", 30, 100, 60, 5, key=f"{key_prefix}_disp",
            help="화면 표시 크기만 조절합니다. 내보내는 PNG/JPG는 배율과 무관하게 항상 출판용 원본(960×768)입니다.",
        ) / 100.0

        st.caption("X축 제목 (그래프별)")
        x_titles = {}
        for vi, var in enumerate(cvars):
            vname = var["Name"]
            unit = f" ({var['Unit']})" if var.get("Unit") else ""
            # 위젯 키가 인덱스 기반이라, 변수 이름/단위를 바꾸면 같은 이유로 옛 제목이 남는다.
            x_titles[vname] = _auto_default_text_input(
                f"X축 제목 — {vname}", f"{vname}{unit}", f"{key_prefix}_xt_{vi}")

        default_y_title = Y_TITLE_NORM if normalize else Y_TITLE_RAW
        y_title = _auto_default_text_input(
            "Y축 제목 (공통)", default_y_title, f"{key_prefix}_ytitle",
            auto_values=(Y_TITLE_NORM, Y_TITLE_RAW))
        st.caption("목표별 선 색상 — 클릭해서 Origin 24색 팔레트에서 고르거나 커스텀 색 입력")
        # target[field] 패턴은 리런 사이에 유지되는 dict 참조가 필요하므로, 세션 상태에
        # {목표명: hex} dict 를 하나 두고 그 안의 값을 직접 편집한다.
        color_store = st.session_state.setdefault(f"{key_prefix}_colorstore", {})
        ccols = st.columns(min(len(targets), 6))
        colors = {}
        for i, tv in enumerate(targets):
            with ccols[i % len(ccols)]:
                colors[tv["Name"]] = origin_color_picker(
                    tv["Name"], color_store, tv["Name"],
                    key=f"{key_prefix}_col_{i}", default=origin_color(i))

    # 공정 변수들은 서로 상관될 수 있어(다중공선성), 한 변수의 효과를 보려면 나머지 변수를 고정해야
    # 교란이 줄어든다. 각 변수에 제약(숫자=범위, 범주=허용값)을 두고, 각 그래프는 자기 축 변수를
    # 뺀 나머지 제약만 적용해 '다른 조건이 비슷한' 점끼리만 비교한다.
    with st.expander("🎛️ 다른 변수 고정 (조건 슬라이스) — 상관된 변수 교란 줄이기", expanded=False):
        st.caption("각 그래프는 X축 변수만 자유롭게 두고, 여기서 정한 나머지 변수 조건에 맞는 점만 그립니다. "
                   "범위를 좁힐수록 다른 조건이 비슷한 점끼리만 비교합니다. 기본은 전체(제약 없음).")
        _constraints = {}
        _kcols = st.columns(2)
        for _i, _var in enumerate(cvars):
            _vn = _var["Name"]
            _col = _kcols[_i % 2]
            if "Categorical" in _var.get("Type", ""):
                _opts = sorted(df_valid[_vn].dropna().astype(str).unique().tolist())
                _sel = _col.multiselect(f"{_vn} (허용값)", _opts, default=_opts, key=f"{key_prefix}_cst_{_i}")
                _constraints[_vn] = ("cat", set(_sel))
            else:
                _s = pd.to_numeric(df_valid[_vn], errors="coerce").dropna()
                if _s.empty or _s.min() == _s.max():
                    _col.caption(f"{_vn}: 값이 1종뿐이라 고정 불필요")
                    _constraints[_vn] = ("num", None, None)
                else:
                    _lo, _hi = float(_s.min()), float(_s.max())
                    _r = _col.slider(f"{_vn} 범위", _lo, _hi, (_lo, _hi), key=f"{key_prefix}_cst_{_i}")
                    _constraints[_vn] = ("num", _r[0], _r[1])

    def _apply_slice(_df, _axis_var):
        # 축 변수(_axis_var)를 제외한 나머지 변수 제약을 적용한 부분집합을 반환한다.
        _mask = pd.Series(True, index=_df.index)
        for _vn, _c in _constraints.items():
            if _vn == _axis_var or _vn not in _df.columns:
                continue
            if _c[0] == "cat":
                if _c[1]:
                    _mask &= _df[_vn].astype(str).isin(_c[1])
            else:
                if _c[1] is not None and _c[2] is not None:
                    _mask &= pd.to_numeric(_df[_vn], errors="coerce").between(_c[1], _c[2])
        return _df[_mask]

    _picked = st.selectbox("그래프로 볼 공정 변수 (X축)", [v["Name"] for v in cvars],
                           key=f"{key_prefix}_pickvar",
                           help="한 번에 하나씩 봐야 조건 슬라이스(다른 변수 고정) 효과를 확인하기 좋습니다.")

    for vi, var in enumerate(cvars):
        vname = var["Name"]
        if var["Name"] != _picked:
            continue
        df_slice = _apply_slice(df_valid, vname)
        style = dict(x_title=x_titles[vname], y_title=y_title, title_font_size=title_fs,
                     tick_font_size=tick_fs, line_width=line_w, show_markers=show_markers, colors=colors,
                     show_trendline=show_trendline, trendline_opacity=trendline_opacity,
                     trend_degree=trend_degree, normalize=normalize)
        _dropped = len(df_valid) - len(df_slice)
        _note = f" · 다른 변수 고정으로 {_dropped}점 제외" if _dropped > 0 else " · (다른 변수 고정 없음)"
        st.markdown(f"**{vname}** — 표시 {len(df_slice)}점{_note}")
        if df_slice.empty:
            st.warning("고정 조건에 맞는 데이터가 없습니다. 위 '🎛️ 다른 변수 고정'에서 범위를 넓혀 보세요.")
            st.divider()
            continue
        # 화면은 축소 배율로, 내보내기는 항상 원본(scale=1.0)으로 — 축소해도 비율이 같아 안 깨진다.
        fig_disp = build_variable_figure(df_slice, var, targets, style, scale=disp_scale)
        fig_full = build_variable_figure(df_slice, var, targets, style, scale=1.0)
        st.plotly_chart(fig_disp, width="content", config={
            "displaylogo": False, "responsive": False,
            "toImageButtonOptions": {"format": "png", "width": int(FIG_W * disp_scale),
                                     "height": int(FIG_H * disp_scale), "scale": 3, "filename": vname},
        })
        e1, e2, e3 = st.columns(3)
        with e1:
            _export_image_button(fig_full, fmt="png", transparent=True, filename=f"{vname}_dist",
                                 label="🖼️ PNG (투명)", btn_id=f"{key_prefix}_png_{vi}")
        with e2:
            _export_image_button(fig_full, fmt="jpeg", transparent=False, filename=f"{vname}_dist",
                                 label="📷 JPG (흰 배경)", btn_id=f"{key_prefix}_jpg_{vi}")
        with e3:
            st.download_button("📊 CSV 다운로드", data=_variable_csv(df_slice, var, targets),
                               file_name=f"{vname}_dist.csv", mime="text/csv",
                               use_container_width=True, key=f"{key_prefix}_csv_{vi}")
