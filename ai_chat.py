"""실험 데이터를 곁에 두고 대화하는 분석 도우미 (Gemini).

설계 원칙이 하나 있다. **숫자는 파이썬이 계산하고, 모델은 해석만 한다.** 언어 모델에게
원자료를 던져 놓고 평균을 내라고 시키면 틀린 수를 자신 있게 말한다. 그래서 조건별
중앙값, 반복 수, 최소 허용값 충족 여부, 직전 최적화 결과까지 전부 여기서 계산해 표로
만들어 건네고, 모델에게는 "이 표에서 무엇이 읽히는가"만 묻는다.

키 보관은 **이 앱이 어디서 돌고 있는지에 따라 달라진다.** Streamlit 은 서버에서 파이썬을
실행하므로, 공유 서버에 게시된 앱에서 키를 파일로 저장하면 그 파일은 서버 한 곳에 생겨
모든 사용자가 같은 칸을 쓰게 된다. 그래서 접속 호스트를 보고 갈라 둔다.

- **본인 PC 에서 실행 중(localhost)** — 패스프레이즈로 암호화해 이 컴퓨터에 보관한다.
- **공유 서버에 게시된 앱** — 서버에는 아무것도 남기지 않는다. 키는 그 사람의 세션
  메모리에만 있고, 다시 입력하는 수고는 **브라우저의 비밀번호 관리자**가 덜어 준다
  (크롬이 저장·자동완성할 수 있도록 입력칸에 표준 autocomplete 속성을 달아 둔다).

어느 쪽이든 복호화된 값은 세션 메모리에만 머문다. Streamlit 의 세션 상태는 사용자마다
분리되므로 다른 사용자에게 보이지 않는다. Excel 저장 경로와는 닿지 않으며, 네트워크
요청은 Gemini 한 곳으로만 나간다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

import secret_store
from analysis import target_direction, direction_label



def inject_html(snippet, height=0):
    """<style>·<script> 를 페이지에 주입한다.

    Streamlit 은 st.markdown 에서 <script> 를 제거하므로 components.html(0 높이 iframe)을
    써 왔다. 그런데 그 API 는 폐기 예고가 붙어 있어 언젠가 사라진다. 사라진 날 앱 전체가
    죽지 않도록, 없으면 st.html 로 넘어간다. 주입하는 스크립트는 window.parent.document 를
    쓰는데, 인라인으로 실행될 때는 window.parent 가 자기 자신이라 그대로 동작한다.
    """
    fn = getattr(components, "html", None)
    if fn is not None:
        return fn(snippet, height=height)
    return st.html(snippet, unsafe_allow_javascript=True)


def running_locally():
    """이 앱이 사용자 본인의 컴퓨터에서 돌고 있는지. 판단이 서지 않으면 '아니오'로 본다.

    서버에 키 파일을 만드는 쪽이 위험한 선택이므로, 애매하면 안전한 쪽(공유 서버로 간주)
    으로 기운다. 호스트 헤더는 브라우저가 보내는 값이라 위조할 수 있지만, 여기서 막으려는
    것은 공격이 아니라 **배포 환경을 착각해 서버에 키를 남기는 사고**다.
    """
    try:
        headers = st.context.headers or {}
    except Exception:
        return False
    host = str(headers.get("Host") or headers.get("host") or "").split(":")[0].lower()
    return host in ("localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0") or host.endswith(".local")


def _password_manager_hints():
    """키 입력칸에 표준 autocomplete 속성을 달아, 크롬이 저장·자동완성하게 한다.

    Streamlit 은 자체 React 컴포넌트를 그리므로 name/autocomplete 속성이 붙지 않는다.
    그 속성이 없으면 브라우저의 비밀번호 관리자가 그 칸을 비밀번호로 인식하지 못한다.
    부모 문서에 직접 손대는 방식은 이 앱이 이미 Enter 가로채기와 화면 축소에 쓰고 있다.
    """
    inject_html("""
<script>
(function() {
  try {
    var doc = window.parent.document;
    var apply = function() {
      // 이 앱의 다른 폼까지 건드리지 않도록, 떠 있는 창 안쪽으로만 범위를 좁힌다.
      var root = doc.getElementById('nbedl-chat-anchor');
      var scope = root ? root.closest('[data-testid="stVerticalBlock"]') : null;
      if (!scope) return;
      var forms = scope.querySelectorAll('[data-testid="stForm"]');
      forms.forEach(function(f) {
        var pw = f.querySelector('input[type="password"]');
        if (!pw || pw.dataset.nbedlHinted) return;
        pw.dataset.nbedlHinted = "1";
        pw.setAttribute("name", "nbedl-gemini-key");
        pw.setAttribute("autocomplete", "current-password");
        // 크롬은 아이디 칸이 함께 있어야 저장 항목을 구분한다. 보이지 않는 칸을 하나 둔다.
        if (!f.querySelector('input[name="nbedl-user"]')) {
          var u = doc.createElement("input");
          u.type = "text"; u.name = "nbedl-user"; u.autocomplete = "username";
          u.value = "gemini"; u.readOnly = true; u.tabIndex = -1;
          u.setAttribute("aria-hidden", "true");
          u.style.cssText = "position:absolute;opacity:0;height:0;width:0;border:0;padding:0;";
          f.prepend(u);
        }
      });
    };
    apply();
    new window.parent.MutationObserver(apply).observe(doc.body, {childList: true, subtree: true});
  } catch (err) { /* 무시 */ }
})();
</script>
""", height=0)


def _md(df):
    """표를 마크다운으로. tabulate 가 없는 환경에서도 깨지지 않게 폴백을 둔다."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


# 이 앱이 키를 실어 보낼 수 있는 유일한 곳. 다른 주소로는 요청하지 않는다.
API_HOST = "generativelanguage.googleapis.com"
API_BASE = f"https://{API_HOST}/v1beta"
SECRET_NAME = "gemini_api_key"
SESSION_KEY = "_gemini_key_plain"      # 복호화된 키가 잠시 머무는 자리 (위젯 key 아님)
HISTORY_KEY = "gemini_chat_history"
MODEL_KEY = "gemini_model_name"
# 기본 모델. 목록은 API 에 물어서 채우지만, 그중 어느 것을 미리 골라 둘지는 정해 둔다.
# 앞에 있는 것부터 찾아 실제로 쓸 수 있는 첫 번째를 고른다 — 모델이 물갈이되어 이름이
# 사라져도 다음 것으로 자연스럽게 내려간다.
MODEL_PREFERENCE = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]
DEFAULT_MODEL = MODEL_PREFERENCE[0]
REQUEST_TIMEOUT = 90

SYSTEM_PROMPT = """당신은 페로브스카이트/실리콘 듀얼모드 광검출기를 연구하는 대학원생의 실험 데이터 분석을 돕습니다.

지켜야 할 것:
1. 아래에 주어진 표의 숫자만 쓰십시오. 표에 없는 값을 추정하거나 지어내지 마십시오. 필요한 값이 표에 없으면 "그 값은 지금 주어진 표에 없습니다"라고 말하고, 어떤 계산을 하면 되는지 알려 주십시오.
2. 새로 산술 계산을 하지 마십시오. 비교와 순위, 경향 읽기, 해석, 실험 설계 제안이 당신의 역할입니다.
3. 표본 수를 항상 함께 보십시오. 반복 수가 3 미만인 조건의 중앙값은 흔들린다는 점을 지적하십시오.
4. 평균과 중앙값이 크게 다른 조건이 보이면 그 차이가 이상치 때문일 수 있다고 짚으십시오.
5. 조건마다 배치가 다르면 조건 효과와 배치 효과가 섞인다는 점(교란)을 염두에 두십시오.
6. 확신할 수 없는 것은 확신할 수 없다고 말하십시오. 추측을 사실처럼 쓰지 마십시오.
7. 한국어로, 완성된 문장으로 답하십시오. 명사구로 문장을 끝내지 마십시오. 표가 도움이 되면 표를 쓰십시오.
"""


# ---------------------------------------------------------------------------
# Gemini 호출
# ---------------------------------------------------------------------------

def _headers(api_key):
    # 키를 URL 질의 문자열이 아니라 헤더에 싣는다. URL 은 로그·히스토리·리퍼러로 새기 쉽다.
    return {"x-goog-api-key": api_key, "Content-Type": "application/json"}


def list_models(api_key):
    """generateContent 를 지원하는 모델 이름 목록. 모델명이 바뀌어도 따라가도록 API 에 묻는다."""
    r = requests.get(f"{API_BASE}/models", headers=_headers(api_key), timeout=30)
    r.raise_for_status()
    out = []
    for m in r.json().get("models", []):
        if "generateContent" in (m.get("supportedGenerationMethods") or []):
            out.append(m["name"].split("/")[-1])
    # 가볍고 값싼 모델을 앞으로. 이름이 바뀌어도 단순 정렬로 폴백된다.
    out.sort(key=lambda n: (0 if "flash" in n else 1, "preview" in n or "exp" in n, n))
    return out


def ask(api_key, model, history, context_md):
    """history 는 [{'role': 'user'|'assistant', 'content': str}, ...]."""
    contents = [{"role": "user" if h["role"] == "user" else "model",
                 "parts": [{"text": h["content"]}]} for h in history]
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + "\n\n# 지금 화면의 데이터\n\n" + context_md}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.2},
    }
    url = f"{API_BASE}/models/{model}:generateContent"
    assert url.startswith(API_BASE)
    r = requests.post(url, headers=_headers(api_key), json=body, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(secret_store.scrub(f"{r.status_code} {r.text[:400]}", api_key))
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        fb = data.get("promptFeedback", {})
        raise RuntimeError(f"응답이 비어 있습니다. {fb}")
    parts = cands[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip() or "(빈 응답)"


# ---------------------------------------------------------------------------
# 모델에게 건넬 데이터 (숫자는 여기서 다 계산한다)
# ---------------------------------------------------------------------------

def build_context(config_vars, target_vars, max_rows=40):
    df = st.session_state.get("df_data")
    if df is None or df.empty:
        return "등록된 데이터가 없습니다."
    cfg = [v["Name"] for v in config_vars if v.get("Name") and v["Name"] in df.columns]
    tgts = [tv for tv in target_vars if tv.get("Name") and tv["Name"] in df.columns]
    valid = df[df["학습_적용"] == True] if "학습_적용" in df.columns else df  # noqa: E712

    from data_manage import AGG_LABEL
    from analysis import COMPOSITE_MODES
    _agg = st.session_state.get("mobo_copt_agg", "median")
    _mode = st.session_state.get("mobo_copt_mode", "geometric")
    out = [f"학습 적용 {len(valid)}행 / 전체 {len(df)}행. 공정 변수: {', '.join(cfg) or '없음'}.",
           f"사용자가 고른 대표값: {AGG_LABEL.get(_agg, _agg).split(' (')[0]} · "
           f"종합 방식: {dict(COMPOSITE_MODES).get(_mode, _mode).split(' (')[0]}.",
           "아래 '조건별 대표값' 표는 중앙값과 평균을 모두 담고 있으니, 둘이 갈리는 조건은 "
           "대표값 선택에 따라 순위가 뒤집힐 수 있다는 점을 함께 보십시오.", ""]

    out.append("## 목표 지표 정의")
    rows = []
    for tv in tgts:
        raw = str(st.session_state.get(f"mobo_floor_{tv['Name']}", "") or "").strip()
        rows.append({"목표": tv["Name"], "방향": direction_label(tv),
                     "단위": tv.get("Unit") or "", "최소 허용값": raw or "(없음)",
                     "계산 포함": "예" if st.session_state.get(f"mobo_incl_{tv['Name']}", True) else "아니오"})
    out.append(_md(pd.DataFrame(rows)) if rows else "(없음)")
    out.append("")

    if cfg and tgts and not valid.empty:
        names = [tv["Name"] for tv in tgts]
        work = valid.copy()
        for n in names:
            work[n] = pd.to_numeric(work[n], errors="coerce")
        g = work.groupby(cfg, dropna=False, sort=True)
        med, mean = g[names].median(), g[names].mean()
        tbl = med.round(6).reset_index()
        tbl.insert(len(cfg), "반복수", g.size().values)
        out.append("## 조건별 중앙값 (같은 조건 반복 시료)")
        out.append(_md(tbl.head(max_rows)))
        out.append("")
        # 평균이 중앙값에서 크게 벗어난 칸은 이상치 신호다. 모델이 놓치지 않게 따로 짚어 준다.
        flags = []
        for n in names:
            ratio = (mean[n] / med[n].replace(0, np.nan)).abs()
            for idx in ratio.index[(ratio > 1.5) | (ratio < 0.67)]:
                flags.append({"조건": str(idx), "목표": n,
                              "중앙값": round(float(med[n].loc[idx]), 6),
                              "평균": round(float(mean[n].loc[idx]), 6)})
        out.append("## 조건별 평균 (같은 조건 반복 시료)")
        mtbl = mean.round(6).reset_index()
        mtbl.insert(len(cfg), "반복수", g.size().values)
        out.append(_md(mtbl.head(max_rows)))
        out.append("")
        if flags:
            out.append("## 평균이 중앙값에서 1.5배 이상 벗어난 칸 (이상치 의심)")
            out.append(_md(pd.DataFrame(flags).head(max_rows)))
            out.append("")

    if "학습_적용" in df.columns:
        n_excl = int((~df["학습_적용"].astype(bool)).sum())
        if n_excl:
            ex = df[~df["학습_적용"].astype(bool)]
            nm = ex["샘플명"].astype(str).tolist() if "샘플명" in ex.columns else []
            out.append(f"## 학습에서 제외된 행 {n_excl}개")
            out.append(", ".join(nm[:max_rows]) or "(이름 없음)")
            out.append("")

    res = st.session_state.get("mobo_ai_result")
    if res:
        out.append("## 직전 다중목표 최적화(MOBO) 결과")
        rows = []
        for i, (pt, pred) in enumerate(zip(res["candidates"], res["predicted_Y"])):
            row = {"후보": i + 1}
            for var, val in zip(config_vars, pt):
                row[var["Name"]] = round(val, 4) if isinstance(val, float) else val
            for tv, p in zip(res["sel_tvs"], pred):
                row["예측 " + tv["Name"]] = round(float(p), 6)
            ei = (res.get("ei_info") or {}).get("items")
            if ei and i < len(ei):
                row["기대개선량"] = round(float(ei[i]["value"]), 6)
            rows.append(row)
        out.append(_md(pd.DataFrame(rows)))
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------

def _render_key_panel():
    """키 잠금 해제 / 저장 / 삭제. 평문 키는 화면에도 위젯 상태에도 남기지 않는다."""
    local = running_locally()
    have = bool(st.session_state.get(SESSION_KEY))
    with st.expander("🔑 Gemini API 키" + (" — 잠금 해제됨" if have else ""), expanded=not have):
        if have:
            st.success(f"이 세션에서 사용 중 · 키 {secret_store.mask(st.session_state[SESSION_KEY])}")
            cols = st.columns(2 if local and secret_store.store_exists() else 1)
            if cols[0].button("🔒 세션에서 내리기", use_container_width=True):
                st.session_state.pop(SESSION_KEY, None)
                st.rerun()
            if local and secret_store.store_exists():
                if cols[1].button("🗑️ 이 컴퓨터에 저장된 키 삭제", use_container_width=True):
                    secret_store.forget_secret(SECRET_NAME)
                    st.session_state.pop(SESSION_KEY, None)
                    st.rerun()
            return

        _password_manager_hints()

        if local:
            _render_local_key_form()
        else:
            _render_shared_key_form()


def _render_shared_key_form():
    """공유 서버에 게시된 앱. 서버에는 아무것도 남기지 않는다."""
    st.caption(
        "이 앱은 **서버에서 실행 중**입니다. 그래서 키를 서버에 저장하지 않습니다 — 저장하면 "
        "그 파일이 서버 한 곳에 생겨 이 앱을 쓰는 모든 사람이 같은 칸을 쓰게 됩니다.  \n"
        "입력하신 키는 **지금 이 브라우저 세션에만** 머물고, 탭을 닫으면 사라집니다. "
        "Streamlit 의 세션은 사용자마다 분리되어 있어 다른 사람에게 보이지 않습니다.  \n"
        "💡 **다시 입력하는 수고는 브라우저에 맡기세요.** 아래 칸은 크롬·엣지의 비밀번호 "
        "관리자가 인식하도록 되어 있어, 처음 한 번 넣으면 저장할지 물어보고 다음부터 "
        "자동으로 채워 줍니다. 키는 그 브라우저 안에만 저장되고 서버로는 가지 않습니다.  \n"
        "키 발급은 Google AI Studio(aistudio.google.com/apikey)에서 합니다. "
        "**키는 개인 것이므로 다른 사람과 공유하지 마세요.**"
    )
    with st.form("gemini_session_key", border=False):
        k = st.text_input("Gemini API 키", type="password", placeholder="AIza…",
                          autocomplete="current-password")
        if st.form_submit_button("▶️ 이 세션에서 사용", type="primary", use_container_width=True):
            if not k.strip():
                st.error("키를 입력하세요.")
            else:
                st.session_state[SESSION_KEY] = k.strip()
                st.rerun()


def _render_local_key_form():
    """본인 컴퓨터에서 실행 중. 패스프레이즈로 암호화해 이 컴퓨터에 보관할 수 있다."""
    if not secret_store.CRYPTO_AVAILABLE:
        st.warning("`cryptography` 패키지가 없어 이번 세션에만 키를 쓸 수 있습니다. "
                   "이 컴퓨터에 보관하시려면 `pip install cryptography` 후 앱을 다시 시작하세요.")
        _render_shared_key_form()
        return
    st.caption(
        "이 앱이 **본인 컴퓨터에서 실행 중**이라, 키를 패스프레이즈로 암호화해 이 컴퓨터에 "
        f"보관할 수 있습니다. 보관 위치: `{secret_store.store_location()}`  \n"
        "실험 데이터 Excel 파일에는 **저장되지 않습니다.** 복호화된 키는 앱이 실행 중인 동안 "
        "메모리에만 있고, 요청은 `generativelanguage.googleapis.com` 한 곳으로만 나갑니다.  \n"
        "키 발급은 Google AI Studio(aistudio.google.com/apikey)에서 합니다."
    )
    if secret_store.store_exists():
        with st.form("gemini_unlock", border=False):
            pw = st.text_input("패스프레이즈", type="password",
                               help="키를 저장할 때 정한 패스프레이즈입니다. 키 자체가 아닙니다.")
            if st.form_submit_button("🔓 잠금 해제", type="primary", use_container_width=True):
                try:
                    st.session_state[SESSION_KEY] = secret_store.load_secret(SECRET_NAME, pw)
                    st.rerun()
                except secret_store.SecretError as e:
                    st.error(str(e))
        st.caption("패스프레이즈를 잊으셨다면 아래에서 키를 다시 등록하세요(기존 것은 덮어씁니다).")

    with st.form("gemini_save", border=False):
        st.markdown("**키 등록 / 다시 등록**")
        k = st.text_input("Gemini API 키", type="password", placeholder="AIza…")
        p1 = st.text_input("사용할 패스프레이즈", type="password")
        p2 = st.text_input("패스프레이즈 확인", type="password")
        c1, c2 = st.columns(2)
        save = c1.form_submit_button("💾 암호화해서 이 컴퓨터에 저장", use_container_width=True)
        once = c2.form_submit_button("▶️ 이번 세션에만 사용", use_container_width=True)
        if save or once:
            if not k.strip():
                st.error("키를 입력하세요.")
            elif once:
                st.session_state[SESSION_KEY] = k.strip()
                st.rerun()
            elif p1 != p2:
                st.error("두 패스프레이즈가 다릅니다.")
            elif len(p1) < 4:
                st.error("패스프레이즈는 4자 이상으로 정하세요.")
            else:
                try:
                    secret_store.save_secret(SECRET_NAME, k.strip(), p1)
                    st.session_state[SESSION_KEY] = k.strip()
                    st.rerun()
                except secret_store.SecretError as e:
                    st.error(str(e))


def render_chat(config_vars, target_vars):
    _render_key_panel()
    api_key = st.session_state.get(SESSION_KEY)
    if not api_key:
        st.info("키를 등록하고 잠금을 해제하면 지금 화면의 데이터를 놓고 대화할 수 있습니다.")
        return

    # 모델 목록은 API 에 물어서 채운다. 모델명이 바뀌어도 코드를 고칠 필요가 없다.
    if "gemini_model_list" not in st.session_state:
        try:
            st.session_state.gemini_model_list = list_models(api_key)
        except Exception as e:
            st.session_state.gemini_model_list = []
            st.warning("모델 목록을 가져오지 못했습니다: " + secret_store.scrub(str(e)[:200], api_key))
    models = st.session_state.gemini_model_list
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    if models:
        if MODEL_KEY not in st.session_state or st.session_state[MODEL_KEY] not in models:
            st.session_state[MODEL_KEY] = next((m for m in MODEL_PREFERENCE if m in models), models[0])
        c1.selectbox("모델", models, key=MODEL_KEY,
                     help="flash 계열이 빠르고 저렴합니다. 숫자가 클수록 새 모델이고, "
                          "긴 추론이 필요하면 pro 계열을 쓰세요.")
    else:
        c1.text_input("모델 이름", key=MODEL_KEY, placeholder=f"예: {DEFAULT_MODEL}")
    if c2.button("🧹 대화 비우기", use_container_width=True):
        st.session_state[HISTORY_KEY] = []
        st.rerun()

    context_md = build_context(config_vars, target_vars)
    with st.expander("📋 모델에게 함께 보내는 데이터 (직접 확인)"):
        st.caption("이 내용만 전송됩니다. 원자료 전체가 아니라 조건별로 집계된 표입니다.")
        st.code(context_md, language="markdown")

    st.caption(
        "💡 예시 질문: 「조건별로 두 모드가 함께 성립하는지 비교해 줘」 · "
        "「평균과 중앙값이 크게 다른 조건이 어디야」 · 「다음 배치를 어떻게 설계하면 좋을까」"
    )

    history = st.session_state.setdefault(HISTORY_KEY, [])
    for h in history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])

    # st.chat_input 은 버전에 따라 컨테이너 안에서 거부될 수 있다. 그럴 때는 일반
    # 입력창으로 조용히 내려앉아, 스트림릿 버전 때문에 탭 전체가 죽지 않게 한다.
    try:
        prompt = st.chat_input("데이터에 대해 물어보세요")
    except Exception:
        with st.form("gemini_ask", border=False, clear_on_submit=True):
            prompt = st.text_input("질문", label_visibility="collapsed",
                                   placeholder="데이터에 대해 물어보세요")
            if not st.form_submit_button("보내기", type="primary"):
                prompt = None
    if not prompt:
        return
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                answer = ask(api_key, st.session_state.get(MODEL_KEY) or DEFAULT_MODEL,
                             history, context_md)
            except Exception as e:
                answer = "요청에 실패했습니다: " + secret_store.scrub(str(e)[:500], api_key)
        st.markdown(answer)
    history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# 떠 있는 채팅창 (좌측 하단)
# ---------------------------------------------------------------------------
# 탭 하나를 통째로 쓰면 데이터를 보면서 물어볼 수가 없다. 그래서 화면 왼쪽 아래에
# 동그란 단추로 떠 있다가, 누르면 그 자리에서 펼쳐지도록 한다. 오른쪽 아래는
# Streamlit 자체 메뉴가 쓰므로 왼쪽이다.
#
# 매뉴얼 서랍과 달리 **뒤를 흐리지 않는다.** 이 창은 데이터를 가리려고 여는 것이
# 아니라 데이터를 보면서 쓰려고 여는 것이므로, 뒤가 읽혀야 한다. 그래서 덮개(backdrop)를
# 두지 않고, 창이 차지하는 사각형 밖은 그대로 클릭된다.
#
# 위치 지정은 CSS 한 줄로 끝나지 않는다. Streamlit 의 DOM 구조(감싸는 div 의 깊이)는
# 버전마다 달라서 :has() 선택자로 조상을 짚으면 쉽게 깨진다. 그래서 눈에 보이지 않는
# 표식을 하나 심고, 자바스크립트로 그 표식의 조상을 찾아 클래스를 붙인다. 리런 때마다
# DOM 이 갈리므로 MutationObserver 로 다시 붙인다.

# ---------------------------------------------------------------------------
# 마스코트
# ---------------------------------------------------------------------------
# 이모지는 글꼴에 따라 모양이 달라지고 크기를 키워도 존재감이 없다. 말풍선이자 로봇
# 얼굴인 도형을 직접 그려 두면 어느 환경에서나 같은 모양으로, 원하는 크기로 나온다.
# 선만으로 그렸으므로 작은 크기에서도 뭉개지지 않는다. 색은 두 벌만 둔다 — 주황 단추
# 위에 얹는 흰색, 밝은 바탕에 놓는 주황색.

MASCOT_WHITE = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij4KICA8ZyBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4KICAgIDxwYXRoIGQ9Ik0yNCA2IFYxMCIvPgogICAgPHJlY3QgeD0iNC44IiB5PSIxMCIgd2lkdGg9IjM4LjQiIGhlaWdodD0iMjQuNSIgcng9IjguNSIvPgogICAgPHBhdGggZD0iTTE4LjYgMjUuMiBxNS40IDQuOCAxMC44IDAiLz4KICA8L2c+CiAgPHBhdGggZD0iTTE0LjggMzMuNiBoOS42IGwtOS42IDEwIHoiIGZpbGw9IiNmZmZmZmYiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyLjQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KICA8Y2lyY2xlIGN4PSIyNCIgY3k9IjMuOSIgcj0iMi45IiBmaWxsPSIjZmZmZmZmIi8+CiAgPGNpcmNsZSBjeD0iMTguMiIgY3k9IjE5LjIiIHI9IjMiIGZpbGw9IiNmZmZmZmYiLz4KICA8Y2lyY2xlIGN4PSIyOS44IiBjeT0iMTkuMiIgcj0iMyIgZmlsbD0iI2ZmZmZmZiIvPgo8L3N2Zz4="
MASCOT_ORANGE = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij4KICA8ZyBmaWxsPSJub25lIiBzdHJva2U9IiNlZDU0MmIiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4KICAgIDxwYXRoIGQ9Ik0yNCA2IFYxMCIvPgogICAgPHJlY3QgeD0iNC44IiB5PSIxMCIgd2lkdGg9IjM4LjQiIGhlaWdodD0iMjQuNSIgcng9IjguNSIvPgogICAgPHBhdGggZD0iTTE4LjYgMjUuMiBxNS40IDQuOCAxMC44IDAiLz4KICA8L2c+CiAgPHBhdGggZD0iTTE0LjggMzMuNiBoOS42IGwtOS42IDEwIHoiIGZpbGw9IiNlZDU0MmIiIHN0cm9rZT0iI2VkNTQyYiIgc3Ryb2tlLXdpZHRoPSIyLjQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KICA8Y2lyY2xlIGN4PSIyNCIgY3k9IjMuOSIgcj0iMi45IiBmaWxsPSIjZWQ1NDJiIi8+CiAgPGNpcmNsZSBjeD0iMTguMiIgY3k9IjE5LjIiIHI9IjMiIGZpbGw9IiNlZDU0MmIiLz4KICA8Y2lyY2xlIGN4PSIyOS44IiBjeT0iMTkuMiIgcj0iMyIgZmlsbD0iI2VkNTQyYiIvPgo8L3N2Zz4="

CHAT_ANCHOR_ID = "nbedl-chat-anchor"
OPEN_KEY = "nbedl_chat_open"

_CHAT_CSS = """
<script>
(function() {
  try {
    var doc = window.parent.document, ID = 'nbedl-chat-style';
    var st = doc.getElementById(ID);
    if (!st) { st = doc.createElement('style'); st.id = ID; doc.head.appendChild(st); }
    st.textContent = [
      /* 떠 있는 창 자체 */
      '.nbedl-chat-panel{position:fixed !important;left:20px;bottom:20px;z-index:9990;',
      '  width:auto !important;}',
      '.nbedl-chat-panel[data-open="1"]{width:min(470px,calc(100vw - 40px)) !important;',
      '  max-height:min(78vh,780px);overflow-y:auto;overflow-x:hidden;',
      '  background:var(--background-color,#ffffff);',
      '  border:1px solid rgba(49,51,63,.18);border-radius:18px;',
      '  box-shadow:0 14px 48px rgba(0,0,0,.22);padding:14px 16px 10px;}',

      /* 닫혀 있을 때 = 알약 모양 단추. 이모지 대신 직접 그린 마스코트를 왼쪽에 얹는다. */
      '.nbedl-chat-panel[data-open="0"] .stButton>button{',
      '  height:62px;padding:0 26px 0 68px;border-radius:31px;border:none !important;',
      '  background-image:url("%MASCOT%"),linear-gradient(135deg,#ed542b,#f68b21) !important;',
      '  background-repeat:no-repeat,no-repeat;',
      '  background-position:18px center,center;',
      '  background-size:38px 38px,100% 100%;',
      '  color:#fff !important;font-size:1.02rem;font-weight:800;letter-spacing:.01em;',
      '  white-space:nowrap;transition:transform .16s ease, box-shadow .16s ease;',
      '  animation:nbedlChatPulse 2.6s ease-out 4;}',
      '.nbedl-chat-panel[data-open="0"] .stButton>button:hover{',
      '  transform:translateY(-2px) scale(1.03);animation:none;',
      '  box-shadow:0 12px 34px rgba(237,84,43,.5) !important;}',
      '.nbedl-chat-panel[data-open="0"] .stButton>button p{',
      '  font-size:1.02rem !important;font-weight:800 !important;color:#fff !important;}',
      /* 처음 몇 번만 파문이 퍼진다. 계속 움직이면 곧 거슬린다. */
      '@keyframes nbedlChatPulse{',
      '  0%{box-shadow:0 8px 26px rgba(0,0,0,.26),0 0 0 0 rgba(237,84,43,.55);}',
      '  70%{box-shadow:0 8px 26px rgba(0,0,0,.26),0 0 0 18px rgba(237,84,43,0);}',
      '  100%{box-shadow:0 8px 26px rgba(0,0,0,.26),0 0 0 0 rgba(237,84,43,0);}}',

      /* 창 안은 여백을 죄어 좁은 폭에서도 읽히게 */
      '.nbedl-chat-panel[data-open="1"] [data-testid="stVerticalBlock"]{gap:.45rem;}',
      '.nbedl-chat-panel .stChatMessage{padding:.4rem .6rem;}',
      '.nbedl-chat-panel p,.nbedl-chat-panel li{font-size:.88rem;}',
      /* 단추가 본문 마지막 줄을 가리지 않도록 아래 여백 */
      '[data-testid="stMain"] .block-container{padding-bottom:120px;}'
    ].join('');

    var mark = function() {
      var a = doc.getElementById('%ANCHOR%');
      if (!a) return;
      var block = a.closest('[data-testid="stVerticalBlock"]');
      if (!block) return;
      doc.querySelectorAll('.nbedl-chat-panel').forEach(function(el) {
        if (el !== block) el.classList.remove('nbedl-chat-panel');
      });
      block.classList.add('nbedl-chat-panel');
      block.setAttribute('data-open', a.dataset.open || '0');
    };
    mark();
    if (!window.parent.__nbedlChatObs) {
      window.parent.__nbedlChatObs = new window.parent.MutationObserver(mark);
      window.parent.__nbedlChatObs.observe(doc.body, {childList: true, subtree: true});
    }
  } catch (err) { /* 무시 */ }
})();
</script>
"""


def render_floating_chat(config_vars, target_vars):
    """화면 왼쪽 아래에 떠 있는 분석 도우미. 탭 밖에서 한 번만 호출한다."""
    is_open = bool(st.session_state.get(OPEN_KEY, False))
    box = st.container()
    with box:
        st.markdown(
            f'<div id="{CHAT_ANCHOR_ID}" data-open="{"1" if is_open else "0"}" '
            f'style="height:0;overflow:hidden;"></div>',
            unsafe_allow_html=True)
        if is_open:
            head, shut = st.columns([5, 1], vertical_alignment="center")
            head.markdown(
                "<div style='display:flex;align-items:center;gap:9px;'>"
                f"<img src='{MASCOT_ORANGE}' width='30' height='30' alt=''>"
                "<div><div style='font-weight:800;font-size:1.02rem;line-height:1.2;'>분석 도우미</div>"
                "<div style='font-size:.73rem;opacity:.6;line-height:1.25;'>"
                "숫자는 앱이 계산해 표로 건네고, 모델은 해석만 합니다.</div></div></div>",
                unsafe_allow_html=True)
            if shut.button("✕", key="nbedl_chat_close", help="닫기"):
                st.session_state[OPEN_KEY] = False
                st.rerun()
            st.divider()
            render_chat(config_vars, target_vars)
        else:
            if st.button("분석 도우미에게 물어보기", key="nbedl_chat_open_btn",
                         help="지금 화면의 데이터를 놓고 대화합니다"):
                st.session_state[OPEN_KEY] = True
                st.rerun()
    inject_html(_CHAT_CSS.replace("%ANCHOR%", CHAT_ANCHOR_ID)
                         .replace("%MASCOT%", MASCOT_WHITE))
