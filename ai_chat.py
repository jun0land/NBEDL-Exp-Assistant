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

import base64
import re
import time

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
# 목록을 못 가져왔을 때 쓰는 대비책. 앞에서부터 실제로 쓸 수 있는 첫 번째를 고르므로,
# 모델이 물갈이되어 이름이 사라져도 다음 것으로 자연스럽게 내려간다. 평소에는 아래
# curate_models 가 계정의 실제 목록에서 세 등급을 뽑아 오므로 이 목록은 쓰이지 않는다.
MODEL_PREFERENCE = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]
DEFAULT_MODEL = MODEL_PREFERENCE[0]
REQUEST_TIMEOUT = 90
ATTEMPTS_FIRST = 2      # 처음 고른 모델에 몇 번까지 다시 물어볼지
ATTEMPTS_FALLBACK = 1   # 대체 모델은 한 번씩만. 여러 개를 빠르게 훑는 편이 낫다
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
IMAGE_KEY = "gemini_chat_images"

# 대화에 쓸 수 있는 모델만 남기는 규칙. 계정에 보이는 모델은 수십 개인데 그 대부분은
# 이미지 생성·음성·임베딩처럼 여기서 쓸 일이 없는 것들이다. 모델을 잘 모르는 사람에게
# 긴 목록은 도움이 아니라 부담이므로, 하나의 축 위에 세 등급만 놓는다.
#
#     빠르고 저렴  ←  Flash-Lite   ·   Flash(기본)   ·   Pro  →  똑똑하고 느림
#
# 등급마다 계정에 있는 것 중 가장 새 판을 뽑는다. 이름 규칙으로 거르므로 새 모델이
# 나와도 그대로 따라가고, preview·exp 처럼 꼬리표가 붙은 것은 규칙에 걸려 빠진다.
_MODEL_RE = re.compile(r"^gemini-(\d+(?:\.\d+)?)-(flash-lite|flash|pro)$")

MODEL_TIERS = [
    ("flash-lite", "가장 빠르고 저렴 · 간단한 확인용"),
    ("flash", "균형 · 기본값"),
    ("pro", "가장 똑똑함 · 느리고 비쌈"),
]
DEFAULT_TIER = "flash"


def curate_models(names):
    """등급마다 가장 새 판 하나씩. [(모델명, 화면에 보일 설명), ...] 로 돌려준다."""
    best = {}
    for n in names:
        m = _MODEL_RE.match(n)
        if not m:
            continue
        ver, tier = float(m.group(1)), m.group(2)
        if tier not in best or ver > best[tier][0]:
            best[tier] = (ver, n)
    out = [(best[t][1], note) for t, note in MODEL_TIERS if t in best]
    return out or [(n, "") for n in names[:5]]


def default_model(curated):
    """기본으로 골라 둘 모델. 중간 등급(Flash)을 먼저 보고, 없으면 있는 것 중 첫째."""
    if not curated:
        return DEFAULT_MODEL
    for name, note in curated:
        if _tier_of(name) == DEFAULT_TIER:
            return name
    return curated[0][0]


def _tier_of(name):
    m = _MODEL_RE.match(name)
    return m.group(2) if m else ""


def _ver_of(name):
    m = _MODEL_RE.match(name)
    return float(m.group(1)) if m else -1.0


# 등급을 갈아타야 할 때의 순서. 답이 아예 안 오는 것보다는 낫다는 이유로 무조건 가벼운
# 쪽으로 내려가면, 판단이 섞인 질문에서 답의 질이 눈에 띄게 떨어진다. 그래서 Flash 가
# 막히면 Lite 가 아니라 Pro 를 먼저 본다. 다만 이것은 마지막 수단이다 (아래 참조).
FALLBACK_BY_TIER = {
    "flash": ["pro", "flash-lite"],
    "flash-lite": ["flash", "pro"],
    "pro": ["flash", "flash-lite"],
}

MAX_FALLBACKS = 3   # 한 질문에 대체 모델을 몇 개까지 시도할지


def fallback_order(chosen, curated, all_names=()):
    """혼잡할 때 넘어갈 순서.

    **먼저 같은 등급의 한 세대 아래로 내려간다.** 3.8-flash 가 막히면 3.7 → 3.6 →
    3.5-flash 순이다. 같은 계열의 이전 판은 성능과 값이 가장 가까운 대체재이므로,
    등급을 갈아타는 것보다 사용자가 기대한 답에 가깝다. 목록에는 등급마다 최신판
    하나만 보이지만, 대체용으로는 계정에 있는 이전 판들도 전부 쓴다.

    같은 등급이 모두 막힌 뒤에야 다른 등급으로 넘어간다.
    """
    tier, ver = _tier_of(chosen), _ver_of(chosen)
    siblings = sorted(
        (n for n in all_names
         if _MODEL_RE.match(n) and _tier_of(n) == tier and _ver_of(n) < ver),
        key=_ver_of, reverse=True)

    out = list(siblings)
    avail = {_tier_of(n): n for n, _ in curated if n != chosen}
    for t in (FALLBACK_BY_TIER.get(tier) or FALLBACK_BY_TIER[DEFAULT_TIER]):
        if t in avail and avail[t] not in out:
            out.append(avail[t])
    out += [n for n, _ in curated if n != chosen and n not in out]
    return out[:MAX_FALLBACKS]


class GeminiError(RuntimeError):
    """사람이 읽을 수 있게 다듬은 API 오류."""


def _friendly_error(status, text):
    t = (text or "")[:400]
    if status in (429,):
        return ("요청 한도를 넘었습니다(429). 잠시 뒤 다시 시도하시거나, "
                "Google AI Studio 에서 이 키의 한도를 확인해 주세요.")
    if status in (500, 502, 503, 504):
        return ("지금 이 모델에 요청이 몰려 있습니다(%d). 잠시 뒤 다시 시도하시거나 "
                "위에서 다른 모델을 골라 보세요." % status)
    if status == 403:
        return "이 키로는 해당 모델을 쓸 수 없습니다(403). 키 권한이나 모델 이름을 확인해 주세요."
    if status == 400 and "API_KEY" in t.upper():
        return "API 키가 올바르지 않습니다(400). 키를 다시 등록해 주세요."
    return f"요청이 거부되었습니다({status}). {t}"

SYSTEM_PROMPT = """당신은 페로브스카이트/실리콘 듀얼모드 광검출기를 연구하는 대학원생의 실험 데이터 분석을 돕습니다.

# 내용에 관한 규칙

1. 아래에 주어진 표의 숫자만 쓰십시오. 표에 없는 값을 추정하거나 지어내지 마십시오. 필요한 값이 표에 없으면 "그 값은 지금 주어진 표에 없습니다"라고 말하고, 어떤 계산을 하면 되는지 알려 주십시오.
2. 새로 산술 계산을 하지 마십시오. 비교와 순위, 경향 읽기, 해석, 실험 설계 제안이 당신의 역할입니다.
3. 표본 수를 항상 함께 보십시오. 반복 수가 3 미만인 조건의 중앙값은 흔들린다는 점을 지적하십시오.
4. 평균과 중앙값이 크게 다른 조건이 보이면 그 차이가 이상치 때문일 수 있다고 짚으십시오.
5. 조건마다 배치가 다르면 조건 효과와 배치 효과가 섞인다는 점(교란)을 염두에 두십시오.
6. 확신할 수 없는 것은 확신할 수 없다고 말하십시오. 추측을 사실처럼 쓰지 마십시오.
7. 선택지가 갈리는 문제에서 당신이 대신 고르지 마십시오. 각 선택지와 그 결과를 보여 주고, 판단은 사용자에게 넘기십시오.

# 형식에 관한 규칙

가장 흔한 실패는 답이 한 덩어리 줄글로 나오는 것입니다. 아래를 지키십시오.

1. **결론을 맨 앞에 한두 문장으로** 씁니다. 근거는 그 뒤에 펼칩니다.
2. **한 문단은 세 줄을 넘기지 않습니다.** 말이 바뀌면 문단을 나눕니다.
3. **근거가 둘 이상이면 목록으로** 나열하고, 각 항목은 굵은 글씨 라벨로 시작합니다.
4. **숫자를 셋 이상 견주면 표로** 만듭니다. 줄글 안에 숫자를 늘어놓지 마십시오.
5. 소제목(`###`)은 답이 길어질 때만 씁니다. 짧은 답에는 쓰지 않습니다.
6. 조건과 목표의 이름은 표에 적힌 그대로 씁니다. 줄여 쓰거나 바꿔 부르지 마십시오.
7. 마지막에 **「그래서 무엇을 하면 되는가」를 한 줄**로 덧붙입니다. 덧붙일 말이 없으면 생략합니다.

## 답의 모양 (예시)

> 세 대표값 모두에서 13초가 1위이므로, 이 순위는 이상치 하나에 좌우되지 않습니다.
>
> - **1위는 흔들리지 않습니다.** 중앙값·절사평균·평균에서 모두 13초가 1위입니다.
> - **2위와 4위는 뒤집힙니다.** 28초와 15초의 순위가 대표값에 따라 바뀝니다.
>
> | 조건 | 중앙값 | 평균 |
> |---|---|---|
> | 13 | 0.695 | 0.610 |
> | 15 | 0.474 | 0.475 |
>
> 28초와 15초 사이의 우열은 지금 데이터로 가릴 수 없으므로, 그 둘을 같은 배치에서 다시 만들어 보십시오.

# 문장

한국어로, 완성된 문장으로 답하십시오. 명사구로 문장을 끝내지 마십시오. 엠대시(—)는 앞뒤 관계를 지나치게 함축하므로 쓰지 말고, 쉼표나 접속사로 이으십시오.
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


def ask(api_key, model, history, context_md, images=None, alternates=()):
    """한 번 물어보고 답을 돌려준다. (답, 실제로 답한 모델) 을 반환한다.

    Gemini 는 수요가 몰리면 503 을 돌려준다. 그것은 잘못 쓴 것이 아니라 잠시 기다리면
    풀리는 상태이므로, 같은 모델에 두 번까지 다시 묻고 그래도 안 되면 대체 모델로
    넘어간다. 사용자에게 오류를 그대로 던지기 전에 할 수 있는 것을 먼저 해 본다.

    images 는 [(mime, base64), ...] 이며 마지막 사용자 발화에 붙는다. Gemini 는 그림을
    읽을 수 있으므로, 그래프나 화면을 캡처해 넣으면 표와 함께 보고 답한다.
    """
    contents = []
    last = len(history) - 1
    for i, h in enumerate(history):
        role = "user" if h["role"] == "user" else "model"
        parts = [{"text": h["content"]}]
        if images and role == "user" and i == last:
            for mime, data in images:
                parts.append({"inline_data": {"mime_type": mime, "data": data}})
        contents.append({"role": role, "parts": parts})

    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + "\n\n# 지금 화면의 데이터\n\n" + context_md}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.2},
    }

    queue = [model] + [m for m in alternates if m != model]
    last_msg = "알 수 없는 오류"
    for mi, m in enumerate(queue):
        url = f"{API_BASE}/models/{m}:generateContent"
        assert url.startswith(API_BASE)
        # 처음 고른 모델에는 몇 번 더 매달리고, 대체 모델은 한 번씩만 빠르게 훑는다.
        attempts = ATTEMPTS_FIRST if mi == 0 else ATTEMPTS_FALLBACK
        for attempt in range(attempts):
            try:
                r = requests.post(url, headers=_headers(api_key), json=body,
                                  timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                last_msg = f"연결하지 못했습니다: {type(e).__name__}"
                time.sleep(1.2 * (attempt + 1))
                continue
            if r.status_code in RETRYABLE_STATUS:
                last_msg = _friendly_error(r.status_code, secret_store.scrub(r.text, api_key))
                if attempt + 1 < attempts or mi + 1 < len(queue):
                    time.sleep(1.2 * (2 ** attempt))
                continue
            if r.status_code >= 400:
                # 재시도해도 달라지지 않는 오류(키·권한·요청 형식)는 바로 알린다.
                raise GeminiError(_friendly_error(r.status_code, secret_store.scrub(r.text, api_key)))
            data = r.json()
            cands = data.get("candidates") or []
            if not cands:
                fb = data.get("promptFeedback", {})
                raise GeminiError(f"응답이 비어 있습니다. 안전 필터에 걸렸을 수 있습니다. {fb}")
            parts = cands[0].get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            return (text or "(빈 응답)"), m
    raise GeminiError(last_msg)


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

def _render_key_panel(compact=False, inner=False):
    """키 상태와 등록 폼.

    compact=True 는 '키가 이미 풀려 있으니 자리를 차지하지 말라'는 뜻이라 아무것도 그리지
    않는다. 잠그기·삭제는 설정 안(inner=True)에서만 보여 준다. 대화하러 연 창에서 키
    관리 상자가 늘 펼쳐져 있으면, 정작 쓸 일 없는 것이 화면의 절반을 먹는다.
    """
    have = bool(st.session_state.get(SESSION_KEY))
    local = running_locally()
    if have and compact:
        return
    if have and inner:
        st.caption(f"🔑 키 {secret_store.mask(st.session_state[SESSION_KEY])} 사용 중")
        cols = st.columns(2 if (local and secret_store.store_exists()) else 1)
        cols[0].button("세션에서 내리기", key="nbedl_key_lock",
                       on_click=_lock_key, use_container_width=True)
        if local and secret_store.store_exists():
            cols[1].button("이 컴퓨터에서 삭제", key="nbedl_key_forget",
                           on_click=_forget_key, use_container_width=True)
        return
    if have:
        return

    _password_manager_hints()
    if local:
        _render_local_key_form()
    else:
        _render_shared_key_form()


def _lock_key():
    st.session_state.pop(SESSION_KEY, None)


def _forget_key():
    secret_store.forget_secret(SECRET_NAME)
    st.session_state.pop(SESSION_KEY, None)


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
    """떠 있는 창의 속. 대화를 시작하기 전에도 화면 절반을 잡아먹지 않도록, 평소에는
    한 줄짜리 설정과 입력창만 보이고 나머지는 '설정' 안으로 접어 둔다."""
    locked = not st.session_state.get(SESSION_KEY)
    _render_key_panel(compact=not locked)
    api_key = st.session_state.get(SESSION_KEY)
    if not api_key:
        return

    if "gemini_model_list" not in st.session_state:
        try:
            raw = list_models(api_key)
            st.session_state.gemini_model_raw = raw          # 대체 모델을 고를 때 쓴다
            st.session_state.gemini_model_list = curate_models(raw)
        except Exception as e:
            st.session_state.gemini_model_raw = []
            st.session_state.gemini_model_list = []
            st.warning("모델 목록을 가져오지 못했습니다: " + secret_store.scrub(str(e)[:200], api_key))
    models = st.session_state.gemini_model_list
    names = [m for m, _ in models]
    notes = dict(models)

    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    if names:
        if MODEL_KEY not in st.session_state or st.session_state[MODEL_KEY] not in names:
            st.session_state[MODEL_KEY] = default_model(models)
        c1.selectbox("모델", names, key=MODEL_KEY, label_visibility="collapsed",
                     format_func=lambda n: f"{n} — {notes[n]}" if notes.get(n) else n,
                     help="왼쪽일수록 빠르고 싸며, 오른쪽일수록 똑똑하고 느립니다. "
                          "간단한 확인은 Lite, 평소에는 Flash, 답이 얕게 느껴지면 Pro 로 바꿔 보세요. "
                          "그림을 읽는 것은 세 등급 모두 됩니다.")
    else:
        c1.text_input("모델", key=MODEL_KEY, label_visibility="collapsed",
                      placeholder=f"예: {DEFAULT_MODEL}")
    c2.button("대화 비우기", key="nbedl_chat_clear", on_click=_clear_chat,
              use_container_width=True)

    context_md = build_context(config_vars, target_vars)
    with st.expander("⚙️ 전송 데이터 · 키 · 도움말"):
        st.caption(
            "**도우미는 이 앱의 화면을 직접 보지 못합니다.** 아래 표로 정리된 숫자만 전달받습니다. "
            "그래프의 모양이나 사진에 대해 물으시려면 입력창의 **📎** 로 그림을 넣어 주세요.  \n"
            "💡 예시: 「조건별로 두 모드가 함께 성립하는지 비교해 줘」 · "
            "「평균과 중앙값이 크게 다른 조건이 어디야」 · 「다음 배치를 어떻게 설계하면 좋을까」"
        )
        st.code(context_md, language="markdown")
        _render_key_panel(compact=False, inner=True)

    history = st.session_state.setdefault(HISTORY_KEY, [])
    for h in history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])

    # 그림 첨부는 입력창 안의 클립으로. 따로 칸을 두면 대화도 시작하기 전에 자리를 먹는다.
    prompt, files = None, []
    try:
        val = st.chat_input("데이터에 대해 물어보세요", accept_file="multiple",
                            file_type=["png", "jpg", "jpeg", "webp"])
        if val is not None:
            prompt = getattr(val, "text", None) if hasattr(val, "text") else str(val)
            files = list(getattr(val, "files", []) or [])
    except TypeError:
        # accept_file 을 모르는 옛 버전
        val = st.chat_input("데이터에 대해 물어보세요")
        prompt = val
    except Exception:
        with st.form("gemini_ask", border=False, clear_on_submit=True):
            prompt = st.text_input("질문", label_visibility="collapsed",
                                   placeholder="데이터에 대해 물어보세요")
            if not st.form_submit_button("보내기", type="primary"):
                prompt = None
    if not prompt and not files:
        return
    prompt = (prompt or "").strip() or "첨부한 그림을 보고 설명해 줘."

    imgs = []
    for f in files[:4]:
        raw = f.getvalue()
        if len(raw) > 6 * 1024 * 1024:
            st.warning(f"{f.name} 은 6 MB 를 넘어 건너뜁니다.")
            continue
        imgs.append((f.type or "image/png", base64.b64encode(raw).decode()))

    shown = prompt + (f"\n\n*🖼 그림 {len(imgs)}장 첨부*" if imgs else "")
    history.append({"role": "user", "content": shown})
    with st.chat_message("user"):
        st.markdown(shown)
    chosen = st.session_state.get(MODEL_KEY) or DEFAULT_MODEL
    alts = fallback_order(chosen, models, st.session_state.get("gemini_model_raw", []))
    if not alts:
        alts = [m for m in MODEL_PREFERENCE if m != chosen][:1]
    with st.chat_message("assistant"):
        with st.spinner("생각 중... (혼잡하면 다시 시도합니다)"):
            try:
                answer, used = ask(api_key, chosen, history, context_md,
                                   images=imgs, alternates=alts)
                if used != chosen:
                    answer = f"*{chosen} 이 혼잡해 **{used}** 로 답했습니다.*\n\n" + answer
            except GeminiError as e:
                answer = str(e)
            except Exception as e:
                answer = "요청에 실패했습니다: " + secret_store.scrub(str(e)[:400], api_key)
        st.markdown(answer)
    history.append({"role": "assistant", "content": answer})


def _clear_chat():
    st.session_state[HISTORY_KEY] = []


# ---------------------------------------------------------------------------
# 마스코트
# ---------------------------------------------------------------------------
# 이모지는 글꼴에 따라 모양이 달라지고 크기를 키워도 존재감이 없다. 말풍선이자 로봇
# 얼굴인 도형을 직접 그려 두면 어느 환경에서나 같은 모양으로, 원하는 크기로 나온다.
# 선만으로 그렸으므로 작은 크기에서도 뭉개지지 않는다. 색은 두 벌만 둔다 — 주황 단추
# 위에 얹는 흰색, 밝은 바탕에 놓는 주황색.

# 꼬리 삼각형의 윗변은 말풍선 아래 선(y=34.5)에 정확히 맞추고 테두리를 주지 않는다.
# 조금이라도 위로 올리거나 테두리를 두면 그 선을 넘어 얼굴 안쪽으로 삐져나온다.
MASCOT_WHITE = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij4KICA8cGF0aCBkPSJNMTUgMzQuNSBoOSBsLTkgOSB6IiBmaWxsPSIjZmZmZmZmIi8+CiAgPGcgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjMiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+CiAgICA8cGF0aCBkPSJNMjQgNiBWMTAiLz4KICAgIDxyZWN0IHg9IjQuOCIgeT0iMTAiIHdpZHRoPSIzOC40IiBoZWlnaHQ9IjI0LjUiIHJ4PSI4LjUiLz4KICAgIDxwYXRoIGQ9Ik0xOC42IDI1LjIgcTUuNCA0LjggMTAuOCAwIi8+CiAgPC9nPgogIDxjaXJjbGUgY3g9IjI0IiBjeT0iMy45IiByPSIyLjkiIGZpbGw9IiNmZmZmZmYiLz4KICA8Y2lyY2xlIGN4PSIxOC4yIiBjeT0iMTkuMiIgcj0iMyIgZmlsbD0iI2ZmZmZmZiIvPgogIDxjaXJjbGUgY3g9IjI5LjgiIGN5PSIxOS4yIiByPSIzIiBmaWxsPSIjZmZmZmZmIi8+Cjwvc3ZnPg=="
MASCOT_ORANGE = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij4KICA8cGF0aCBkPSJNMTUgMzQuNSBoOSBsLTkgOSB6IiBmaWxsPSIjZWQ1NDJiIi8+CiAgPGcgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZWQ1NDJiIiBzdHJva2Utd2lkdGg9IjMiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+CiAgICA8cGF0aCBkPSJNMjQgNiBWMTAiLz4KICAgIDxyZWN0IHg9IjQuOCIgeT0iMTAiIHdpZHRoPSIzOC40IiBoZWlnaHQ9IjI0LjUiIHJ4PSI4LjUiLz4KICAgIDxwYXRoIGQ9Ik0xOC42IDI1LjIgcTUuNCA0LjggMTAuOCAwIi8+CiAgPC9nPgogIDxjaXJjbGUgY3g9IjI0IiBjeT0iMy45IiByPSIyLjkiIGZpbGw9IiNlZDU0MmIiLz4KICA8Y2lyY2xlIGN4PSIxOC4yIiBjeT0iMTkuMiIgcj0iMyIgZmlsbD0iI2VkNTQyYiIvPgogIDxjaXJjbGUgY3g9IjI5LjgiIGN5PSIxOS4yIiByPSIzIiBmaWxsPSIjZWQ1NDJiIi8+Cjwvc3ZnPg=="

CHAT_ANCHOR_ID = "nbedl-chat-anchor"
OPEN_KEY = "nbedl_chat_open"

_CHAT_CSS = """
<script>
(function() {
  try {
    var doc = window.parent.document, win = window.parent, ID = 'nbedl-chat-style';
    var st = doc.getElementById(ID);
    if (!st) { st = doc.createElement('style'); st.id = ID; doc.head.appendChild(st); }
    st.textContent = [
      /* 떠 있는 창. 왼쪽 위치는 사이드바 폭에 따라 자바스크립트가 정해 준다. */
      '.nbedl-chat-panel{position:fixed !important;bottom:20px;z-index:9990;',
      '  left:var(--nbedl-chat-left,20px);width:auto !important;',
      '  transition:left .2s ease;}',
      '.nbedl-chat-panel[data-open="1"]{',
      '  width:min(470px,calc(100vw - var(--nbedl-chat-left,20px) - 20px)) !important;',
      '  max-height:min(78vh,780px);overflow-y:auto;overflow-x:hidden;',
      '  background:var(--background-color,#ffffff);',
      '  border:1px solid rgba(49,51,63,.18);border-radius:18px;',
      '  box-shadow:0 14px 48px rgba(0,0,0,.22);padding:14px 16px 10px;}',

      /* 닫혀 있을 때 = 알약 모양 단추. 이모지 대신 직접 그린 마스코트를 왼쪽에 얹는다.
         help 툴팁이 단추를 span 으로 한 겹 더 감싸므로 자식(>)이 아니라 자손으로 짚는다. */
      '.nbedl-chat-panel[data-open="0"] [data-testid="stButton"] button{',
      '  height:62px !important;padding:0 26px 0 68px !important;',
      '  border-radius:31px !important;border:none !important;',
      '  background-image:url("%MASCOT%"),linear-gradient(135deg,#ed542b,#f68b21) !important;',
      '  background-repeat:no-repeat,no-repeat !important;',
      '  background-position:18px center,center !important;',
      '  background-size:38px 38px,100% 100% !important;',
      '  color:#fff !important;font-size:1.02rem;font-weight:800;letter-spacing:.01em;',
      '  white-space:nowrap;transition:transform .16s ease, box-shadow .16s ease;',
      '  animation:nbedlChatPulse 2.6s ease-out 4;}',
      '.nbedl-chat-panel[data-open="0"] [data-testid="stButton"] button:hover{',
      '  transform:translateY(-2px) scale(1.03);animation:none;',
      '  box-shadow:0 12px 34px rgba(237,84,43,.5) !important;}',
      '.nbedl-chat-panel[data-open="0"] [data-testid="stButton"] button p,',
      '.nbedl-chat-panel[data-open="0"] [data-testid="stButton"] button div{',
      '  font-size:1.02rem !important;font-weight:800 !important;color:#fff !important;}',
      /* 처음 몇 번만 파문이 퍼진다. 계속 움직이면 곧 거슬린다. */
      '@keyframes nbedlChatPulse{',
      '  0%{box-shadow:0 8px 26px rgba(0,0,0,.26),0 0 0 0 rgba(237,84,43,.55);}',
      '  70%{box-shadow:0 8px 26px rgba(0,0,0,.26),0 0 0 18px rgba(237,84,43,0);}',
      '  100%{box-shadow:0 8px 26px rgba(0,0,0,.26),0 0 0 0 rgba(237,84,43,0);}}',

      /* 창 안은 여백을 죄어 좁은 폭에서도 읽히게.
         창 자체가 곧 stVerticalBlock 이므로 자손 선택자로는 창의 gap 을 못 줄인다.
         창을 직접 짚는 규칙을 따로 둔다. 확장 상자는 흰 카드를 만드는 안쪽 여백(24 px)이
         겹겹이 쌓여 대화도 시작하기 전에 화면 절반을 먹으므로 없앤다. */
      '.nbedl-chat-panel[data-open="1"]{gap:.35rem !important;}',
      '.nbedl-chat-panel[data-open="1"] [data-testid="stVerticalBlock"]{gap:.3rem !important;}',
      '.nbedl-chat-panel[data-open="1"] [data-testid="stExpander"]{padding:0 !important;}',
      '.nbedl-chat-panel[data-open="1"] [data-testid="stExpander"] summary{',
      '  padding-top:.3rem !important;padding-bottom:.3rem !important;}',
      '.nbedl-chat-panel[data-open="1"] [data-testid="stExpander"] summary p{',
      '  font-size:.84rem !important;}',
      '.nbedl-chat-panel .stChatMessage{padding:.35rem .55rem;}',
      '.nbedl-chat-panel p,.nbedl-chat-panel li{font-size:.88rem;}',
      /* 창 안의 보통 단추는 좁은 칸에서도 글자가 잘리지 않게 */
      '.nbedl-chat-panel[data-open="1"] [data-testid="stButton"] button{',
      '  padding-left:.3rem;padding-right:.3rem;white-space:nowrap;}',

      /* 머리글은 한 덩어리 flex, 닫기 단추는 창 오른쪽 위에 절대 위치로 고정한다.
         칸으로 나누면 칸 높이가 제각각이라 세로 정렬이 늘 몇 픽셀씩 어긋난다. */
      '.nbedl-chat-head{display:flex;align-items:center;gap:9px;padding-right:44px;',
'  min-height:46px;margin-bottom:10px;}',
      '.nbedl-chat-head .t{font-weight:800;font-size:1.02rem;line-height:1.2;}',
      '.nbedl-chat-head .s{font-size:.72rem;opacity:.6;line-height:1.25;}',
      '.nbedl-chat-panel[data-open="1"]{position:fixed !important;}',
      /* 창 안쪽 여백 14 + 머리글 높이의 절반 23 - 단추 높이의 절반 17 = 20 */
      '.st-key-nbedl_chat_close{position:absolute !important;top:20px;right:14px;',
      '  width:auto !important;z-index:3;}',
      '.st-key-nbedl_chat_close button{width:34px !important;height:34px !important;',
      '  min-width:34px !important;min-height:34px !important;padding:0 !important;',
      '  border-radius:9px !important;font-size:.9rem !important;line-height:1 !important;}',
      '.st-key-nbedl_chat_header [data-testid="stButton"] button{',
      '  width:36px !important;height:36px !important;min-width:36px !important;',
      '  padding:0 !important;border-radius:9px !important;',
      '  font-size:.95rem !important;line-height:1 !important;}',
      /* 단추가 본문 마지막 줄을 가리지 않도록 아래 여백 */
      '[data-testid="stMain"] .block-container{padding-bottom:120px;}'
    ].join('');

    // 사이드바는 z-index 가 이 창보다 훨씬 높아서, 왼쪽 끝에 두면 그 아래로 숨는다.
    // 그래서 사이드바의 오른쪽 끝을 재어 그만큼 비켜 놓는다. 접으면 화면 밖으로
    // 밀려나므로 0 이 되어 원래 자리로 돌아온다.
    var place = function() {
      var sb = doc.querySelector('[data-testid="stSidebar"]');
      var w = 0;
      if (sb) {
        var r = sb.getBoundingClientRect();
        if (r.width > 0) w = Math.max(0, Math.round(r.right));
      }
      doc.documentElement.style.setProperty('--nbedl-chat-left', (w + 20) + 'px');
    };

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
      place();
    };
    mark();
    if (!win.__nbedlChatObs) {
      win.__nbedlChatObs = new win.MutationObserver(mark);
      win.__nbedlChatObs.observe(doc.body, {childList: true, subtree: true});
      win.addEventListener('resize', place);
      // 사이드바를 접고 펴는 동안에도 따라가도록 몇 번 더 재 본다(애니메이션 때문).
      doc.addEventListener('click', function() {
        [0, 120, 260, 420].forEach(function(t) { win.setTimeout(place, t); });
      }, true);
    }
  } catch (err) { /* 무시 */ }
})();
</script>
"""


def _toggle_chat():
    st.session_state[OPEN_KEY] = not st.session_state.get(OPEN_KEY, False)


def render_floating_chat(config_vars, target_vars):
    """화면 왼쪽 아래에 떠 있는 분석 도우미. 탭 밖에서 한 번만 호출한다.

    여닫기는 on_click 콜백으로 처리한다. 버튼 안에서 상태를 바꾸고 st.rerun() 을 부르면
    위젯이 부른 리런과 합쳐 두 번 돌지만, 콜백은 리런 전에 실행되므로 한 번으로 끝난다.
    """
    is_open = bool(st.session_state.get(OPEN_KEY, False))
    box = st.container()
    with box:
        st.markdown(
            f'<div id="{CHAT_ANCHOR_ID}" data-open="{"1" if is_open else "0"}" '
            f'style="height:0;overflow:hidden;"></div>',
            unsafe_allow_html=True)
        if is_open:
            # 머리글은 한 덩어리 HTML 로 그리고, 닫기 단추는 창 오른쪽 위에 절대 위치로
            # 붙인다. 두 칸(columns)으로 나누면 칸 높이가 제각각이라 세로 정렬이 어긋난다.
            # key 를 주면 Streamlit 이 st-key-<key> 클래스를 붙여 주므로 그것으로 짚는다.
            st.markdown(
                "<div class='nbedl-chat-head'>"
                f"<img src='{MASCOT_ORANGE}' width='30' height='30' alt=''>"
                "<div><div class='t'>분석 도우미</div>"
                "<div class='s'>숫자는 앱이 계산해 표로 건네고, 모델은 해석만 합니다.</div>"
                "</div></div>",
                unsafe_allow_html=True)
            with st.container(key="nbedl_chat_close"):
                st.button("✕", key="nbedl_chat_close_btn", on_click=_toggle_chat)
            render_chat(config_vars, target_vars)
        else:
            st.button("분석 도우미에게 물어보기", key="nbedl_chat_open_btn",
                      on_click=_toggle_chat, help="지금 화면의 데이터를 놓고 대화합니다")
    inject_html(_CHAT_CSS.replace("%ANCHOR%", CHAT_ANCHOR_ID)
                         .replace("%MASCOT%", MASCOT_WHITE))
