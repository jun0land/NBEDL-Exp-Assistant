"""실험 데이터를 곁에 두고 대화하는 분석 도우미 (Gemini).

설계 원칙이 하나 있다. **숫자는 파이썬이 계산하고, 모델은 해석만 한다.** 언어 모델에게
원자료를 던져 놓고 평균을 내라고 시키면 틀린 수를 자신 있게 말한다. 그래서 조건별
중앙값, 반복 수, 최소 허용값 충족 여부, 직전 최적화 결과까지 전부 여기서 계산해 표로
만들어 건네고, 모델에게는 "이 표에서 무엇이 읽히는가"만 묻는다.

키는 이 PC 에 암호화되어 보관되고(secret_store), 복호화된 값은 세션 메모리에만 머문다.
Excel 저장 경로와는 닿지 않으며, 네트워크 요청은 Gemini 한 곳으로만 나간다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import streamlit as st

import secret_store
from analysis import target_direction, direction_label

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

    out = [f"학습 적용 {len(valid)}행 / 전체 {len(df)}행. 공정 변수: {', '.join(cfg) or '없음'}.", ""]

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
        out.append("## 조건별 대표값 (같은 조건 반복 시료의 중앙값)")
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
    """키 잠금 해제 / 저장 / 삭제. 평문 키는 화면에도 세션 위젯에도 남기지 않는다."""
    have = bool(st.session_state.get(SESSION_KEY))
    with st.expander("🔑 Gemini API 키" + (" — 잠금 해제됨" if have else ""), expanded=not have):
        if not secret_store.CRYPTO_AVAILABLE:
            st.error("`cryptography` 패키지가 필요합니다. 터미널에서 `pip install cryptography` 후 앱을 다시 시작하세요.")
            return
        st.caption(
            "키는 **이 PC 에만** 패스프레이즈로 암호화되어 보관됩니다. "
            f"보관 위치: `{secret_store.store_location()}`  \n"
            "실험 데이터 Excel 파일에는 **저장되지 않습니다.** 복호화된 키는 앱이 실행 중인 동안 "
            "메모리에만 있고, 요청은 `generativelanguage.googleapis.com` 한 곳으로만 나갑니다.  \n"
            "키 발급은 Google AI Studio(aistudio.google.com/apikey)에서 합니다."
        )
        if have:
            st.success(f"잠금 해제됨 · 키 {secret_store.mask(st.session_state[SESSION_KEY])}")
            c1, c2 = st.columns(2)
            if c1.button("🔒 잠그기 (세션에서 내리기)", use_container_width=True):
                st.session_state.pop(SESSION_KEY, None)
                st.rerun()
            if c2.button("🗑️ 저장된 키 삭제", use_container_width=True):
                secret_store.forget_secret(SECRET_NAME)
                st.session_state.pop(SESSION_KEY, None)
                st.rerun()
            return

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
            if st.form_submit_button("💾 암호화해서 저장", use_container_width=True):
                if not k.strip():
                    st.error("키를 입력하세요.")
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
        c1.selectbox("모델", models, key=MODEL_KEY,
                     help="flash 계열이 빠르고 저렴합니다. 긴 추론이 필요하면 pro 계열을 쓰세요.")
    else:
        c1.text_input("모델 이름", key=MODEL_KEY, placeholder="예: gemini-2.5-flash")
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
                answer = ask(api_key, st.session_state.get(MODEL_KEY) or "gemini-2.5-flash",
                             history, context_md)
            except Exception as e:
                answer = "요청에 실패했습니다: " + secret_store.scrub(str(e)[:500], api_key)
        st.markdown(answer)
    history.append({"role": "assistant", "content": answer})
