"""API 키가 실험 데이터 Excel 로 새지 않는다는 것을 고정해 두는 회귀 테스트.

Excel 저장은 시트를 명시적으로 나열하는 방식이라 지금은 새어 나갈 통로가 없다. 하지만
나중에 "세션 상태를 통째로 저장" 같은 편의 기능이 들어오면 조용히 깨질 수 있는 종류의
불변식이라, 테스트로 붙잡아 둔다.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAKE_KEY = "AIzaSy_THIS_MUST_NEVER_APPEAR_IN_THE_WORKBOOK"


def test_api_key_absent_from_workbook(tmp_path):
    import app

    df = pd.DataFrame({"학습_적용": [True], "샘플명": ["A-1"], "t": [15], "R": [0.048]})
    config_vars = [{"Name": "t", "Unit": "s", "Min": 0, "Max": 30}]
    target_vars = [{"Name": "R", "Unit": "A/W", "Direction": "Maximize"}]
    meta = [{"AI_Target_Selection": "[]", "AI_Target_Weights": "{}", "AI_Target_Floors": "{}"}]

    raw = app.build_excel_bytes(df, config_vars, target_vars, meta)
    assert FAKE_KEY.encode() not in raw

    out = tmp_path / "book.xlsx"
    out.write_bytes(raw)
    text = "".join(
        sheet.to_csv() for sheet in pd.read_excel(out, sheet_name=None).values()
    )
    assert FAKE_KEY not in text


def test_secret_store_writes_no_plaintext(tmp_path, monkeypatch):
    import secret_store

    monkeypatch.setattr(secret_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(secret_store, "STORE_PATH", tmp_path / "secrets.json")

    secret_store.save_secret("gemini_api_key", FAKE_KEY, "passphrase")
    on_disk = (tmp_path / "secrets.json").read_text(encoding="utf-8")
    assert FAKE_KEY not in on_disk
    assert secret_store.load_secret("gemini_api_key", "passphrase") == FAKE_KEY

    try:
        secret_store.load_secret("gemini_api_key", "wrong")
    except secret_store.SecretError:
        pass
    else:
        raise AssertionError("잘못된 패스프레이즈로 복호화가 성공했다")


def test_only_gemini_host_is_contacted():
    import ai_chat

    assert ai_chat.API_HOST == "generativelanguage.googleapis.com"
    assert ai_chat.API_BASE.startswith("https://" + ai_chat.API_HOST)
    source = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_chat.py"),
        encoding="utf-8",
    ).read()
    # 코드 안에 다른 http(s) 목적지가 있으면 안 된다 (설명 문구의 도메인 표기는 제외).
    import re
    urls = set(re.findall(r"https?://[\w.\-]+", source))
    assert urls <= {"https://" + ai_chat.API_HOST}, urls
