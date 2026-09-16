"""개인 API 키를 이 PC에 암호화해서 보관한다.

크롬이 비밀번호를 다루는 방식과 같은 구조다. 키 자체는 어디에도 평문으로 남지 않고,
패스프레이즈에서 유도한 열쇠로 암호화한 덩어리만 파일에 남는다. 패스프레이즈를 모르면
그 파일은 아무 쓸모가 없다.

보관 위치를 프로젝트 폴더가 아니라 사용자 홈 아래로 잡은 것은 의도적이다. 프로젝트
폴더는 git 저장소이자 동기화 대상이라, 암호문이라 해도 그 안에 두면 원격 저장소로
따라 올라갈 여지가 생긴다.

Excel 저장 경로와는 완전히 분리되어 있다. build_excel_bytes 는 저장할 시트를 명시적으로
나열하는 방식이므로 이 모듈의 값이 섞여 들어갈 통로가 없고, 그 사실을 회귀 테스트로
고정해 둔다(tests/test_secret_isolation.py).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except Exception:  # pragma: no cover - 설치 안 된 환경
    AESGCM = None
    CRYPTO_AVAILABLE = False

STORE_DIR = Path(os.path.expanduser("~")) / ".nbedl-exp-assistant"
STORE_PATH = STORE_DIR / "secrets.json"

# scrypt 파라미터. 패스프레이즈는 짧으므로 유도 비용을 충분히 크게 잡아야 사전 공격을
# 느리게 만들 수 있다. n=2**15 는 보통 PC 에서 0.1 초 안팎이라 체감 지연이 없다.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _KEY_LEN = 2 ** 15, 8, 1, 32
# OpenSSL 의 기본 메모리 상한은 32 MB 라서 위 파라미터(128*n*r = 32 MB)가 그대로 걸린다.
# 여유를 두어 명시적으로 올려 준다.
_SCRYPT_MAXMEM = 96 * 1024 * 1024


class SecretError(Exception):
    """패스프레이즈가 틀렸거나 저장 파일이 손상된 경우."""


def _derive(passphrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(passphrase.encode("utf-8"), salt=salt,
                          n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
                          dklen=_KEY_LEN, maxmem=_SCRYPT_MAXMEM)


def store_exists() -> bool:
    return STORE_PATH.is_file()


def store_location() -> str:
    return str(STORE_PATH)


def save_secret(name: str, value: str, passphrase: str) -> None:
    """value 를 패스프레이즈로 암호화해 저장한다. 평문은 파일에 남지 않는다."""
    if not CRYPTO_AVAILABLE:
        raise SecretError("cryptography 패키지가 필요합니다. `pip install cryptography` 후 다시 시도하세요.")
    if not passphrase:
        raise SecretError("패스프레이즈를 입력하세요.")
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = _derive(passphrase, salt)
    blob = AESGCM(key).encrypt(nonce, value.encode("utf-8"), name.encode("utf-8"))
    data = _read_raw()
    data[name] = {
        "salt": base64.b64encode(salt).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "blob": base64.b64encode(blob).decode(),
    }
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(STORE_PATH)
    try:
        os.chmod(STORE_PATH, 0o600)   # 윈도우에서는 효과가 제한적이지만 해가 없다
    except OSError:
        pass


def load_secret(name: str, passphrase: str) -> str:
    """패스프레이즈로 복호화한 값을 돌려준다. 틀리면 SecretError."""
    if not CRYPTO_AVAILABLE:
        raise SecretError("cryptography 패키지가 필요합니다. `pip install cryptography` 후 다시 시도하세요.")
    rec = _read_raw().get(name)
    if not rec:
        raise SecretError("저장된 키가 없습니다.")
    try:
        key = _derive(passphrase, base64.b64decode(rec["salt"]))
        out = AESGCM(key).decrypt(base64.b64decode(rec["nonce"]),
                                  base64.b64decode(rec["blob"]), name.encode("utf-8"))
    except Exception:
        # 어떤 이유로 실패했는지 구분해 알려 주면 공격자에게 힌트가 된다. 한 가지로 묶는다.
        raise SecretError("패스프레이즈가 맞지 않거나 저장된 내용이 손상되었습니다.")
    return out.decode("utf-8")


def forget_secret(name: str) -> None:
    data = _read_raw()
    if data.pop(name, None) is not None:
        STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_raw() -> dict:
    if not STORE_PATH.is_file():
        return {}
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mask(secret: str) -> str:
    """로그나 오류 메시지에 키가 섞여 나가지 않게 가린다."""
    if not secret:
        return ""
    return f"{secret[:4]}…{secret[-2:]}" if len(secret) > 10 else "…"


def scrub(text: str, secret: str) -> str:
    """예외 문자열 등에 키가 그대로 담겨 있으면 지운다."""
    if secret and text:
        return text.replace(secret, "[REDACTED]")
    return text
