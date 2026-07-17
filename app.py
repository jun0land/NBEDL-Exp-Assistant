import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import base64
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
import streamlit.components.v1 as components
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards

# ==========================================
# 1. 페이지 기본 설정 및 이탈 방지
# ==========================================
st.set_page_config(page_title="NBEDL Exp Assistant", layout="wide")

components.html(
    """
    <script>
    window.onbeforeunload = function() { return "데이터가 저장되지 않았을 수 있습니다."; };
    setInterval(function() {
        var inputs = document.querySelectorAll('input');
        inputs.forEach(function(input) {
            input.setAttribute('autocomplete', 'new-password');
        });
    }, 1000);
    </script>
    """, height=0,
)

# ==========================================
# 2. 이미지 Base64 인코딩
# ==========================================
@st.cache_data(show_spinner=False)
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f: return base64.b64encode(f.read()).decode()
    return ""

# 배경 이미지 최적화:
# 원본 PNG는 4096px/8.5MB라 base64로 약 11MB에 달해 로딩을 크게 지연시켰다.
# 2048px WebP(82KB, PSNR 40.5dB로 육안 차이 없음)로 줄이고, static 서빙이 가능하면
# URL로 참조해 브라우저가 캐시하도록 한다(새로고침 시 재전송 없음).
# static 서빙이 불가한 환경에서는 base64 인라인으로 폴백한다. (원본 PNG는 소스로 보존)
BG_STATIC_PATH = 'static/liquid_bg.webp'

def _resolve_bg_url():
    if os.path.exists(BG_STATIC_PATH):
        try:
            if st.get_option("server.enableStaticServing"):
                return "app/static/liquid_bg.webp"
        except Exception:
            pass
        return f"data:image/webp;base64,{get_base64_of_bin_file(BG_STATIC_PATH)}"
    b64 = get_base64_of_bin_file('liquid_bg.png')
    return f"data:image/png;base64,{b64}" if b64 else ""

bg_url = _resolve_bg_url()
logo_base64 = get_base64_of_bin_file('logo.png')

logo_html = f'<img src="data:image/png;base64,{logo_base64}" height="42" style="vertical-align: middle; margin-right: 12px;">' if logo_base64 else ""

# =========================================================================
# 3. 완벽 분석 적용 CSS
# =========================================================================
custom_css = f"""
<style>
/* 폰트 적용 (아이콘 폰트 절대 보호) */
html, body, p, h1, h2, h3, h4, h5, h6, label, span, div {{ font-family: 'Pretendard', sans-serif; }}
.material-symbols-rounded, .material-icons, [class*="icon"], [data-baseweb="icon"] {{ font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important; }}

/* 최상위 배경 이미지 */
.stApp {{
    background: linear-gradient(135deg, rgba(255,255,255,0.45), rgba(255,255,255,0.25)), url("{bg_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #1a1a1a;
}}

/* 네온(Shadow) 효과 완벽 삭제 */
* {{ text-shadow: none !important; }}

/* 불투명 방해막 모조리 투명하게 제거 */
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {{
    background: transparent !important;
}}

/* ========================================================================= */
/* 진정한 Liquid Glass 블럭 (사용자님 피드백 블러값 그대로 유지) */
/* ========================================================================= */
[data-testid="stForm"], 
[data-testid="stExpander"], 
[data-testid="stVerticalBlockBorderWrapper"], 
.title-glass-container {{
    background: rgba(255, 255, 255, 0.15) !important; 
    backdrop-filter: blur(48px) saturate(150%) !important; 
    -webkit-backdrop-filter: blur(48px) saturate(150%) !important;
    border: none !important; 
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.05) !important; 
    border-radius: 20px !important; 
    padding: 24px !important;
    margin-bottom: 16px !important;
}}

/* 컨테이너 내부의 보이지 않는 흰색 블럭 강제 투명화 */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background: transparent !important;
}}

/* 사이드바는 유리 겹침 방지 */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent !important; backdrop-filter: none !important; box-shadow: none !important; padding: 0 !important;
}}

/* ✅ [해결] 타이틀 박스 가로 정렬 (줄바꿈 원천 차단) */
.title-glass-container {{
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    padding: 16px 24px !important;
    margin-bottom: 24px !important;
    margin-top: -10px !important;
    border-left: 6px solid #ed542b !important; 
}}
.title-glass-container img {{
    flex-shrink: 0 !important;
}}
.title-glass-container h2 {{
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.1 !important;
    display: inline-block !important;
    white-space: nowrap !important;
}}

/* ========================================================================= */
/* 탭(Tabs) 내부의 모든 하위 요소 배경까지 투명하게 강제 날리기 */
/* ========================================================================= */
[data-testid="stTabs"] [data-baseweb="tab-list"],
[data-testid="stTabs"] [data-baseweb="tab-list"] * {{
    background-color: transparent !important;
    background: transparent !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    border: none !important;
    color: #7a716c !important;
    font-weight: 800 !important;
    padding: 10px 8px !important; 
    box-shadow: none !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: #ed542b !important;
    border-bottom: 3px solid #ed542b !important; 
    border-radius: 0 !important;
}}

/* --------------------------------------------------- */
/* 파일 업로더 회색 바탕 없애고 유리 질감 뚫어주기 */
/* --------------------------------------------------- */
[data-testid="stFileUploader"] {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(24px) !important;
    border: 1px dashed rgba(237, 84, 43, 0.4) !important;
    border-radius: 16px !important;
    padding: 16px !important;
}}
[data-testid="stFileUploader"] section {{ background: transparent !important; }}
[data-testid="stFileUploader"] button {{ background: rgba(255,255,255,0.4) !important; border: 1px solid rgba(255,255,255,0.6) !important; box-shadow: none !important; }}
[data-testid="stFileUploader"] button:hover {{ background: #ed542b !important; color: white !important; border-color: #ed542b !important; }}

/* 폼 내부에 변수별로 묶인 소그룹 */
[data-testid="stForm"] [data-testid="column"] {{
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(24px) saturate(150%) !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}}

/* 입력칸(Input) */
div[data-baseweb="input"], div[data-baseweb="select"] > div {{
    background: rgba(255, 255, 255, 0.5) !important; 
    backdrop-filter: blur(12px) !important;
    border-radius: 10px !important; 
    border: 1px solid rgba(255, 255, 255, 0.6) !important; 
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important; 
}}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {{
    background: rgba(255,255,255,0.8) !important;
    border-color: #ed542b !important;
}}

/* 버튼 공통 투명화 및 Hover */
.stButton > button {{
    border-radius: 12px !important; 
    border: 1px solid rgba(255, 255, 255, 0.5) !important; 
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(20px) !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
    transition: all 0.2s ease !important; 
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important; 
    background: #ed542b !important; 
    border-color: #ed542b !important; 
    color: white !important; 
    box-shadow: 0 8px 20px rgba(237,84,43,0.25) !important; 
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #ed542b, #f68b21) !important;
    border: none !important;
    color: white !important;
}}
.stButton > button[kind="primary"]:hover {{
    filter: brightness(1.1) !important; 
    box-shadow: 0 8px 20px rgba(237,84,43,0.4) !important;
}}

/* 차트 배경 완전 투명화 및 캔버스 곡률 추가 */
[data-testid="stVegaLiteChart"] {{
    background: transparent !important; 
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}}
[data-testid="stVegaLiteChart"] canvas {{
    border-radius: 12px !important;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 3.5 사용 설명서(메뉴얼) 팝업
# ==========================================
MANUAL_HTML = """
<p>이 앱은 실험 조건과 결과를 입력하면, <b>AI(베이지안 최적화)</b>가 다음에 시도해볼
<b>최적의 공정 조건</b>을 추천해 주는 실험 보조 도구입니다.</p>

<h4>🚀 단계별 따라하기</h4>

<div class="nbedl-step">STEP 1. 프로젝트 설정 <span class="nbedl-loc">· 실험 설정 화면</span></div>
<ul>
  <li>이미 실험 데이터 파일(.xlsx)이 있다면 → 맨 위 <b>"기존 실험 데이터 불러오기"</b>에 업로드하면 설정이 자동으로 채워집니다.</li>
  <li>처음이라면 직접 입력: <b>실험 프로젝트 이름</b>(예: <code>NBEDL_Experiment_01</code>), <b>목표 지표 이름</b>(예: <code>J_sc</code>)과 <b>최적화 방향</b>(최대화/최소화), <b>환경 변수</b>(온도·습도 등, 선택), <b>공정 변수</b>(이름·단위·타입·범위). <b>"➕ 공정 변수 블럭 추가"</b>로 여러 개 추가.</li>
  <li>다 됐으면 <b>"🚀 실험 시작 및 대시보드 생성"</b> 클릭</li>
</ul>

<div class="nbedl-step">STEP 2. 실험 결과 입력 <span class="nbedl-loc">· 📝 신규 실험 입력 탭</span></div>
<ul>
  <li>각 조건값과 결과값을 넣고 <b>"➕ 데이터 추가"</b> 버튼으로 저장 <span class="nbedl-tip">(Enter는 값 확정만, 추가는 버튼으로)</span></li>
  <li>잘못 넣었다면 <b>"↩️ 마지막 입력 취소"</b>로 직전 데이터 삭제</li>
</ul>

<div class="nbedl-step">STEP 3. 데이터 검토 <span class="nbedl-loc">· 🗂️ 데이터베이스 관리 탭</span></div>
<ul>
  <li>지금까지 입력한 모든 이력을 표에서 확인·수정</li>
  <li><b>"학습 적용"</b> 체크를 해제하면 그 데이터는 AI 학습에서 제외됩니다 (이상치 처리에 유용)</li>
</ul>

<div class="nbedl-step">STEP 4. AI 추천 받기 <span class="nbedl-loc">· 🤖 AI 최적화 대시보드 탭</span></div>
<ul>
  <li><b>"🚀 AI 계산 실행"</b> 클릭 → 다음에 시도할 <b>추천 조건 3가지</b> 제시</li>
  <li><b>경향 곡선</b>으로 실험이 목표에 수렴하는지 확인</li>
  <li>⚠️ 최소 <b>2개 이상의 유효 데이터</b>가 있어야 계산됩니다</li>
</ul>

<div class="nbedl-step">STEP 5. 저장 &amp; 관리 <span class="nbedl-loc">· 왼쪽 사이드바</span></div>
<ul>
  <li><b>"📥 최신 데이터 Excel 다운로드"</b>로 저장 → 다음에 이 파일을 STEP 1에서 올리면 이어서 작업 가능</li>
  <li>"🛠️ 환경 설정으로 돌아가기" / "⚠️ 모든 데이터 초기화"</li>
</ul>

<h4>⚠️ 주의사항 &amp; 팁</h4>
<div class="nbedl-note">
  <ul>
    <li><b>자동 저장이 안 됩니다.</b> 데이터는 브라우저 세션에만 있어, 새로고침하거나 창을 닫으면 사라질 수 있습니다. <b>작업 후 반드시 Excel로 다운로드</b>하세요.</li>
    <li><b>"모든 데이터 초기화"는 되돌릴 수 없습니다.</b> 초기화 전에 꼭 저장하세요.</li>
    <li><b>데이터가 많을수록 AI 추천이 정확</b>해집니다. 초반엔 다양한 조건을 폭넓게 시도해 보세요.</li>
    <li>같은 조건을 여러 번 반복 측정하면, AI가 <b>이상치를 자동으로 걸러</b> 평균을 사용합니다.</li>
  </ul>
</div>
"""

DRAWER_CSS = """
#nbedl-manual-root, #nbedl-manual-root * { box-sizing: border-box; }
#nbedl-manual-root { font-family: 'Pretendard', sans-serif; }

.nbedl-bookmark {
  position: fixed; top: 168px; right: 0; z-index: 2147483401;
  display: flex; align-items: center; justify-content: center;
  padding: 18px 9px; writing-mode: vertical-rl; text-orientation: mixed;
  background: linear-gradient(160deg, #ed542b, #f68b21); color: #fff;
  font-weight: 800; letter-spacing: 2px; font-size: 15px;
  border-radius: 14px 0 0 14px; box-shadow: -4px 6px 18px rgba(237,84,43,0.35);
  cursor: pointer; user-select: none;
  transition: padding-right .18s ease, box-shadow .18s ease, transform .18s ease;
}
.nbedl-bookmark:hover { padding-right: 15px; box-shadow: -7px 8px 24px rgba(237,84,43,0.5); }

.nbedl-backdrop {
  position: fixed; inset: 0; z-index: 2147483500;
  background: rgba(20,20,20,0.30);
  -webkit-backdrop-filter: blur(3px); backdrop-filter: blur(3px);
  opacity: 0; pointer-events: none; transition: opacity .35s ease;
}
.nbedl-panel {
  /* zoom 스케일 하에서 vh/vw는 뷰포트 고정이라 어긋남 → top/bottom 앵커 + % 사용 */
  position: fixed; top: 0; bottom: 0; right: 0; width: min(460px, 92%);
  z-index: 2147483501; overflow-y: auto; padding: 30px 30px 44px;
  background: rgba(255,255,255,0.94);
  -webkit-backdrop-filter: blur(26px) saturate(160%); backdrop-filter: blur(26px) saturate(160%);
  border-left: 6px solid #ed542b; box-shadow: -18px 0 50px rgba(0,0,0,0.20);
  transform: translateX(106%); transition: transform .38s cubic-bezier(.22,.61,.36,1);
  color: #1a1a1a;
}
#nbedl-manual-root.open .nbedl-backdrop { opacity: 1; pointer-events: auto; }
#nbedl-manual-root.open .nbedl-panel { transform: translateX(0); }

.nbedl-close {
  position: absolute; top: 16px; right: 18px; width: 34px; height: 34px;
  border: none; border-radius: 10px; background: rgba(0,0,0,0.06); color: #333;
  font-size: 15px; cursor: pointer; transition: background .2s ease, color .2s ease;
}
.nbedl-close:hover { background: #ed542b; color: #fff; }

.nbedl-title { font-size: 21px; font-weight: 900; color: #ed542b; margin: 2px 40px 18px 0;
  border-bottom: 2px solid rgba(237,84,43,0.25); padding-bottom: 12px; }
.nbedl-panel h4 { font-size: 16px; font-weight: 800; color: #ed542b; margin: 24px 0 8px; }
.nbedl-panel p { line-height: 1.65; margin: 8px 0; }
.nbedl-panel ul { margin: 6px 0 14px; padding-left: 20px; }
.nbedl-panel li { line-height: 1.6; margin: 5px 0; }
.nbedl-step { font-weight: 800; margin: 18px 0 4px; color: #1a1a1a; }
.nbedl-loc { font-weight: 600; color: #9a8f89; font-size: 0.9em; }
.nbedl-tip { color: #ed542b; font-weight: 700; font-size: 0.9em; }
.nbedl-panel code { background: rgba(237,84,43,0.10); color: #c53a17; padding: 1px 6px; border-radius: 6px; font-size: 0.9em; }
.nbedl-note { background: rgba(255,244,235,0.85); border-left: 4px solid #f68b21; border-radius: 10px; padding: 14px 16px; margin-top: 16px; }
.nbedl-note ul { margin: 0; }
"""

def render_manual_drawer():
    """우측 모서리 북마크 탭 + 슬라이드-인 메뉴얼 패널을 부모 문서에 주입한다.
    st.markdown은 <script>를 제거하므로 components.html(iframe)에서 JS로 주입한다."""
    payload = """
<script>
(function() {
  try {
    var doc = window.parent.document;
    var ROOT_ID = 'nbedl-manual-root';
    var STYLE_ID = 'nbedl-manual-style';
    var oldRoot = doc.getElementById(ROOT_ID);
    var wasOpen = !!(oldRoot && oldRoot.classList.contains('open'));  // 리런 시 열림 상태 유지
    if (oldRoot) oldRoot.remove();
    var oldStyle = doc.getElementById(STYLE_ID);  if (oldStyle) oldStyle.remove();

    var style = doc.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `__CSS__`;
    doc.head.appendChild(style);

    var root = doc.createElement('div');
    root.id = ROOT_ID;
    root.innerHTML = `
      <div class="nbedl-bookmark" title="사용 설명서 열기"><span>📖 사용 설명서</span></div>
      <div class="nbedl-backdrop"></div>
      <aside class="nbedl-panel" role="dialog" aria-label="사용 설명서">
        <button class="nbedl-close" title="닫기 (Esc)">✕</button>
        <div class="nbedl-title">📖 NBEDL Exp Assistant 사용 설명서</div>
        __CONTENT__
      </aside>`;
    doc.body.appendChild(root);
    if (wasOpen) root.classList.add('open');

    var openFn  = function() { root.classList.add('open'); };
    var closeFn = function() { root.classList.remove('open'); };
    root.querySelector('.nbedl-bookmark').addEventListener('click', openFn);
    root.querySelector('.nbedl-backdrop').addEventListener('click', closeFn);
    root.querySelector('.nbedl-close').addEventListener('click', closeFn);

    if (!window.parent.__nbedlEsc) {
      window.parent.__nbedlEsc = true;
      doc.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
          var r = doc.getElementById('nbedl-manual-root');
          if (r) r.classList.remove('open');
        }
      });
    }
  } catch (err) { /* cross-origin 등 접근 불가 시 조용히 무시 */ }
})();
</script>
"""
    payload = payload.replace("__CSS__", DRAWER_CSS).replace("__CONTENT__", MANUAL_HTML)
    components.html(payload, height=0)


def disable_form_enter_submit():
    """입력 폼(st.form) 안에서 Enter가 폼을 제출(데이터 추가)하지 않도록 막는다.
    값은 입력칸에 그대로 남고, 추가는 '데이터 추가' 버튼으로만 수행된다."""
    components.html("""
<script>
(function() {
  try {
    var doc = window.parent.document;
    if (window.parent.__nbedlEnterGuard) return;
    window.parent.__nbedlEnterGuard = true;
    doc.addEventListener('keydown', function(e) {
      if (e.key !== 'Enter') return;
      var t = e.target;
      if (!t || !t.closest) return;
      var inForm = t.closest('[data-testid="stForm"]');
      var tag = (t.tagName || '').toLowerCase();
      if (inForm && (tag === 'input' || tag === 'select')) {
        e.preventDefault();
        e.stopPropagation();
        if (e.stopImmediatePropagation) e.stopImmediatePropagation();
      }
    }, true);  // capture 단계에서 가로채 Streamlit의 제출 핸들러보다 먼저 처리
  } catch (err) { /* 무시 */ }
})();
</script>
""", height=0)


def apply_ui_zoom():
    """화면 크기에 따라 UI를 '확대/축소'한다. (공간만 늘어나는 리플로우 대신 스케일)
    기준 1440px에서 1.0배, 0.85~1.35배로 클램프.
    zoom은 transform:scale과 달리 재레이아웃이라 글자가 뭉개지지 않는다.

    주의: body 전체에 zoom을 걸면 Streamlit 내부의 100vh 기반 레이아웃(stMain 등)과
    충돌한다(zoom은 vh를 보정하지 않아 부모보다 커져 화면이 위로 밀림). 그래서 vh 의존이
    없는 콘텐츠 컨테이너와 드로어에만 적용한다.
    또한 인라인 스타일은 리런 시 DOM이 교체되며 사라지므로 <style> 규칙으로 주입한다."""
    components.html("""
<script>
(function() {
  try {
    var win = window.parent, doc = win.document;
    var DESIGN = 1440, MIN = 0.85, MAX = 1.35;
    var ID = 'nbedl-zoom-style';
    var st = doc.getElementById(ID);
    if (!st) { st = doc.createElement('style'); st.id = ID; doc.head.appendChild(st); }
    win.__nbedlApplyZoom = function() {
      var z = win.innerWidth / DESIGN;
      z = Math.max(MIN, Math.min(MAX, z));
      doc.body.style.zoom = '';   // 과거 body-zoom 방식 잔재 제거
      st.textContent =
        '[data-testid="stMain"] .block-container { zoom: ' + z + '; }' +
        '#nbedl-manual-root { zoom: ' + z + '; }';
    };
    win.__nbedlApplyZoom();
    if (!win.__nbedlZoomBound) {
      win.__nbedlZoomBound = true;
      win.addEventListener('resize', function() { win.__nbedlApplyZoom(); });
    }
  } catch (err) { /* 무시 */ }
})();
</script>
""", height=0)


apply_ui_zoom()
render_manual_drawer()
disable_form_enter_submit()

# ==========================================
# 4. 시스템 상태 초기화
# ==========================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Setup"
if "exp_name" not in st.session_state:
    st.session_state.exp_name = ""
if "config_vars" not in st.session_state:
    st.session_state.config_vars = []
if "target_info" not in st.session_state:
    st.session_state.target_info = {"name": "", "direction": "Maximize"}
if "passive_vars" not in st.session_state:
    st.session_state.passive_vars = []
if "df_data" not in st.session_state:
    st.session_state.df_data = pd.DataFrame()

# ==========================================
# 5. 전처리 및 엑셀 로드 함수
# ==========================================
def process_robust_data(df, feature_cols, target_col):
    grouped = df.groupby(feature_cols)
    robust_X, robust_y = [], []
    for name, group in grouped:
        y_vals = group[target_col].tolist()
        if len(y_vals) >= 3:
            q1, q3 = np.percentile(y_vals, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            valid_y = [y for y in y_vals if lower <= y <= upper]
            if not valid_y: valid_y = y_vals
        else:
            valid_y = y_vals
        x_val = list(name) if isinstance(name, tuple) else [name]
        robust_X.append(x_val)
        robust_y.append(np.mean(valid_y))
    return robust_X, robust_y

@st.cache_data(show_spinner=False, max_entries=3)
def build_excel_bytes(df, config_vars, meta_data):
    """Excel 바이트 생성. 다운로드 버튼 때문에 매 리런마다 재생성되던 것을 캐시한다.
    (데이터가 바뀌면 자동으로 다시 생성된다)"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        pd.DataFrame(config_vars).to_excel(writer, sheet_name='Config_Vars', index=False)
        pd.DataFrame(meta_data).to_excel(writer, sheet_name='Config_Meta', index=False)
    return output.getvalue()

def load_excel_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
    df_meta = pd.read_excel(xls, 'Config_Meta')
    st.session_state.target_info = {
        "name": df_meta.iloc[0]['Target_Name'],
        "direction": df_meta.iloc[0]['Direction']
    }
    
    if 'Exp_Name' in df_meta.columns and pd.notna(df_meta.iloc[0]['Exp_Name']):
        st.session_state.exp_name = str(df_meta.iloc[0]['Exp_Name'])
        
    p_vars_str = str(df_meta.iloc[0]['Passive_Vars'])
    if p_vars_str and p_vars_str != "nan":
        st.session_state.passive_vars = [v.strip() for v in p_vars_str.split(",")]
    else:
        st.session_state.passive_vars = []
        
    df_vars = pd.read_excel(xls, 'Config_Vars')
    config_list = df_vars.to_dict('records')
    for var in config_list:
        if "Old_Name" not in var or pd.isna(var["Old_Name"]):
            var["Old_Name"] = var.get("Name", "")
    st.session_state.config_vars = config_list
    
    st.session_state.df_data = pd.read_excel(xls, 'Data')
    st.session_state.app_mode = "Dashboard"

# ==========================================
# [화면 A] 실험 세팅 모드 (Setup)
# ==========================================
if st.session_state.app_mode == "Setup":
    st.markdown(f'<div class="title-glass-container">{logo_html}<h2>NBEDL AI 기반 공정 최적화 시스템</h2></div>', unsafe_allow_html=True)
    
    col_load, col_basic = st.columns([1, 1.8], gap="medium")

    with col_load:
        with st.container(border=True):
            st.markdown("<h5 style='font-weight: 800;'>기존 실험 데이터 불러오기</h5>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("엑셀(xlsx) 파일을 업로드하면 설정이 자동으로 채워집니다.", type=["xlsx"])
            if uploaded_file:
                load_excel_data(uploaded_file)
                st.rerun()

    with col_basic:
        with st.container(border=True):
            colored_header(label="기본 프로젝트 설정", description="실험 이름과 최적화 목표 지표를 설정하세요.", color_name="orange-70")
            st.session_state.exp_name = st.text_input("📝 실험 프로젝트 이름", value=st.session_state.exp_name, placeholder="예: NBEDL_Experiment_01")

            col_t1, col_t2, col_t3 = st.columns(3)
            target_name = col_t1.text_input("목표 지표 이름", value=st.session_state.target_info["name"], placeholder="예: J_sc")
            target_dir = col_t2.selectbox("최적화 방향", ["Maximize", "Minimize"])

            passive_val = ",".join(st.session_state.passive_vars) if st.session_state.passive_vars else ""
            passive_input = col_t3.text_input("환경 변수 (쉼표 구분)", value=passive_val, placeholder="예: 온도 (°C), 습도 (%)")
    
    with st.container(border=True):
        colored_header(label="🔬 최적화 대상 공정 변수 입력", description="AI가 탐색할 공정 조건의 이름과 변수 범위를 지정하세요.", color_name="orange-70")
        
        for i, var in enumerate(st.session_state.config_vars):
            with st.container(border=True): 
                c1, c_u, c2, c3, c4 = st.columns([2, 1, 2, 2, 2])
                var["Name"] = c1.text_input(f"변수 {i+1} 이름", value=var.get("Name", ""), key=f"name_{i}", placeholder="예: 스핀코팅 속도1")
                var["Unit"] = c_u.text_input("단위", value=var.get("Unit", ""), key=f"unit_{i}", placeholder="예: rpm")
                
                type_options = ["Real (실수)", "Integer (정수)", "Categorical (범주)"]
                safe_type = var.get("Type", "Real (실수)")
                if safe_type not in type_options: safe_type = "Real (실수)"
                    
                var["Type"] = c2.selectbox("타입", type_options, key=f"type_{i}", index=type_options.index(safe_type))
                
                if "Real" in var["Type"]:
                    var["Min"] = c3.number_input("최소값", value=float(var.get("Min", 0.0)), key=f"rmin_{i}")
                    var["Max"] = c4.number_input("최대값", value=float(var.get("Max", 10.0)), key=f"rmax_{i}")
                    var["Options"] = ""
                elif "Integer" in var["Type"]:
                    var["Min"] = c3.number_input("최소값", value=int(var.get("Min", 0)), step=1, key=f"imin_{i}")
                    var["Max"] = c4.number_input("최대값", value=int(var.get("Max", 100)), step=1, key=f"imax_{i}")
                    var["Options"] = ""
                elif "Categorical" in var["Type"]:
                    var["Min"], var["Max"] = 0, 0
                    var["Options"] = c3.text_input("옵션 (쉼표 구분)", value=var.get("Options", ""), key=f"cat_{i}", placeholder="예: CB, Toluene")
        
        if st.button("➕ 공정 변수 블럭 추가", use_container_width=True):
            st.session_state.config_vars.append({
                "Old_Name": "", "Name": "", "Unit": "", "Type": "Real (실수)", 
                "Min": 0.0, "Max": 10.0, "Options": ""
            })
            st.rerun()

    st.write("")
    if st.button("🚀 실험 시작 및 대시보드 생성", type="primary", use_container_width=True):
        if not target_name.strip():
            st.error("목표 지표 이름을 입력해야 합니다.")
        elif not st.session_state.config_vars:
            st.error("최소 1개 이상의 공정 변수를 추가해야 합니다.")
        else:
            st.session_state.target_info = {"name": target_name, "direction": target_dir}
            p_vars = [v.strip() for v in passive_input.split(",") if v.strip()]
            st.session_state.passive_vars = p_vars
            
            if not st.session_state.df_data.empty:
                rename_dict = {}
                for var in st.session_state.config_vars:
                    old = var.get("Old_Name", "")
                    new = var["Name"]
                    if old and old != new and old in st.session_state.df_data.columns:
                        rename_dict[old] = new
                if rename_dict:
                    st.session_state.df_data.rename(columns=rename_dict, inplace=True)
            
            for var in st.session_state.config_vars:
                var["Old_Name"] = var["Name"]
            
            new_cols = ["학습_적용"] + p_vars + [v["Name"] for v in st.session_state.config_vars] + [target_name]
            
            if not st.session_state.df_data.empty:
                for col in new_cols:
                    if col not in st.session_state.df_data.columns:
                        st.session_state.df_data[col] = np.nan
            else:
                st.session_state.df_data = pd.DataFrame(columns=new_cols)
                
            st.session_state.app_mode = "Dashboard"
            st.rerun()

# ==========================================
# [화면 B] 실험 진행 모드 (대시보드 탭 레이아웃)
# ==========================================
elif st.session_state.app_mode == "Dashboard":
    t_name = st.session_state.target_info["name"]
    t_dir = st.session_state.target_info["direction"]
    f_names = [v["Name"] for v in st.session_state.config_vars]
    
    display_exp_name = st.session_state.exp_name if st.session_state.exp_name.strip() else "NBEDL_Experiment"
    
    st.markdown(f'<div class="title-glass-container">{logo_html}<h2>NBEDL Exp Assistant : {display_exp_name}</h2></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📂 데이터 관리 패널")
        meta_data = {
            "Exp_Name": [display_exp_name],
            "Target_Name": [t_name],
            "Direction": [t_dir],
            "Passive_Vars": [",".join(st.session_state.passive_vars)]
        }
        excel_bytes = build_excel_bytes(st.session_state.df_data, st.session_state.config_vars, meta_data)
        file_name_export = f"{display_exp_name}_Data.xlsx"

        st.download_button(label="📥 최신 데이터 Excel 다운로드", data=excel_bytes, file_name=file_name_export, type="primary", use_container_width=True)
        st.divider()
        if st.button("🛠️ 환경 설정으로 돌아가기", use_container_width=True):
            st.session_state.app_mode = "Setup"
            st.rerun()
        if st.button("⚠️ 모든 데이터 초기화", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["📝 신규 실험 입력", "🗂️ 데이터베이스 관리", "🤖 AI 최적화 대시보드"])

    with tab1:
        with st.container(border=True):
            colored_header(label="새로운 스플릿 실험 결과 입력", description="값을 모두 적은 후 하단 '데이터 추가' 버튼을 클릭하세요.", color_name="orange-70")
            with st.form("input_form", clear_on_submit=True):
                cols = st.columns(len(st.session_state.passive_vars) + len(st.session_state.config_vars) + 1)
                new_row = {"학습_적용": True}
                idx = 0
                
                label_style = "<div style='font-size: 14px; font-weight: 800; padding-bottom: 8px; color: #1a1a1a;'>{}</div>"
                
                for p_var in st.session_state.passive_vars:
                    with cols[idx]:
                        st.markdown(label_style.format(p_var), unsafe_allow_html=True)
                        new_row[p_var] = st.text_input(p_var, value="", label_visibility="collapsed", key=f"input_{p_var}")
                    idx += 1
                    
                for var in st.session_state.config_vars:
                    unit_str = f" ({var['Unit']})" if var.get("Unit") else ""
                    disp_name = f"{var['Name']}{unit_str}"
                    with cols[idx]:
                        st.markdown(label_style.format(disp_name), unsafe_allow_html=True)
                        if "Real" in var["Type"]:
                            new_row[var["Name"]] = st.number_input(disp_name, value=float(var["Min"]), step=0.1, label_visibility="collapsed", key=f"input_{var['Name']}")
                        elif "Integer" in var["Type"]:
                            new_row[var["Name"]] = st.number_input(disp_name, value=int(var["Min"]), step=1, label_visibility="collapsed", key=f"input_{var['Name']}")
                        elif "Categorical" in var["Type"]:
                            opts = [o.strip() for o in var["Options"].split(",")]
                            new_row[var["Name"]] = st.selectbox(disp_name, opts, label_visibility="collapsed", key=f"input_{var['Name']}")
                    idx += 1
                    
                with cols[idx]:
                    st.markdown(label_style.format(f"결과값 ({t_name})"), unsafe_allow_html=True)
                    new_row[t_name] = st.number_input(t_name, value=0.0, label_visibility="collapsed", key="input_target")
                
                st.write("") 
                submitted = st.form_submit_button("➕ 데이터 추가", type="primary", use_container_width=True)
                if submitted:
                    st.session_state.df_data = pd.concat([st.session_state.df_data, pd.DataFrame([new_row])], ignore_index=True)
                    st.success("데이터 저장 완료! 데이터베이스 관리 탭에서 확인하세요.")
                    st.rerun()

            if st.button("↩️ 마지막 입력 취소 (직전 데이터 삭제)"):
                if not st.session_state.df_data.empty:
                    st.session_state.df_data = st.session_state.df_data.iloc[:-1] 
                    st.success("가장 최근 데이터가 삭제되었습니다.")
                    st.rerun()
                else:
                    st.warning("삭제할 데이터가 없습니다.")

    with tab2:
        with st.container(border=True):
            colored_header(label="전체 실험 데이터 아카이브", description="입력 이력을 한눈에 검토하고 이상치 데이터의 AI 반영 여부를 수정할 수 있습니다.", color_name="orange-70")
            st.session_state.df_data = st.data_editor(
                st.session_state.df_data, 
                use_container_width=True, 
                hide_index=True, 
                column_config={"학습_적용": st.column_config.CheckboxColumn("학습 적용")}
            )

    with tab3:
        valid_df = st.session_state.df_data[st.session_state.df_data["학습_적용"] == True]
        c1, c2 = st.columns([1.2, 1])

        with c1:
            with st.container(border=True):
                colored_header(label=f"📈 최적화 경향 곡선", description=f"실험이 진행됨에 따라 타겟 지표({t_name})의 수렴 상태를 보여줍니다.", color_name="green-70")
                if len(valid_df) > 0:
                    chart_data = valid_df[t_name].expanding().max() if "Maximize" in t_dir else valid_df[t_name].expanding().min()
                    st.line_chart(chart_data, height=350)
                else:
                    st.info("분석용 데이터가 입력되지 않았습니다.")

        with c2:
            with st.container(border=True):
                colored_header(label="🤖 베이지안 추천 차기 조건", description="가우시안 프로세스 알고리즘에 기반하여 제안된 3가지 최적 조건 셋입니다.", color_name="orange-70")
                if st.button("🚀 AI 계산 실행", type="primary", use_container_width=True):
                    if len(valid_df) < 2:
                        st.warning("정밀 분석을 위해 최소 2개 이상의 유효 데이터가 필요합니다.")
                    else:
                        with st.spinner("알고리즘 연산 중..."):
                            X_train, y_train = process_robust_data(valid_df, f_names, t_name)
                            ai_spaces = []
                            for var in st.session_state.config_vars:
                                if "Real" in var["Type"]: ai_spaces.append(Real(var["Min"], var["Max"], name=var["Name"]))
                                elif "Integer" in var["Type"]: ai_spaces.append(Integer(var["Min"], var["Max"], name=var["Name"]))
                                elif "Categorical" in var["Type"]: ai_spaces.append(Categorical([o.strip() for o in var["Options"].split(",")], name=var["Name"]))
                            
                            y_train_fit = [-val for val in y_train] if "Maximize" in t_dir else y_train
                                
                            opt = Optimizer(dimensions=ai_spaces, base_estimator="GP", acq_func="EI", random_state=None)
                            
                            X_train_safe = []
                            y_train_fit_safe = []
                            for i, point in enumerate(X_train):
                                if all(ai_spaces[j].low <= val <= ai_spaces[j].high for j, val in enumerate(point)):
                                    X_train_safe.append(point)
                                    y_train_fit_safe.append(y_train_fit[i])
                            
                            opt.tell(X_train_safe, y_train_fit_safe)
                            next_points = opt.ask(n_points=3)
                        
                        if "prev_next_points" in st.session_state and st.session_state.prev_next_points == next_points:
                            st.info("💡 **AI 수렴 상태 판단:** 현재 입력된 데이터 풀 안에서 해당 지점이 가장 최적의 공정 조건 범위로 강력하게 매핑되었습니다.")
                        
                        st.session_state.prev_next_points = next_points
                            
                        for i, points in enumerate(next_points):
                            with st.container(border=True): 
                                st.markdown(f"<h5 style='margin:0; font-weight: 800; color: #ed542b;'>실험 후보 {i+1}</h5>", unsafe_allow_html=True)
                                st.divider()
                                cols_rec = st.columns(len(f_names))
                                for idx, (var, val) in enumerate(zip(st.session_state.config_vars, points)):
                                    unit_str = f" {var['Unit']}" if var.get("Unit") else ""
                                    cols_rec[idx].metric(label=var["Name"], value=f"{round(val, 3)}{unit_str}")
                                style_metric_cards(background_color="transparent", border_left_color="#ed542b", border_color="transparent", box_shadow=False)
                        
                        with st.expander("🔍 AI 연산 피팅 로그 데이터"):
                            debug_df = pd.DataFrame(X_train, columns=f_names)
                            debug_df[t_name] = y_train
                            st.dataframe(debug_df, use_container_width=True)