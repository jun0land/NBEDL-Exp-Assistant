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
from origin_charts import render_variable_charts, render_excluded_expander
from data_manage import render_data_manager
import analysis

# 다중 목표(2개 이상) 최적화에만 쓰는 BoTorch는 무겁고(torch 포함) 목표 1개짜리 사용자에게는
# 불필요하다. import를 감싸서, 미설치 상태에서도 단일 목표 경로(skopt)는 그대로 동작하고
# 다중 목표 탭에서만 안내 메시지를 띄운다.
try:
    import torch
    # 컨테이너는 CPU 쿼터가 1코어 안팎으로 작아도 torch는 기본적으로 호스트가 보고하는
    # 전체 코어 수만큼 스레드를 띄운다 — 작은 쿼터를 여러 스레드가 다투면서 실제 필요한
    # 것보다 훨씬 높은 CPU 사용률로 잡혀 Streamlit Cloud 스로틀을 유발한다. 스레드를
    # 고정해 피크 사용률을 낮춘다 (계산 시간은 다소 늘지만 스로틀보다는 낫다).
    torch.set_num_threads(1)
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import SumMarginalLogLikelihood
    # qNEHVI는 수치적으로 불안정할 수 있다는 BoTorch 자체 경고가 있어, 같은 API의
    # 로그 스케일 버전(qLogNEHVI)을 쓴다 — 안정성뿐 아니라 그래디언트가 더 잘 살아있어
    # 같은 예산(restarts/samples)으로도 더 적은 반복에 수렴하는 경향이 있다.
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    # 목표 5개 이상은 하이퍼볼륨 계산 자체가 조합적으로 폭발해(실측: 목표 2개 2초 -> 4개 20초
    # -> 8개는 8분 넘게도 안 끝남) qLogNEHVI를 못 쓴다. ParEGO 스타일 스칼라화(단일목표 EI를
    # 랜덤 가중치로 여러 번)로 자동 전환하기 위한 임포트.
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import normalize, unnormalize
    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False

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

/* 큰 모니터에서 블럭이 화면 끝까지 좌우로 늘어나면 한 줄이 너무 길어져 가시성이 떨어진다.
   콘텐츠 폭을 zoom 기준 디자인 폭(1440px)으로 제한하고 가운데 정렬한다. */
[data-testid="stMain"] .block-container {{
    max-width: 1440px !important;
    margin-left: auto !important;
    margin-right: auto !important;
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

/* 버튼 공통 투명화 및 Hover
   (일반 st.button 뿐 아니라 st.download_button/st.form_submit_button도 같은 스타일을
   받도록 .stDownloadButton, .stFormSubmitButton도 함께 지정한다. 안 그러면 이 두 위젯은
   커스텀 그라데이션을 못 받고 Streamlit 기본 빨간색(primary red)으로 튀어 보인다.) */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(20px) !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
    transform: translateY(-2px) !important;
    background: #ed542b !important;
    border-color: #ed542b !important;
    color: white !important;
    box-shadow: 0 8px 20px rgba(237,84,43,0.25) !important;
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #ed542b, #f68b21) !important;
    border: none !important;
    color: white !important;
}}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
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
<p>이 앱은 실험 조건과 결과를 입력하면, <b>AI(베이지안 최적화)</b>가 다음에 시도해볼 <b>최적의 공정 조건</b>을 추천해 주는 실험 보조 도구입니다. 자세한 분석 모델·원칙은 오른쪽 <b>📊 분석 방법</b> 배너를 참고하세요.</p>

<h4>🚀 단계별 따라하기</h4>

<div class="nbedl-step">STEP 1. 프로젝트 설정 <span class="nbedl-loc">· 실험 설정 화면</span></div>
<ul>
  <li>기존 데이터 파일(.xlsx)이 있으면 상단 <b>"기존 실험 데이터 불러오기"</b>에 올리면 설정·이상치 방법까지 복원됩니다.</li>
  <li>처음이면 직접 입력: <b>실험 이름</b>, <b>목표 지표</b>(예: <code>J_sc</code>)와 <b>방향</b>(최대화/최소화) — <b>목표를 2개 이상</b> 넣으면 자동으로 다중목표(파레토) 최적화, <b>환경 변수</b>(선택), <b>공정 변수</b>(이름·단위·타입·범위)를 <b>"➕"</b>로 추가.</li>
  <li><b>"🚀 실험 시작 및 대시보드 생성"</b> 클릭</li>
</ul>

<div class="nbedl-step">STEP 2. 실험 결과 입력 <span class="nbedl-loc">· 📝 신규 실험 입력 탭</span></div>
<ul>
  <li>샘플명·조건·결과값을 넣고 <b>"➕ 데이터 추가"</b> <span class="nbedl-tip">(Enter는 값 확정만, 추가는 버튼으로)</span></li>
  <li>잘못 넣었으면 <b>"↩️ 마지막 입력 취소"</b></li>
</ul>

<div class="nbedl-step">STEP 3. 데이터 관리 <span class="nbedl-loc">· 🗂️ 데이터베이스 관리 탭</span></div>
<ul>
  <li><b>검색</b>·<b>공정 변수별 필터</b>로 원하는 행만 좁히고, <b>전체 선택/해제</b>로 <b>"학습 적용"</b>을 일괄 관리(해제 = AI 학습에서 제외)</li>
  <li><b>필터된 행 일괄 삭제</b>, <b>표 높이</b> 조절(큰 모니터에서 넓게)</li>
  <li><b>요약 통계</b> — 변수별 범위·평균과, 모든 목표의 trade-off를 고려한 <b>종합 최적 조건</b> 표시</li>
</ul>

<div class="nbedl-step">STEP 4. 데이터 진단 <span class="nbedl-loc">· 🔬 데이터 진단 탭</span></div>
<ul>
  <li><b>이상치 판정 방법</b>(IQR·Z·MAD·ESD·없음)과 <b>유의수준 α</b> 선택 — 강건 평균·제외 목록·박스플롯에 함께 적용</li>
  <li><b>제외된 데이터</b>를 사유별로 확인, <b>박스플롯</b>으로 반복 측정 분포·이상치 확인</li>
  <li><b>목표별 수렴 곡선</b>으로 실험이 목표에 수렴하는지 확인</li>
</ul>

<div class="nbedl-step">STEP 5. AI 추천 받기 <span class="nbedl-loc">· 🤖 AI 최적화 대시보드 탭</span></div>
<ul>
  <li><b>"🚀 AI 계산 실행"</b> → 다음에 시도할 <b>추천 조건 3가지</b> (최소 2개 유효 데이터 필요)</li>
  <li><b>공정 변수별 그래프</b> — 표시 배율·정규화·추세선 조절, <b>PNG(투명)·JPG·CSV</b>로 내보내기(출판용 960×768)</li>
</ul>

<div class="nbedl-step">STEP 6. 저장 &amp; 관리 <span class="nbedl-loc">· 왼쪽 사이드바</span></div>
<ul>
  <li><b>"📥 최신 데이터 Excel 다운로드"</b>로 저장 → STEP 1에서 다시 올리면 이어서 작업(이상치 설정 포함 복원)</li>
  <li>"🛠️ 환경 설정으로 돌아가기" / "⚠️ 모든 데이터 초기화"</li>
</ul>

<h4>⚠️ 주의사항 &amp; 팁</h4>
<div class="nbedl-note">
  <ul>
    <li><b>자동 저장이 안 됩니다.</b> 데이터는 브라우저 세션에만 있어 새로고침·창 닫기 시 사라질 수 있습니다. <b>작업 후 반드시 Excel로 다운로드</b>하세요.</li>
    <li><b>"모든 데이터 초기화"는 되돌릴 수 없습니다.</b></li>
    <li><b>데이터가 많을수록 AI 추천이 정확</b>해집니다.</li>
    <li><b>이상치 제거</b>는 같은 조건 3회 이상 반복 그룹에서만 일어나며, 방법·민감도는 <b>🔬 데이터 진단</b> 탭에서 고릅니다. 원리는 <b>📊 분석 방법</b> 배너 참고.</li>
  </ul>
</div>
"""

METHODOLOGY_HTML = """
<p>이 앱이 쓰는 분석 모델과 원칙을 정리했습니다. 모든 계산은 <b>학습 적용된 데이터</b>만 사용합니다.</p>

<h4>① 이상치 처리 (반복 측정)</h4>
<p><b>완전히 같은 공정 조건</b>(모든 공정 변수가 일치)을 여러 번 측정한 그룹 안에서만 이상치를 판정합니다. 조건이 조금이라도 다르면 다른 그룹이라 서로 비교하지 않습니다. 그룹의 반복 수 <b>n이 3 미만이면 판정하지 않습니다</b>(통계적으로 무의미). "🔬 데이터 진단" 탭에서 방법을 고르며, <b>"자동 추천"</b>을 선택하면 데이터의 반복 수에 맞춰 아래에서 자동으로 골라 줍니다.</p>
<p><b>방법별 이상적 사용 조건 (반복 수 n 기준)</b></p>
<ul>
  <li><b>없음</b> — n&lt;3(반복 부족)이거나 이상치를 빼고 싶지 않을 때</li>
  <li><b>ESD (Rosner)</b> — <b>n ≈ 3~수십</b>. 정규 근사 가정의 정식 검정으로 <b>소표본(삼반복 등)에서 가장 실용적</b>. 유의수준 α로 민감도 조절</li>
  <li><b>수정 Z-점수 (MAD)</b> — <b>n ≳ 5</b>, 비정규·치우친 분포에 강건. 단 <b>동일값이 절반 이상이면 MAD=0</b>이 되어 무력(삼반복에 취약)</li>
  <li><b>IQR / 박스플롯</b> — <b>n ≳ 10</b> 권장. 사분위가 안정적일 때의 표준, 분포 가정 없음. <b>n=3에선 거의 못 잡음</b></li>
  <li><b>Z-점수</b> — <b>n ≳ 30 &amp; 정규분포</b>. 소표본에선 이상치가 표준편차를 부풀려 스스로를 가리는 마스킹으로 실패</li>
</ul>
<p><b>자동 추천 규칙:</b> 반복 그룹의 대표 크기(중앙값)를 보고 — n&lt;3 → 없음 · n≤4 → ESD · n 5~9 → ESD · n≥10 → IQR. 공정 변수가 많을수록 같은 조건 반복이 잘게 쪼개져 n이 작아지므로, 변수 수보다 실제 반복 수를 직접 봅니다.</p>
<p class="nbedl-tip">이상치 판정에 절대적 정답은 없습니다. 방법마다 결과가 달라질 수 있어 데이터 성격에 맞게 고르세요 — 그래서 데이터에 맞춘 자동 추천을 기본으로 둡니다.</p>

<h4>② 유의수준 α (ESD 전용)</h4>
<p>ESD는 α로 민감도를 조절합니다. <b>α가 클수록 이상치를 더 많이 제거</b>해 학습 데이터가 줄어듭니다. 기본 0.05, 프리셋 0.01, 직접 입력 가능하며 <b>Excel 파일에 저장</b>되어 재업로드 시 복원됩니다. (IQR·Z·MAD는 α 대신 관례 상수 1.5×·3·3.5 사용)</p>

<h4>③ 강건 평균</h4>
<p>같은 조건 반복값에서 이상치를 뺀 나머지의 <b>평균</b>을 그 조건의 대표값으로 씁니다. 값이 전부 이상치로 판정되면(무의미) 아무것도 빼지 않습니다.</p>

<h4>④ 단일 목표 최적화 — 베이지안 최적화</h4>
<p>목표가 1개면 <b>가우시안 프로세스(GP)</b>로 목표를 모델링하고 <b>기대 개선량(Expected Improvement)</b>이 큰 다음 조건을 제안합니다(skopt). "지금 최고보다 얼마나 더 좋아질 것으로 기대되는가"를 최대화합니다.</p>

<h4>⑤ 다중 목표 최적화 — MOBO (qNEHVI)</h4>
<p>목표가 2개 이상이면 상충하는 목표들의 <b>파레토 프론트</b>를 넓히는 조건을 찾습니다. 기대 개선량을 다차원으로 일반화한 <b>기대 하이퍼볼륨 개선(qNEHVI, BoTorch)</b>을 씁니다 — 목표가 1개면 EI와 같아지는 자연스러운 확장입니다.</p>

<h4>⑥ 추세선 — 다항 회귀</h4>
<p>공정 변수별 그래프의 추세선은 <b>최소제곱 다항 회귀</b>입니다(기본 1차=선형, 차수 조절 가능). 점들의 경향 요약일 뿐 인과를 뜻하지 않습니다.</p>

<h4>⑦ 정규화 &amp; 종합 최적 조건</h4>
<p>여러 목표를 한 그래프에 겹칠 때 각 목표를 <b>min–max로 0~1 정규화</b>합니다. "종합 최적 조건"은 각 목표를 방향(최대화/최소화)에 맞춰 0~1로 만든 <b>desirability</b>의 (가중)평균이 가장 높은 실험 조건 — 한쪽으로 치우치지 않은 trade-off 균형점을 고릅니다.</p>

<h4>⑧ 공정 변수의 상관(다중공선성)과 조건 슬라이스</h4>
<p>실험에서는 공정 변수들이 서로 <b>독립이 아니라 상관</b>되기 쉽습니다(예: 온도를 올리면 반응이 빨라져 시간도 함께 조절하게 됨). 통계적으로 입력 변수끼리 상관이 크면 <b>다중공선성</b>이라 하고, "목표 vs 변수 하나" 그래프에서 다른 변수들이 같이 움직여 <b>어느 변수의 효과인지 뒤섞이는 교란(confounding)</b>이 생깁니다.</p>
<p>실험 특성상 상관 자체는 피하기 어렵습니다. 그래서 공정 변수별 그래프의 <b>"🎛️ 다른 변수 고정 (조건 슬라이스)"</b>로 <b>나머지 변수를 좁은 범위/특정 값에 고정</b>한 채 한 변수만 훑어봅니다 — 다른 조건이 비슷한 점끼리만 비교해 그 변수의 효과를 더 또렷하게 봅니다. 각 그래프는 자기 X축 변수만 자유롭게 두고 나머지 제약을 적용합니다.</p>
<p class="nbedl-tip">범위를 좁힐수록 교란은 줄지만 남는 점이 적어져 노이즈에 민감해집니다. 조금씩 넓혀 가며 균형을 찾으세요. (근본 해결은 실험 설계 단계에서 변수를 직교화하는 것)</p>
"""

DRAWER_CSS_TMPL = """
#__ROOT__, #__ROOT__ * { box-sizing: border-box; }
#__ROOT__ { font-family: 'Pretendard', sans-serif; }
#__ROOT__ .nbedl-bookmark {
  position: fixed; top: __TOP__; right: 0; z-index: 2147483401;
  display: flex; align-items: center; justify-content: center;
  padding: 18px 9px; writing-mode: vertical-rl; text-orientation: mixed;
  background: linear-gradient(160deg, __GRADA__, __GRADB__); color: #fff;
  font-weight: 800; letter-spacing: 2px; font-size: 15px;
  border-radius: 14px 0 0 14px; box-shadow: -4px 6px 18px rgba(0,0,0,0.22);
  cursor: pointer; user-select: none;
  transition: padding-right .18s ease, box-shadow .18s ease;
}
#__ROOT__ .nbedl-bookmark:hover { padding-right: 15px; box-shadow: -7px 8px 24px rgba(0,0,0,0.32); }
#__ROOT__ .nbedl-backdrop {
  position: fixed; inset: 0; z-index: 2147483500;
  background: rgba(20,20,20,0.30);
  -webkit-backdrop-filter: blur(3px); backdrop-filter: blur(3px);
  opacity: 0; pointer-events: none; transition: opacity .35s ease;
}
#__ROOT__ .nbedl-panel {
  position: fixed; top: 0; bottom: 0; right: 0; width: min(460px, 92%);
  z-index: 2147483501; overflow-y: auto; padding: 30px 30px 44px;
  background: rgba(255,255,255,0.94);
  -webkit-backdrop-filter: blur(26px) saturate(160%); backdrop-filter: blur(26px) saturate(160%);
  border-left: 6px solid __ACCENT__; box-shadow: -18px 0 50px rgba(0,0,0,0.20);
  transform: translateX(106%); transition: transform .38s cubic-bezier(.22,.61,.36,1);
  color: #1a1a1a;
}
#__ROOT__.open .nbedl-backdrop { opacity: 1; pointer-events: auto; }
#__ROOT__.open .nbedl-panel { transform: translateX(0); }
#__ROOT__ .nbedl-close {
  position: absolute; top: 16px; right: 18px; width: 34px; height: 34px;
  border: none; border-radius: 10px; background: rgba(0,0,0,0.06); color: #333;
  font-size: 15px; cursor: pointer; transition: background .2s ease, color .2s ease;
}
#__ROOT__ .nbedl-close:hover { background: __ACCENT__; color: #fff; }
#__ROOT__ .nbedl-title { font-size: 21px; font-weight: 900; color: __ACCENT__; margin: 2px 40px 18px 0;
  border-bottom: 2px solid rgba(0,0,0,0.12); padding-bottom: 12px; }
#__ROOT__ .nbedl-panel h4 { font-size: 16px; font-weight: 800; color: __ACCENT__; margin: 24px 0 8px; }
#__ROOT__ .nbedl-panel p { line-height: 1.65; margin: 8px 0; }
#__ROOT__ .nbedl-panel ul { margin: 6px 0 14px; padding-left: 20px; }
#__ROOT__ .nbedl-panel li { line-height: 1.6; margin: 5px 0; }
#__ROOT__ .nbedl-step { font-weight: 800; margin: 18px 0 4px; color: #1a1a1a; }
#__ROOT__ .nbedl-loc { font-weight: 600; color: #9a8f89; font-size: 0.9em; }
#__ROOT__ .nbedl-tip { color: __ACCENT__; font-weight: 700; font-size: 0.9em; }
#__ROOT__ .nbedl-panel code { background: rgba(0,0,0,0.06); color: #c53a17; padding: 1px 6px; border-radius: 6px; font-size: 0.9em; }
#__ROOT__ .nbedl-note { background: rgba(255,244,235,0.85); border-left: 4px solid #f68b21; border-radius: 10px; padding: 14px 16px; margin-top: 16px; }
#__ROOT__ .nbedl-note ul { margin: 0; }
"""

def render_side_drawer(root_id, *, top_css, grad_a, grad_b, accent, bookmark_label, panel_title, content_html):
    """오른쪽 모서리 북마크 + 슬라이드-인 패널을 부모 문서에 주입한다(여러 개 가능).
    st.markdown은 <script>를 제거하므로 components.html(iframe)에서 JS로 주입한다.
    CSS는 #root_id 로 스코프되어 드로어끼리 충돌하지 않는다."""
    css = (DRAWER_CSS_TMPL.replace("__ROOT__", root_id).replace("__TOP__", top_css)
           .replace("__GRADA__", grad_a).replace("__GRADB__", grad_b).replace("__ACCENT__", accent))
    payload = """
<script>
(function() {
  try {
    var doc = window.parent.document;
    var ROOT_ID = '__ROOTID__';
    var STYLE_ID = ROOT_ID + '-style';
    var oldRoot = doc.getElementById(ROOT_ID);
    var wasOpen = !!(oldRoot && oldRoot.classList.contains('open'));
    if (oldRoot) oldRoot.remove();
    var oldStyle = doc.getElementById(STYLE_ID); if (oldStyle) oldStyle.remove();

    var style = doc.createElement('style'); style.id = STYLE_ID;
    style.textContent = `__CSS__`; doc.head.appendChild(style);

    var root = doc.createElement('div'); root.id = ROOT_ID; root.className = 'nbedl-drawer';
    root.innerHTML = `
      <div class="nbedl-bookmark" title="__LABEL__"><span>__LABEL__</span></div>
      <div class="nbedl-backdrop"></div>
      <aside class="nbedl-panel" role="dialog" aria-label="__LABEL__">
        <button class="nbedl-close" title="닫기 (Esc)">✕</button>
        <div class="nbedl-title">__TITLE__</div>
        __CONTENT__
      </aside>`;
    doc.body.appendChild(root);
    if (wasOpen) root.classList.add('open');

    var openFn = function() {
      var all = doc.querySelectorAll('.nbedl-drawer.open');
      for (var i = 0; i < all.length; i++) all[i].classList.remove('open');
      root.classList.add('open');
    };
    var closeFn = function() { root.classList.remove('open'); };
    root.querySelector('.nbedl-bookmark').addEventListener('click', openFn);
    root.querySelector('.nbedl-backdrop').addEventListener('click', closeFn);
    root.querySelector('.nbedl-close').addEventListener('click', closeFn);

    if (!window.parent.__nbedlEsc) {
      window.parent.__nbedlEsc = true;
      doc.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
          var op = doc.querySelectorAll('.nbedl-drawer.open');
          for (var i = 0; i < op.length; i++) op[i].classList.remove('open');
        }
      });
    }
  } catch (err) { /* cross-origin 등 접근 불가 시 무시 */ }
})();
</script>
"""
    payload = (payload.replace("__ROOTID__", root_id).replace("__CSS__", css)
               .replace("__LABEL__", bookmark_label).replace("__TITLE__", panel_title)
               .replace("__CONTENT__", content_html))
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
    """화면 크기에 따라 UI를 '축소'한다. (공간만 늘어나는 리플로우 대신 스케일)
    기준 1440px에서 1.0배, 0.85~1.0배로 클램프.
    zoom은 transform:scale과 달리 재레이아웃이라 글자가 뭉개지지 않는다.

    MAX를 1.35가 아닌 1.0으로 클램프한 이유: 1440px보다 넓은(큰 모니터) 화면에서
    확대까지 하면 UI가 과도하게 커진다는 피드백이 있었다. 축소(좁은 화면에서 0.85배)는
    허용하되 1440px 이상에서는 더 키우지 않고 원래 크기(1.0)로 고정한다.
    (Photodetector-app의 동일 버그 수정을 그대로 반영: pd_app/theme.py 참고)

    주의: body 전체에 zoom을 걸면 Streamlit 내부의 100vh 기반 레이아웃(stMain 등)과
    충돌한다(zoom은 vh를 보정하지 않아 부모보다 커져 화면이 위로 밀림). 그래서 vh 의존이
    없는 콘텐츠 컨테이너와 드로어에만 적용한다.
    또한 인라인 스타일은 리런 시 DOM이 교체되며 사라지므로 <style> 규칙으로 주입한다."""
    components.html("""
<script>
(function() {
  try {
    var win = window.parent, doc = win.document;
    var DESIGN = 1440, MIN = 0.85, MAX = 1.0;
    var ID = 'nbedl-zoom-style';
    var st = doc.getElementById(ID);
    if (!st) { st = doc.createElement('style'); st.id = ID; doc.head.appendChild(st); }
    win.__nbedlApplyZoom = function() {
      var z = win.innerWidth / DESIGN;
      z = Math.max(MIN, Math.min(MAX, z));
      doc.body.style.zoom = '';   // 과거 body-zoom 방식 잔재 제거
      st.textContent =
        '[data-testid="stMain"] .block-container { zoom: ' + z + '; }' +
        '.nbedl-drawer { zoom: ' + z + '; }';
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
render_side_drawer("nbedl-manual-root", top_css="20%", grad_a="#ed542b", grad_b="#f68b21", accent="#ed542b",
                   bookmark_label="📖 사용 설명서", panel_title="📖 NBEDL Exp Assistant 사용 설명서", content_html=MANUAL_HTML)
render_side_drawer("nbedl-method-root", top_css="46%", grad_a="#0c8599", grad_b="#20c997", accent="#0c8599",
                   bookmark_label="📊 분석 방법", panel_title="📊 분석 방법 · 모델과 원칙", content_html=METHODOLOGY_HTML)
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
if "target_vars" not in st.session_state:
    st.session_state.target_vars = []
if "passive_vars" not in st.session_state:
    st.session_state.passive_vars = []
if "df_data" not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if "outlier_method" not in st.session_state:
    st.session_state.outlier_method = "auto"
if "outlier_alpha" not in st.session_state:
    st.session_state.outlier_alpha = 0.05

# ==========================================
# 5. 전처리 및 엑셀 로드 함수
# ==========================================
def coerce_bool_col(series):
    """'학습_적용' 같은 체크박스 컬럼을 깨끗한 bool로 강제한다.

    Excel 왕복(openpyxl round-trip)이나 concat 중 dtype 추론 과정에서 이 컬럼이
    NaN/문자열("TRUE"/"FALSE")/숫자(0,1) 등 bool이 아닌 값으로 섞이면
    st.column_config.CheckboxColumn이 StreamlitAPIException을 던진다. 또한 dtype이
    깨끗하지 않으면 `df["학습_적용"] == True` 비교도 조용히 틀린 값을 걸러낼 수 있어
    표시 전뿐 아니라 필터링 전에도 항상 이 함수를 거쳐야 한다.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(True)

    def _conv(v):
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return True
        if isinstance(v, (int, float)):
            return bool(v)
        return str(v).strip().lower() not in ("false", "0", "", "none", "nan")

    return series.map(_conv).astype(bool)

def process_robust_data(df, feature_cols, target_col):
    grouped = df.groupby(feature_cols)
    robust_X, robust_y = [], []
    for name, group in grouped:
        y_vals = pd.to_numeric(group[target_col], errors="coerce").dropna().tolist()
        if not y_vals:
            continue
        x_val = list(name) if isinstance(name, tuple) else [name]
        robust_X.append(x_val)
        robust_y.append(np.mean(y_vals))
    return robust_X, robust_y

def process_robust_data_multi(df, feature_cols, target_cols):
    """process_robust_data와 같은 조건별 단순 평균을 목표 지표 여러 개에 동시에 적용한다.
    같은 X(공정 조건)로 한 번만 그룹핑해서 목표별 평균을 나란히 계산 — 다중목표 경로 전용."""
    grouped = df.groupby(feature_cols)
    robust_X, robust_Y = [], []
    for name, group in grouped:
        row_y = []
        for t_col in target_cols:
            yv = pd.to_numeric(group[t_col], errors="coerce").dropna().tolist()
            row_y.append(np.mean(yv) if yv else float("nan"))
        x_val = list(name) if isinstance(name, tuple) else [name]
        robust_X.append(x_val)
        robust_Y.append(row_y)
    return robust_X, robust_Y

MOBO_HYPERVOLUME_MAX_OBJECTIVES = 4  # 이보다 목표가 많으면 스칼라화(ParEGO) 경로로 자동 전환

def run_mobo(X_train, Y_train, config_vars, directions, n_candidates=3):
    """다중목표 베이지안 최적화 (BoTorch).

    단일목표 EI는 "현재 최고값 대비 개선 기댓값"을 계산하는데, 이를 다차원으로 일반화한 것이
    기대 하이퍼볼륨 개선량(EHVI)이다 — 목표가 1개면 하이퍼볼륨은 구간 길이가 되어 EI와 정확히
    같은 식으로 축소되므로 원리는 동일하다(목표 1개는 skopt 경로를 그대로 쓴다). 목표가
    MOBO_HYPERVOLUME_MAX_OBJECTIVES개 이하일 때는 파레토 프론트(하이퍼볼륨)를 가장 넓히는
    다음 실험 후보를 정확히 찾는다(qLogNEHVI).

    목표가 그보다 많으면 하이퍼볼륨 계산 자체가 조합적으로 폭발한다 — 실측 기준 목표 2개는
    2초, 4개는 20초, 8개는 8분이 지나도 안 끝났다(Streamlit Cloud CPU 스로틀의 직접 원인).
    이 경우 ParEGO 방식(Knowles 2006)으로 전환한다: 후보마다 서로 다른 랜덤 가중치로 목표들을
    체비셰프 스칼라화해 하나의 값으로 합친 뒤, 그 값에 대한 평범한 단일목표 EI를 최적화한다.
    하이퍼볼륨/파레토 분할 계산이 전혀 없어 목표 개수에 비례해서만 느려지고, 후보마다 가중치가
    달라 파레토 프론트의 서로 다른 지점을 겨냥하므로 여전히 다양한 트레이드오프를 보여준다.

    Categorical 변수는 옵션 인덱스(0..k-1)로 연속 인코딩한 뒤 반올림해서 되돌린다 — 완전한
    혼합공간 최적화(optimize_acqf_mixed)보다 단순하지만, 후보 3개를 뽑는 실험 추천 용도로는
    충분한 근사다.

    directions: target_cols와 같은 순서의 "Maximize"/"Minimize" 리스트.
    반환: (candidates_raw, predicted_Y_raw) - 둘 다 원래 단위(부호 반전 없이).
    """
    torch.manual_seed(0)

    specs = []  # (low, high, kind, options_or_None)
    for var in config_vars:
        if "Real" in var["Type"]:
            specs.append((float(var["Min"]), float(var["Max"]), "real", None))
        elif "Integer" in var["Type"]:
            specs.append((float(var["Min"]), float(var["Max"]), "integer", None))
        else:
            opts = [o.strip() for o in var["Options"].split(",")]
            specs.append((0.0, float(max(len(opts) - 1, 0)), "categorical", opts))

    def encode_x(point):
        enc = []
        for val, (lo, hi, kind, opts) in zip(point, specs):
            enc.append(float(opts.index(val)) if kind == "categorical" and val in opts else float(val) if kind != "categorical" else 0.0)
        return enc

    def decode_x(vec):
        out = []
        for val, (lo, hi, kind, opts) in zip(vec, specs):
            if kind == "categorical":
                idx = int(round(max(lo, min(hi, val))))
                out.append(opts[idx])
            elif kind == "integer":
                out.append(int(round(val)))
            else:
                out.append(float(val))
        return out

    bounds_raw = torch.tensor([[s[0] for s in specs], [s[1] for s in specs]], dtype=torch.double)

    X_raw = torch.tensor([encode_x(p) for p in X_train], dtype=torch.double)
    Y_raw = torch.tensor(Y_train, dtype=torch.double)
    # Minimize 목표는 부호를 반전해 전부 "최대화" 기준으로 통일한다 (BoTorch 관례).
    sign = torch.tensor([1.0 if "Maximize" in d else -1.0 for d in directions], dtype=torch.double)
    Y_adj = Y_raw * sign

    X_norm = normalize(X_raw, bounds=bounds_raw)

    models = [SingleTaskGP(X_norm, Y_adj[:, i:i + 1], outcome_transform=Standardize(m=1))
              for i in range(Y_adj.shape[-1])]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    n_obj = Y_adj.shape[-1]
    standard_bounds = torch.zeros(2, X_norm.shape[-1], dtype=torch.double)
    standard_bounds[1] = 1.0
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))

    if n_obj <= MOBO_HYPERVOLUME_MAX_OBJECTIVES:
        # 정확한 파레토 하이퍼볼륨 기반 (qLogNEHVI) — 목표가 적을 때만 감당 가능.
        y_range = (Y_adj.max(dim=0).values - Y_adj.min(dim=0).values).clamp(min=1e-6)
        ref_point = Y_adj.min(dim=0).values - 0.1 * y_range
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point.tolist(), X_baseline=X_norm,
            sampler=sampler, prune_baseline=True,
        )
        candidates_norm, _ = optimize_acqf(
            acq_function=acq, bounds=standard_bounds, q=n_candidates,
            num_restarts=5, raw_samples=128,
            options={"batch_limit": 5, "maxiter": 200}, sequential=True,
        )
    else:
        # ParEGO 스칼라화 — 후보마다 다른 랜덤 가중치로 목표를 하나의 값으로 합쳐
        # 평범한 단일목표 EI를 최적화한다 (하이퍼볼륨 계산 없음 -> 목표 개수에 선형).
        cand_rows = []
        for i in range(n_candidates):
            gen = torch.Generator().manual_seed(i)
            weights = torch.rand(n_obj, generator=gen, dtype=torch.double)
            weights = weights / weights.sum()
            scalarized_obj = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=Y_adj))
            acq = qLogNoisyExpectedImprovement(
                model=model, X_baseline=X_norm, objective=scalarized_obj,
                sampler=sampler, prune_baseline=True,
            )
            cand, _ = optimize_acqf(
                acq_function=acq, bounds=standard_bounds, q=1,
                num_restarts=5, raw_samples=128,
                options={"batch_limit": 5, "maxiter": 200},
            )
            cand_rows.append(cand)
        candidates_norm = torch.cat(cand_rows, dim=0)

    candidates_raw_t = unnormalize(candidates_norm, bounds=bounds_raw)

    with torch.no_grad():
        posterior_mean_adj = model.posterior(candidates_norm).mean
    posterior_mean_raw = posterior_mean_adj * sign

    candidates_raw = [decode_x(row.tolist()) for row in candidates_raw_t]
    predicted_Y = posterior_mean_raw.tolist()
    return candidates_raw, predicted_Y

@st.cache_data(show_spinner=False, max_entries=3)
def build_excel_bytes(df, config_vars, target_vars, meta_data):
    """Excel 바이트 생성. 다운로드 버튼 때문에 매 리런마다 재생성되던 것을 캐시한다.
    (데이터가 바뀌면 자동으로 다시 생성된다)"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        pd.DataFrame(config_vars).to_excel(writer, sheet_name='Config_Vars', index=False)
        pd.DataFrame(target_vars).to_excel(writer, sheet_name='Target_Vars', index=False)
        pd.DataFrame(meta_data).to_excel(writer, sheet_name='Config_Meta', index=False)
    return output.getvalue()

def load_excel_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
    df_meta = pd.read_excel(xls, 'Config_Meta')

    if 'Target_Vars' in xls.sheet_names:
        target_list = pd.read_excel(xls, 'Target_Vars').to_dict('records')
    else:
        # 구버전 파일 호환: 목표 지표가 1개뿐이던 시절의 Config_Meta 형식
        target_list = [{"Name": df_meta.iloc[0]['Target_Name'], "Direction": df_meta.iloc[0]['Direction']}]
    for tv in target_list:
        if "Old_Name" not in tv or pd.isna(tv["Old_Name"]):
            tv["Old_Name"] = tv.get("Name", "")
    st.session_state.target_vars = target_list

    if 'Exp_Name' in df_meta.columns and pd.notna(df_meta.iloc[0]['Exp_Name']):
        st.session_state.exp_name = str(df_meta.iloc[0]['Exp_Name'])

    if "Outlier_Method" in df_meta.columns and pd.notna(df_meta.iloc[0]["Outlier_Method"]):
        st.session_state.outlier_method = str(df_meta.iloc[0]["Outlier_Method"])
    if "Outlier_Alpha" in df_meta.columns and pd.notna(df_meta.iloc[0]["Outlier_Alpha"]):
        st.session_state.outlier_alpha = float(df_meta.iloc[0]["Outlier_Alpha"])

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
    col_title, col_upload = st.columns([2.6, 1.2], gap="medium", vertical_alignment="center")

    with col_title:
        st.markdown(f'<div class="title-glass-container">{logo_html}<h2>NBEDL AI 기반 공정 최적화 시스템</h2></div>', unsafe_allow_html=True)

    with col_upload:
        with st.popover("📂 기존 실험 데이터 불러오기", use_container_width=True):
            uploaded_file = st.file_uploader("엑셀(xlsx) 파일을 업로드하면 설정이 자동으로 채워집니다.", type=["xlsx"])
            if uploaded_file:
                load_excel_data(uploaded_file)
                st.rerun()

    col_basic, col_target = st.columns(2, gap="medium")

    with col_basic:
        with st.container(border=True):
            colored_header(label="기본 프로젝트 설정", description="실험 이름과 환경 변수를 설정하세요.", color_name="orange-70")
            st.session_state.exp_name = st.text_input("📝 실험 프로젝트 이름", value=st.session_state.exp_name, placeholder="예: NBEDL_Experiment_01")

            passive_val = ",".join(st.session_state.passive_vars) if st.session_state.passive_vars else ""
            passive_input = st.text_input("환경 변수 (쉼표 구분)", value=passive_val, placeholder="예: 온도 (°C), 습도 (%)")

    with col_target:
        with st.container(border=True):
            colored_header(label="🎯 최적화 목표 지표 설정", description="목표 지표를 여러 개 등록하면, 1개일 땐 단일목표 최적화를, 2개 이상이면 파레토 최적(MOBO) 탐색을 자동으로 수행합니다.", color_name="orange-70")

            for i, tv in enumerate(st.session_state.target_vars):
                with st.container(border=True):
                    ct1, ct_u, ct2 = st.columns([2, 1, 1])
                    tv["Name"] = ct1.text_input(f"목표 지표 {i+1} 이름", value=tv.get("Name", ""), key=f"tname_{i}", placeholder="예: J_sc")
                    tv["Unit"] = ct_u.text_input("단위", value=tv.get("Unit", ""), key=f"tunit_{i}", placeholder="예: mA/cm²")
                    dir_options = ["Maximize", "Minimize"]
                    safe_dir = tv.get("Direction", "Maximize")
                    if safe_dir not in dir_options: safe_dir = "Maximize"
                    tv["Direction"] = ct2.selectbox("최적화 방향", dir_options, key=f"tdir_{i}", index=dir_options.index(safe_dir))

            if st.button("➕ 목표 지표 블럭 추가", use_container_width=True):
                st.session_state.target_vars.append({"Old_Name": "", "Name": "", "Unit": "", "Direction": "Maximize"})
                st.rerun()

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
        if not st.session_state.target_vars or any(not tv["Name"].strip() for tv in st.session_state.target_vars):
            st.error("목표 지표를 최소 1개 이상, 이름을 채워서 등록해야 합니다.")
        elif not st.session_state.config_vars:
            st.error("최소 1개 이상의 공정 변수를 추가해야 합니다.")
        else:
            p_vars = [v.strip() for v in passive_input.split(",") if v.strip()]
            st.session_state.passive_vars = p_vars

            if not st.session_state.df_data.empty:
                rename_dict = {}
                for var in st.session_state.config_vars + st.session_state.target_vars:
                    old = var.get("Old_Name", "")
                    new = var["Name"]
                    if old and old != new and old in st.session_state.df_data.columns:
                        rename_dict[old] = new
                if rename_dict:
                    st.session_state.df_data.rename(columns=rename_dict, inplace=True)

            for var in st.session_state.config_vars + st.session_state.target_vars:
                var["Old_Name"] = var["Name"]

            new_cols = (["학습_적용", "샘플명"] + p_vars + [v["Name"] for v in st.session_state.config_vars]
                        + [tv["Name"] for tv in st.session_state.target_vars])

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
    if "학습_적용" in st.session_state.df_data.columns:
        st.session_state.df_data["학습_적용"] = coerce_bool_col(st.session_state.df_data["학습_적용"])

    # Data 시트(실제 데이터 테이블)의 컬럼 순서는 최초 생성 시점에 고정되어 이후 절대
    # 바뀌지 않는다. 반면 target_vars(=Target_Vars 시트)는 엑셀에서 직접 정렬/편집하면
    # 그 순서가 어긋날 수 있다 — 화면 표시는 항상 실제 데이터 컬럼 순서를 따르도록,
    # target_vars를 df_data.columns 등장 순서에 맞춰 정렬한다 (아직 데이터에 반영 안 된
    # 신규 목표는 뒤로 보낸다).
    _data_cols = list(st.session_state.df_data.columns)
    st.session_state.target_vars.sort(
        key=lambda tv: _data_cols.index(tv["Name"]) if tv["Name"] in _data_cols else len(_data_cols)
    )

    target_names_all = [tv["Name"] for tv in st.session_state.target_vars]
    f_names = [v["Name"] for v in st.session_state.config_vars]
    
    display_exp_name = st.session_state.exp_name if st.session_state.exp_name.strip() else "NBEDL_Experiment"
    
    st.markdown(f'<div class="title-glass-container">{logo_html}<h2>NBEDL Exp Assistant : {display_exp_name}</h2></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📂 데이터 관리 패널")
        meta_data = {
            "Exp_Name": [display_exp_name],
            "Passive_Vars": [",".join(st.session_state.passive_vars)],
            "Outlier_Method": [st.session_state.outlier_method],
            "Outlier_Alpha": [st.session_state.outlier_alpha]
        }
        excel_bytes = build_excel_bytes(st.session_state.df_data, st.session_state.config_vars, st.session_state.target_vars, meta_data)
        file_name_export = f"{display_exp_name}_Data.xlsx"

        st.download_button(label="📥 최신 데이터 Excel 다운로드", data=excel_bytes, file_name=file_name_export, type="primary", use_container_width=True)
        st.divider()
        if st.button("🛠️ 환경 설정으로 돌아가기", use_container_width=True):
            st.session_state.app_mode = "Setup"
            st.rerun()
        if st.button("⚠️ 모든 데이터 초기화", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["📝 신규 실험 입력", "🗂️ 데이터베이스 관리", "🔬 데이터 진단", "🤖 AI 최적화 대시보드"])

    with tab1:
        with st.container(border=True):
            colored_header(label="새로운 스플릿 실험 결과 입력", description="값을 모두 적은 후 하단 '데이터 추가' 버튼을 클릭하세요.", color_name="orange-70")
            with st.form("input_form", clear_on_submit=True):
                cols = st.columns(1 + len(st.session_state.passive_vars) + len(st.session_state.config_vars) + len(target_names_all))
                new_row = {"학습_적용": True}
                idx = 0

                label_style = "<div style='font-size: 14px; font-weight: 800; padding-bottom: 8px; color: #1a1a1a;'>{}</div>"

                with cols[idx]:
                    st.markdown(label_style.format("샘플명"), unsafe_allow_html=True)
                    new_row["샘플명"] = st.text_input("샘플명", value="", label_visibility="collapsed", key="input_sample_name")
                idx += 1

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
                    
                for tv in st.session_state.target_vars:
                    t_var_name = tv["Name"]
                    unit_str = f", {tv['Unit']}" if tv.get("Unit") else ""
                    with cols[idx]:
                        st.markdown(label_style.format(f"결과값 ({t_var_name}{unit_str})"), unsafe_allow_html=True)
                        new_row[t_var_name] = st.number_input(t_var_name, value=0.0, step=0.000001, format="%.8f", label_visibility="collapsed", key=f"input_target_{t_var_name}")
                    idx += 1

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
            colored_header(label="전체 실험 데이터 아카이브", description="검색·필터로 원하는 조건을 좁히고, 전체 선택/해제로 학습 적용을 일괄 관리하세요.", color_name="orange-70")
            render_data_manager(st.session_state.config_vars, st.session_state.target_vars, st.session_state.passive_vars)

    with tab3:
        with st.container(border=True):
            colored_header(label="🔬 이상치 판정 설정", description="이상치 판정 방법과 유의수준을 정합니다. 강건 평균·제외 목록·박스플롯에 함께 적용됩니다.", color_name="orange-70")
            m, a = analysis.render_method_controls(st.session_state.outlier_method, st.session_state.outlier_alpha, st.session_state.df_data, f_names, key_prefix="diag")
            st.session_state.outlier_method = m
            st.session_state.outlier_alpha = a
            eff_method = analysis.resolve_method(m, st.session_state.df_data, f_names)
        with st.container(border=True):
            colored_header(label="🔎 이상치 검토 (추천 → 직접 결정)", description="알고리즘이 이상치로 추천한 데이터를 보고, 학습에서 뺄지 직접 정합니다. 자동 제거하지 않습니다.", color_name="orange-70")
            analysis.render_outlier_review(f_names, target_names_all, eff_method, st.session_state.outlier_alpha, key_prefix="diag")
        with st.container(border=True):
            colored_header(label="📦 반복 측정 분포 (박스플롯)", description="같은 조건 반복 측정의 분포와 이상치를 봅니다. (학습 적용 데이터만 — 데이터베이스 관리에서 체크 해제한 행은 빠집니다)", color_name="green-70")
            _valid_box = st.session_state.df_data[st.session_state.df_data["학습_적용"] == True]
            analysis.render_boxplots(_valid_box, st.session_state.config_vars, st.session_state.target_vars, eff_method, st.session_state.outlier_alpha, key_prefix="diagbox")
        with st.container(border=True):
            colored_header(label="📈 목표별 수렴 곡선", description="실험이 진행됨에 따라 각 목표가 어떻게 수렴하는지 봅니다. (학습 적용 데이터)", color_name="green-70")
            _valid_conv = st.session_state.df_data[st.session_state.df_data["학습_적용"] == True]
            if len(_valid_conv) > 0:
                for tv in st.session_state.target_vars:
                    tn, td = tv["Name"], tv.get("Direction", "Maximize")
                    if tn in _valid_conv.columns:
                        cdata = _valid_conv[tn].expanding().max() if "Maximize" in td else _valid_conv[tn].expanding().min()
                        st.caption(f"{tn} ({td})")
                        st.line_chart(cdata, height=200)
            else:
                st.info("분석용 데이터가 입력되지 않았습니다.")

    with tab4:
        analysis.render_applied_exclusions(f_names, target_names_all, key_prefix="ai")
        if not target_names_all:
            st.warning("등록된 목표 지표가 없습니다. 사이드바의 '환경 설정으로 돌아가기'에서 목표 지표를 추가하세요.")

        elif len(target_names_all) == 1:
            # ---- 목표 지표 1개: 기존 단일목표 경로 (skopt GP + EI) ----
            t_name = target_names_all[0]
            t_dir = st.session_state.target_vars[0]["Direction"]
            t_unit = st.session_state.target_vars[0].get("Unit", "")
            t_label = f"{t_name}, {t_unit}" if t_unit else t_name

            valid_df = st.session_state.df_data[st.session_state.df_data["학습_적용"] == True]

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

                            X_train_safe = []
                            y_train_fit_safe = []
                            for i, point in enumerate(X_train):
                                if all(ai_spaces[j].low <= val <= ai_spaces[j].high for j, val in enumerate(point)):
                                    X_train_safe.append(point)
                                    y_train_fit_safe.append(y_train_fit[i])

                            # 유효 데이터가 전부 공정 변수 설정 범위(Min~Max) 밖이면 학습 데이터가
                            # 텅 비어 skopt.Optimizer.tell()이 내부에서 np.argmin([])으로 죽는다 —
                            # 계산 전에 걸러서 사용자에게 원인을 알려준다.
                            next_points = None
                            if X_train_safe:
                                opt = Optimizer(dimensions=ai_spaces, base_estimator="GP", acq_func="EI", random_state=None)
                                opt.tell(X_train_safe, y_train_fit_safe)
                                next_points = opt.ask(n_points=3)

                        if next_points is None:
                            st.error("등록된 유효 데이터가 모두 공정 변수의 설정 범위(최소~최대값) 밖에 있어 계산할 수 없습니다. 환경 설정에서 범위를 확인하거나 데이터를 다시 확인하세요.")
                        else:
                            # 목표 지표별로 직전 결과를 따로 기억한다 (다른 지표로 전환 후 재계산했을 때
                            # 이전 지표의 결과와 잘못 비교되지 않도록).
                            if "prev_next_points_by_target" not in st.session_state:
                                st.session_state.prev_next_points_by_target = {}
                            if st.session_state.prev_next_points_by_target.get(t_name) == next_points:
                                st.info("💡 **AI 수렴 상태 판단:** 현재 입력된 데이터 풀 안에서 해당 지점이 가장 최적의 공정 조건 범위로 강력하게 매핑되었습니다.")

                            st.session_state.prev_next_points_by_target[t_name] = next_points

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

        else:
            # ---- 목표 지표 2개 이상: 다중목표 베이지안 최적화 (MOBO, qNEHVI/BoTorch) ----
            # 원리는 단일목표 EI와 동일한 "기대 개선량" 개념을 하이퍼볼륨으로 일반화한 것뿐이라
            # (목표 1개면 EI로 정확히 축소됨) 위 skopt 경로와 별개 알고리즘이 아니라 자연스러운
            # 확장이다. 다만 여러 목표를 동시에 보므로 "하나의 목표를 골라 계산"하는 개념 자체가
            # 없어 위의 목표 선택 UI는 여기선 쓰지 않는다.
            target_labels = ", ".join(f"{tv['Name']}({tv['Direction']})" for tv in st.session_state.target_vars)
            st.info(f"🎯 다중 목표 동시 최적화 (파레토 최적) — 등록된 {len(target_names_all)}개 지표를 함께 고려합니다: {target_labels}")

            valid_df = st.session_state.df_data[st.session_state.df_data["학습_적용"] == True]

            with st.container(border=True):
                colored_header(label="🤖 파레토 최적 후보 (MOBO)", description="qNEHVI 알고리즘으로 여러 목표를 동시에 개선할 다음 실험 후보를 제안합니다.", color_name="orange-70")
                if not _BOTORCH_AVAILABLE:
                    st.error("다중 목표 최적화에는 `botorch` 패키지가 필요합니다. `pip install botorch`로 설치한 뒤 앱을 다시 시작하세요.")
                elif st.button("🚀 AI 계산 실행", type="primary", use_container_width=True):
                    if len(valid_df) < 2:
                        st.warning("정밀 분석을 위해 최소 2개 이상의 유효 데이터가 필요합니다.")
                    else:
                        with st.spinner("다중목표 알고리즘 연산 중..."):
                            X_train, Y_train = process_robust_data_multi(valid_df, f_names, target_names_all)
                            directions = [tv["Direction"] for tv in st.session_state.target_vars]
                            mobo_error = None
                            try:
                                candidates, predicted_Y = run_mobo(
                                    X_train, Y_train, st.session_state.config_vars, directions, n_candidates=3
                                )
                            except Exception as e:
                                candidates, predicted_Y = None, None
                                mobo_error = str(e)

                        if mobo_error:
                            st.error(f"다중목표 계산 중 오류가 발생했습니다: {mobo_error}")
                        else:
                            if st.session_state.get("prev_mobo_points") == candidates:
                                st.info("💡 **AI 수렴 상태 판단:** 현재 데이터 풀 안에서 파레토 최적 후보가 안정적으로 수렴했습니다.")
                            st.session_state.prev_mobo_points = candidates

                            for i, (point, pred) in enumerate(zip(candidates, predicted_Y)):
                                with st.container(border=True):
                                    st.markdown(f"<h5 style='margin:0; font-weight: 800; color: #ed542b;'>실험 후보 {i+1}</h5>", unsafe_allow_html=True)
                                    st.divider()
                                    cols_rec = st.columns(len(f_names))
                                    for idx, (var, val) in enumerate(zip(st.session_state.config_vars, point)):
                                        unit_str = f" {var['Unit']}" if var.get("Unit") else ""
                                        disp_val = f"{round(val, 3)}{unit_str}" if isinstance(val, float) else f"{val}{unit_str}"
                                        cols_rec[idx].metric(label=var["Name"], value=disp_val)
                                    style_metric_cards(background_color="transparent", border_left_color="#ed542b", border_color="transparent", box_shadow=False)
                                    pred_str = " · ".join(
                                        f"{tv['Name']} ≈ {p:.6f}{' ' + tv['Unit'] if tv.get('Unit') else ''}"
                                        for tv, p in zip(st.session_state.target_vars, pred)
                                    )
                                    st.caption(f"예측 목표값: {pred_str}")
        # ---- 공정 변수별 목표 지표 분포 (Origin 스타일) ----
        st.divider()
        with st.container(border=True):
            colored_header(label="📊 공정 변수별 목표 지표 분포 (Origin 스타일)",
                           description="공정 변수 1개당 그래프 1개. X축은 공정 변수 값, Y축은 목표 지표를 0~1로 정규화해 겹쳐 보여줍니다. (학습 적용 데이터만)",
                           color_name="blue-70")
            valid_df_charts = st.session_state.df_data[st.session_state.df_data["학습_적용"] == True]
            render_variable_charts(valid_df_charts, st.session_state.config_vars, st.session_state.target_vars, key_prefix="vardist")