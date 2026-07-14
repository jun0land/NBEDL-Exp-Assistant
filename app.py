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
# 1. 페이지 기본 설정 및 이탈 방지 (+ 자동완성 차단)
# ==========================================
st.set_page_config(page_title="NBEDL Exp Assistant", layout="wide")

components.html(
    """
    <script>
    window.onbeforeunload = function() {
        return "데이터가 저장되지 않았을 수 있습니다. 정말 나가시겠습니까?";
    };
    setInterval(function() {
        var inputs = document.querySelectorAll('input');
        inputs.forEach(function(input) {
            input.setAttribute('autocomplete', 'new-password');
            input.setAttribute('data-lpignore', 'true');
        });
    }, 1000);
    </script>
    """,
    height=0,
)

# ==========================================
# 2. 이미지 Base64 인코딩 & 글로벌 UI/UX CSS 주입
# ==========================================
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

bg_base64 = get_base64_of_bin_file('liquid_bg.png')
logo_base64 = get_base64_of_bin_file('logo.png')

# 로고와 텍스트의 완벽한 중앙 정렬
logo_html = f'<img src="data:image/png;base64,{logo_base64}" height="42" style="vertical-align: middle; margin-right: 12px; margin-bottom: 0px;">' if logo_base64 else ""

custom_css = f"""
<style>
/* 전역 글꼴 Pretendard 강제 적용 */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
* {{ font-family: 'Pretendard', sans-serif !important; }}

.stApp {{
    background: linear-gradient(135deg, rgba(255,255,255,0.40), rgba(247,239,232,0.28)), url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #241d1a;
}}

header[data-testid="stHeader"] {{ background: transparent !important; }}

/* 텍스트 배경색 대비 가독성 확보 (흰색 후광 렌더링) */
p, span, label, h1, h2, h3, h4, h5, h6, li {{
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.9), 0 0 6px rgba(255, 255, 255, 0.6);
}}
button span, .stButton button p {{ text-shadow: none !important; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.66), rgba(255, 255, 255, 0.46)) !important;
    backdrop-filter: blur(22px) saturate(145%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(145%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.56) !important;
}}

/* ✅ 타이틀 전용 배경: 테두리 없이 우측으로 갈수록 투명해지는 직사각형 페이드아웃 배경 */
.title-glass-container {{
    background: linear-gradient(to right, rgba(255,255,255,0.85) 0%, rgba(255,255,255,0.5) 40%, rgba(255,255,255,0) 100%);
    padding: 16px 24px 16px 20px;
    margin-bottom: 24px;
    margin-top: -10px;
    margin-left: -1rem;
    display: flex;
    align-items: center;
    border-left: 5px solid #ed542b; /* 포인트 라인 */
}}
.title-glass-container h2 {{ 
    margin: 0; color: #1a1a1a; font-weight: 800; padding: 0; line-height: 1.1; display: flex; align-items: center; 
}}

/* 메인 폼, 데이터 그리드 등 카드 컨테이너 (곡률 낮춤 20px -> 12px) */
[data-testid="stForm"], [data-testid="stExpander"] {{
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.86), rgba(252, 248, 244, 0.82)) !important;
    backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255,255,255,0.58) !important;
    box-shadow: 0 16px 40px rgba(17, 24, 39, 0.08) !important;
    border-radius: 12px !important;
    padding: 24px;
}}

/* 상단 탭(Tabs) 영역 가독성 강화 디자인 */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(12px);
    padding: 6px;
    border-radius: 12px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #5e5652 !important;
    padding: 10px 18px !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: rgba(255, 255, 255, 0.95) !important;
    color: #ed542b !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
}}

/* ✅ 입력칸과 글자(라벨)를 모두 감싸는 개별 유리 블럭 컨테이너 */
[data-testid="stForm"] [data-testid="column"] {{
    background: linear-gradient(135deg, rgba(255,255,255,0.6), rgba(255,255,255,0.3));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04), inset 0 2px 4px rgba(255,255,255,0.6);
    margin-bottom: 12px;
}}

/* ✅ 내부 입력칸(Input) - 배경을 반투명하게 하고 테두리를 단정하게 (곡률 낮춤) */
div[data-baseweb="input"], div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.7) !important; 
    border-radius: 6px !important; /* 각진 느낌 강화 */
    border: 1px solid rgba(255,255,255,0.9) !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
    transition: all 0.2s ease !important;
    overflow: hidden !important; 
}}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {{
    border-color: #ed542b !important;
    background: rgba(255,255,255,0.95) !important;
    box-shadow: inset 0 1px 3px rgba(237,84,43,0.1), 0 0 0 2px rgba(237,84,43,0.15) !important;
}}
div[data-baseweb="input"] > div {{ background: transparent !important; border: none !important; }}

/* 오렌지 버튼 */
.stButton > button {{
    border-radius: 8px !important; /* 각진 버튼 */
    border: 1px solid rgba(255,255,255,0.55) !important;
    background: rgba(255,255,255,0.4) !important;
    font-weight: 700 !important;
    color: #241d1a !important;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #ed542b, #f68b21) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(237,84,43,0.3) !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(17,24,39,0.12) !important;
    border-color: #ed542b !important;
    color: white !important;
    background: #ed542b !important;
}}

/* 차트 컨테이너 (그래프 튀어나감 완벽 방어) */
[data-testid="stVegaLiteChart"] {{
    background: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.8) !important;
    padding: 0 !important; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    overflow: hidden !important;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 3. 시스템 상태 초기화
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
# 4. 전처리 및 엑셀 로드 함수
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
    # 개선된 투명 그라데이션 타이틀 박스
    st.markdown(f"""
    <div class="title-glass-container">
        {logo_html}
        <h2>NBEDL AI 기반 공정 최적화 시스템</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📂 기존 실험 엑셀(xlsx) 파일 업로드", type=["xlsx"])
    if uploaded_file:
        load_excel_data(uploaded_file)
        st.rerun()
    
    st.divider()
    
    st.session_state.exp_name = st.text_input("📝 실험 프로젝트 이름 (엑셀 파일명으로 사용됩니다)", value=st.session_state.exp_name, placeholder="예: NBEDL_Experiment_01")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    target_name = col_t1.text_input("목표 지표 이름", value=st.session_state.target_info["name"], placeholder="예: J_sc")
    target_dir = col_t2.selectbox("최적화 방향", ["Maximize", "Minimize"])
    
    passive_val = ",".join(st.session_state.passive_vars) if st.session_state.passive_vars else ""
    passive_input = col_t3.text_input("환경 변수 (쉼표 구분)", value=passive_val, placeholder="예: 온도 (°C), 습도 (%)")
    
    colored_header(label="🔬 최적화 대상 공정 변수 입력", description="AI가 탐색할 공정 조건의 이름과 변수 범위를 지정하세요.", color_name="orange-70")
    
    if st.button("➕ 공정 변수 추가"):
        st.session_state.config_vars.append({
            "Old_Name": "", "Name": "", "Unit": "", "Type": "Real (실수)", 
            "Min": 0.0, "Max": 10.0, "Options": ""
        })
        st.rerun()
        
    for i, var in enumerate(st.session_state.config_vars):
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
            
    st.divider()
    if st.button("🚀 실험 시작 및 대시보드 생성", type="primary"):
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
    
    # 개선된 투명 그라데이션 타이틀 박스
    st.markdown(f"""
    <div class="title-glass-container">
        {logo_html}
        <h2>NBEDL Exp Assistant : {display_exp_name}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # --- 사이드바 ---
    with st.sidebar:
        st.header("📂 데이터 관리 패널")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.df_data.to_excel(writer, sheet_name='Data', index=False)
            pd.DataFrame(st.session_state.config_vars).to_excel(writer, sheet_name='Config_Vars', index=False)
            meta_data = {
                "Exp_Name": [display_exp_name],
                "Target_Name": [t_name], 
                "Direction": [t_dir], 
                "Passive_Vars": [",".join(st.session_state.passive_vars)]
            }
            pd.DataFrame(meta_data).to_excel(writer, sheet_name='Config_Meta', index=False)
            
        file_name_export = f"{display_exp_name}_Data.xlsx"
        
        st.download_button(label="📥 최신 데이터 Excel 다운로드", data=output.getvalue(), file_name=file_name_export, type="primary", use_container_width=True)
        st.divider()
        if st.button("🛠️ 환경 설정으로 돌아가기", use_container_width=True):
            st.session_state.app_mode = "Setup"
            st.rerun()
        if st.button("⚠️ 모든 데이터 초기화", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # --- 메인 화면 탭(Tabs) ---
    tab1, tab2, tab3 = st.tabs(["📝 신규 실험 입력", "🗂️ 데이터베이스 관리", "🤖 AI 최적화 대시보드"])

    # ----------------------------------------
    # [Tab 1] 신규 데이터 입력 공간
    # ----------------------------------------
    with tab1:
        colored_header(label="새로운 스플릿 실험 결과 입력", description="값을 모두 적은 후 하단 '데이터 추가' 버튼을 클릭하세요.", color_name="orange-70")
        
        with st.form("input_form", clear_on_submit=True):
            cols = st.columns(len(st.session_state.passive_vars) + len(st.session_state.config_vars) + 1)
            new_row = {"학습_적용": True}
            idx = 0
            
            # 라벨 텍스트 스타일: 마진을 줄여 래퍼 박스 안에서 깔끔하게 정렬되도록 수정
            label_style = "<div style='font-size: 14px; font-weight: 800; padding-bottom: 6px; color: #1a1a1a;'>{}</div>"
            
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

    # ----------------------------------------
    # [Tab 2] 데이터베이스 관리 (에디터)
    # ----------------------------------------
    with tab2:
        colored_header(label="전체 실험 데이터 아카이브", description="입력 이력을 한눈에 검토하고 이상치 데이터의 AI 반영 여부를 수정할 수 있습니다.", color_name="orange-70")
        st.session_state.df_data = st.data_editor(
            st.session_state.df_data, 
            use_container_width=True, 
            hide_index=True, 
            column_config={"학습_적용": st.column_config.CheckboxColumn("학습 적용")}
        )

    # ----------------------------------------
    # [Tab 3] AI 최적화 대시보드
    # ----------------------------------------
    with tab3:
        valid_df = st.session_state.df_data[st.session_state.df_data["학습_적용"] == True]
        c1, c2 = st.columns([1.2, 1])

        with c1:
            # ✅ 중복 텍스트 삭제 및 깔끔한 제목
            colored_header(label=f"📈 최적화 경향 곡선", description=f"실험이 진행됨에 따라 타겟 지표({t_name})의 수렴 상태를 보여줍니다.", color_name="green-70")
            if len(valid_df) > 0:
                chart_data = valid_df[t_name].expanding().max() if "Maximize" in t_dir else valid_df[t_name].expanding().min()
                st.line_chart(chart_data, height=350)
            else:
                st.info("분석용 데이터가 입력되지 않았습니다.")

        with c2:
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
                        with st.container():
                            st.markdown(f"**실험 후보 {i+1}**")
                            cols_rec = st.columns(len(f_names))
                            for idx, (var, val) in enumerate(zip(st.session_state.config_vars, points)):
                                unit_str = f" {var['Unit']}" if var.get("Unit") else ""
                                cols_rec[idx].metric(label=var["Name"], value=f"{round(val, 3)}{unit_str}")
                            style_metric_cards(background_color="rgba(255,255,255,0.4)", border_left_color="#ed542b", border_color="rgba(255,255,255,0.6)", box_shadow=False)
                    
                    with st.expander("🔍 AI 연산 피팅 로그 데이터"):
                        debug_df = pd.DataFrame(X_train, columns=f_names)
                        debug_df[t_name] = y_train
                        st.dataframe(debug_df, use_container_width=True)