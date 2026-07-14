import streamlit as st
import pandas as pd
import numpy as np
import io
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

# 1. 페이지 이름 변경 반영
st.set_page_config(page_title="NBEDL Exp Assistant", layout="wide")

# ==========================================
# 시스템 상태 초기화 (기본값들을 빈칸으로 변경하여 Placeholder 적용)
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
# 전처리 및 엑셀 로드 함수
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
    
    # 설정 복원
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
        
    # 변수 복원 및 Old_Name 보장
    df_vars = pd.read_excel(xls, 'Config_Vars')
    config_list = df_vars.to_dict('records')
    for var in config_list:
        if "Old_Name" not in var or pd.isna(var["Old_Name"]):
            var["Old_Name"] = var.get("Name", "")
    st.session_state.config_vars = config_list
    
    st.session_state.df_data = pd.read_excel(xls, 'Data')
    st.session_state.app_mode = "Dashboard"

# ==========================================
# [화면 A] 실험 세팅 모드 (Builder)
# ==========================================
if st.session_state.app_mode == "Setup":
    # 요청하신 타이틀로 변경
    st.title("⚙️ NBEDL AI 기반 공정 최적화 시스템")
    
    uploaded_file = st.file_uploader("📂 기존 실험 엑셀(xlsx) 파일 업로드", type=["xlsx"])
    if uploaded_file:
        load_excel_data(uploaded_file)
        st.rerun()
    
    st.divider()
    
    # 텍스트 입력 칸 Placeholder(회색 글자) 적용
    st.session_state.exp_name = st.text_input("📝 실험 프로젝트 이름 (엑셀 파일명으로 사용됩니다)", value=st.session_state.exp_name, placeholder="예: NBEDL_Experiment_01")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    target_name = col_t1.text_input("목표 지표 이름", value=st.session_state.target_info["name"], placeholder="예: J_sc")
    target_dir = col_t2.selectbox("최적화 방향", ["Maximize (최대화)", "Minimize (최소화)"])
    
    passive_val = ",".join(st.session_state.passive_vars) if st.session_state.passive_vars else ""
    passive_input = col_t3.text_input("환경 변수 (쉼표 구분)", value=passive_val, placeholder="예: 온도 (°C), 습도 (%)")
    
    # 요청하신 서브헤더 및 버튼 이름 적용
    st.subheader("🔬 최적화 대상 공정 변수 입력")
    if st.button("➕ 공정 변수 추가"):
        # Type 오류 수정 ("Real (실수)"로 명확히 지정)
        st.session_state.config_vars.append({
            "Old_Name": "", "Name": "", "Unit": "", "Type": "Real (실수)", 
            "Min": 0.0, "Max": 10.0, "Options": ""
        })
        st.rerun()
        
    for i, var in enumerate(st.session_state.config_vars):
        c1, c_u, c2, c3, c4 = st.columns([2, 1, 2, 2, 2])
        var["Name"] = c1.text_input(f"변수 {i+1} 이름", value=var.get("Name", ""), key=f"name_{i}", placeholder="예: 스핀코팅 속도1")
        var["Unit"] = c_u.text_input("단위", value=var.get("Unit", ""), key=f"unit_{i}", placeholder="예: rpm")
        
        # 안전한 index 찾기 로직
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
            st.error("목표 지표 이름을 입력해야 합니다. (예: J_sc)")
        elif not st.session_state.config_vars:
            st.error("최소 1개 이상의 공정 변수를 추가해야 합니다.")
        else:
            st.session_state.target_info = {"name": target_name, "direction": target_dir}
            p_vars = [v.strip() for v in passive_input.split(",") if v.strip()]
            st.session_state.passive_vars = p_vars
            
            # [중요] 이름이 바뀐 변수 찾아서 과거 데이터 칼럼명 덮어쓰기
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
# [화면 B] 실험 진행 모드 (대시보드)
# ==========================================
elif st.session_state.app_mode == "Dashboard":
    t_name = st.session_state.target_info["name"]
    t_dir = st.session_state.target_info["direction"]
    f_names = [v["Name"] for v in st.session_state.config_vars]
    
    # 실험 이름이 비어있으면 기본값 적용
    display_exp_name = st.session_state.exp_name if st.session_state.exp_name.strip() else "NBEDL_Experiment"
    st.title(f"🧪 NBEDL Exp Assistant : {display_exp_name}")
    
    with st.sidebar:
        st.header("📂 실험 데이터 관리")
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
        
        st.download_button(label="📥 현재 상태 Excel 다운로드", data=output.getvalue(), file_name=file_name_export, type="primary")
        
        st.divider()
        if st.button("🛠️ 설정 화면으로 돌아가기\n(현재 데이터 유지)"):
            st.session_state.app_mode = "Setup"
            st.rerun()
            
        if st.button("⚠️ 모든 데이터 초기화"):
            st.session_state.clear()
            st.rerun()

    st.subheader("📝 신규 데이터 입력")
    with st.form("input_form"):
        cols = st.columns(len(st.session_state.passive_vars) + len(st.session_state.config_vars) + 1)
        new_row = {"학습_적용": True}
        idx = 0
        
        label_style = "<div style='min-height: 40px; display: flex; align-items: flex-end; font-size: 14px; padding-bottom: 5px; color: #555;'>{}</div>"
        
        for p_var in st.session_state.passive_vars:
            with cols[idx]:
                st.markdown(label_style.format(p_var), unsafe_allow_html=True)
                new_row[p_var] = st.text_input(p_var, value="", label_visibility="collapsed")
            idx += 1
            
        for var in st.session_state.config_vars:
            unit_str = f" ({var['Unit']})" if var.get("Unit") else ""
            disp_name = f"{var['Name']}{unit_str}"
            
            with cols[idx]:
                st.markdown(label_style.format(disp_name), unsafe_allow_html=True)
                if "Real" in var["Type"]:
                    new_row[var["Name"]] = st.number_input(disp_name, value=float(var["Min"]), step=0.1, label_visibility="collapsed")
                elif "Integer" in var["Type"]:
                    new_row[var["Name"]] = st.number_input(disp_name, value=int(var["Min"]), step=1, label_visibility="collapsed")
                elif "Categorical" in var["Type"]:
                    opts = [o.strip() for o in var["Options"].split(",")]
                    new_row[var["Name"]] = st.selectbox(disp_name, opts, label_visibility="collapsed")
            idx += 1
            
        with cols[idx]:
            st.markdown(label_style.format(f"결과값 ({t_name})"), unsafe_allow_html=True)
            new_row[t_name] = st.number_input(t_name, value=0.0, label_visibility="collapsed")
        
        if st.form_submit_button("데이터 추가"):
            st.session_state.df_data = pd.concat([st.session_state.df_data, pd.DataFrame([new_row])], ignore_index=True)
            st.success("데이터가 저장되었습니다.")
            st.rerun()

    if st.button("↩️ 마지막 입력 취소 (가장 최근 데이터 삭제)"):
        if not st.session_state.df_data.empty:
            st.session_state.df_data = st.session_state.df_data.iloc[:-1] 
            st.success("가장 최근 데이터가 삭제되었습니다.")
            st.rerun()
        else:
            st.warning("삭제할 데이터가 없습니다.")

    st.divider()
    st.subheader("🗂️ 원본 데이터 수정 및 필터링")
    st.caption("표 안의 셀을 더블클릭해 직접 수정하거나, 표 좌측 빈칸을 눌러 전체 행(Row)을 선택한 후 키보드 Delete 키로 삭제할 수 있습니다.")
    
    st.session_state.df_data = st.data_editor(
        st.session_state.df_data, 
        use_container_width=True, 
        hide_index=False, 
        num_rows="dynamic", 
        column_config={"학습_적용": st.column_config.CheckboxColumn("학습 적용")}
    )

    st.divider()
    c1, c2 = st.columns(2)
    valid_df = st.session_state.df_data[st.session_state.df_data["학습_적용"] == True]

    with c1:
        st.markdown(f"**📈 최적화 진행도 ({t_dir})**")
        if len(valid_df) > 0:
            chart_data = valid_df[t_name].expanding().max() if "Maximize" in t_dir else valid_df[t_name].expanding().min()
            st.line_chart(chart_data, height=250)
        else:
            st.info("데이터가 부족합니다.")

    with c2:
        st.markdown("**🤖 AI 다음 조건 추천**")
        if st.button("계산 실행", type="primary"):
            if len(valid_df) < 2:
                st.warning("최소 2개 이상의 유효 데이터가 필요합니다.")
            else:
                with st.spinner("AI 최적화 가동 중..."):
                    X_train, y_train = process_robust_data(valid_df, f_names, t_name)
                    ai_spaces = []
                    for var in st.session_state.config_vars:
                        if "Real" in var["Type"]: ai_spaces.append(Real(var["Min"], var["Max"], name=var["Name"]))
                        elif "Integer" in var["Type"]: ai_spaces.append(Integer(var["Min"], var["Max"], name=var["Name"]))
                        elif "Categorical" in var["Type"]: ai_spaces.append(Categorical([o.strip() for o in var["Options"].split(",")], name=var["Name"]))
                    
                    y_train_fit = [-val for val in y_train] if "Maximize" in t_dir else y_train
                        
                    opt = Optimizer(dimensions=ai_spaces, base_estimator="GP", acq_func="EI", random_state=42)
                    opt.tell(X_train, y_train_fit)
                    next_x = opt.ask()
                    
                st.success("✅ 다음 최적 스플릿 조건")
                for var, val in zip(st.session_state.config_vars, next_x):
                    unit_str = f" {var['Unit']}" if var.get("Unit") else ""
                    st.metric(label=var["Name"], value=f"{val}{unit_str}")