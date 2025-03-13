import importlib.metadata
import streamlit  # Streamlit을 강제 로드

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
import zipfile
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정
import matplotlib as mpl
if platform.system() == 'Windows':
    # 윈도우의 경우 폰트 경로 직접 지정
    font_path = r'C:\Windows\Fonts\malgun.ttf'
    font_name = fm.FontProperties(fname=font_path, size=10).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

# 그래프 스타일 설정
plt.style.use('seaborn-v0_8')
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--', alpha=0.7)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)
plt.rc('axes', titlesize=14)
plt.rc('legend', fontsize=10)

def run_analysis(uploaded_file, thresholds_info, pval_settings):
    # 업로드된 엑셀 파일 읽기 (Sheet1, header가 첫 행)
    try:
        data = pd.read_excel(uploaded_file, sheet_name='Sheet1', header=0)
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None, None, None

    # 7~18열(0-index: 6~17) 데이터 추출
    S_data = data.iloc[:, 6:18].values  # shape: (row, 12)
    
    # 변수명 (고정)
    VAR_Name = [
        'BIS자기자본비율',     # col7
        'BIS비율변동성',      # col8
        '연체대출채권비율',    # col9
        '총자산순이익률',     # col10 (예: max(S)-S 변환 적용)
        'ROA변동성',         # col11
        'LCR',              # col12
        '요주의 대비 대손충당금 비율', # col13
        '가계대출 분할상환 비중',   # col14
        '후순위채권 비중',    # col15
        '뱅크런취약예금 비중',  # col16
        '자금조달편중도',     # col17
        'NSFR'              # col18
    ]
    
    # 결과 저장용 구조체 초기화
    final_results = {
        'order': list(range(1, 13)),
        '하단 %': [th[0] for th in thresholds_info],
        '상단 %': [th[1] for th in thresholds_info],
        '최소치': [],
        '최대치': [],
        '분포': [],
        '모수1': [],
        '모수2': [],
        '모수3': []
    }
    results_detail = []  # 각 변수별 상세 결과 저장
    graph_buffers = {}   # 각 변수별 그래프를 BytesIO에 저장 (키: 변수명)
    
    # 각 변수별로 분석 및 분포 피팅 수행
    for i in range(12):
        var_name = VAR_Name[i]
        lower, upper, remove_nonpos, transform_mode = thresholds_info[i]
    
        # 1) 해당 열 데이터 추출, NaN 제거 및 정렬
        col = S_data[:, i]
        col = np.array(col, dtype=float)
        col = col[~np.isnan(col)]
        S_sorted = np.sort(col)
        n = len(S_sorted)
        if n == 0:
            st.warning(f"[{var_name}] 유효 데이터가 없습니다.")
            final_results['최소치'].append(None)
            final_results['최대치'].append(None)
            final_results['분포'].append('')
            final_results['모수1'].append('')
            final_results['모수2'].append('')
            final_results['모수3'].append('')
            continue
    
        # 2) 백분위 기반 임계치 슬라이싱 (Matlab과 동일: S(Sp1+1:Sp2))
        Sp1 = int(np.ceil(n * lower))
        Sp2 = int(np.ceil(n * upper))
        S_filtered = S_sorted[Sp1+1 : Sp2]
    
        # 3) 0 이하 값 제거 (필요한 경우)
        if remove_nonpos:
            S_filtered = S_filtered[S_filtered > 0]
    
        if len(S_filtered) == 0:
            st.warning(f"[{var_name}] 임계치 적용 후 데이터가 없습니다.")
            final_results['최소치'].append(round(np.min(S_sorted), 4))
            final_results['최대치'].append(round(np.max(S_sorted), 4))
            final_results['분포'].append('')
            final_results['모수1'].append('')
            final_results['모수2'].append('')
            final_results['모수3'].append('')
            continue
    
        # 4) 데이터 변환 (maxS 또는 minS 적용)
        if transform_mode == 'maxS':
            base_max = S_filtered[-1]
            S_filtered = base_max - S_filtered
            S_filtered = np.sort(S_filtered)
        elif transform_mode == 'minS':
            base_min = S_filtered[0]
            S_filtered = S_filtered - base_min
            S_filtered = np.sort(S_filtered)
    
        # 5) 경험적 CDF 계산 및 여러 분포 피팅
        n_f = len(S_filtered)
        empirical_cdf = np.arange(1, n_f + 1) / n_f
    
        candidate_params = {}
        candidate_errors = {}
        candidate_T = {}
    
        # 예시: Burr 분포 피팅
        try:
            p = stats.burr12.fit(S_filtered, floc=0)
            burr_c, burr_d, _, burr_scale = p
            param_list = [burr_scale, burr_c, burr_d]
            cdf_vals = stats.burr12.cdf(S_filtered, burr_c, burr_d, loc=0, scale=burr_scale)
            abs_diff = np.abs(cdf_vals - empirical_cdf)
            mae = np.mean(abs_diff)
            mse = np.mean((cdf_vals - empirical_cdf) ** 2)
            ks_stat, pval = stats.kstest(S_filtered, 'burr12', args=(burr_c, burr_d, 0, burr_scale))
            h = 1 if pval < pval_settings["Burr"] else 0
            candidate_params['Burr'] = param_list
            candidate_errors['Burr'] = [mae, np.max(abs_diff), mse, h]
            candidate_T['Burr'] = (S_filtered, empirical_cdf, cdf_vals)
        except Exception as e:
            pass
        # Weibull
        try:
            p = stats.weibull_min.fit(S_filtered, floc=0)
            shape, _, scale = p
            param_list = [scale, shape, np.nan]
            cdf_vals = stats.weibull_min.cdf(S_filtered, shape, loc=0, scale=scale)
            abs_diff = np.abs(cdf_vals - empirical_cdf)
            mae = np.mean(abs_diff)
            mse = np.mean((cdf_vals - empirical_cdf) ** 2)
            ks_stat, pval = stats.kstest(S_filtered, 'weibull_min', args=(shape, 0, scale))
            h = 1 if pval < pval_settings["Weibull"] else 0
            candidate_params['Weibull'] = param_list
            candidate_errors['Weibull'] = [mae, np.max(abs_diff), mse, h]
            candidate_T['Weibull'] = (S_filtered, empirical_cdf, cdf_vals)
        except Exception as e:
            pass
        # Gamma
        try:
            p = stats.gamma.fit(S_filtered, floc=0)
            a, _, scale = p
            param_list = [a, scale, np.nan]
            cdf_vals = stats.gamma.cdf(S_filtered, a, loc=0, scale=scale)
            abs_diff = np.abs(cdf_vals - empirical_cdf)
            mae = np.mean(abs_diff)
            mse = np.mean((cdf_vals - empirical_cdf) ** 2)
            ks_stat, pval = stats.kstest(S_filtered, 'gamma', args=(a, 0, scale))
            h = 1 if pval < pval_settings["Gamma"] else 0
            candidate_params['Gamma'] = param_list
            candidate_errors['Gamma'] = [mae, np.max(abs_diff), mse, h]
            candidate_T['Gamma'] = (S_filtered, empirical_cdf, cdf_vals)
        except Exception as e:
            pass
        # Lognormal
        try:
            p = stats.lognorm.fit(S_filtered, floc=0)
            sigma, _, scale = p
            mu = np.log(scale)
            param_list = [mu, sigma, np.nan]
            cdf_vals = stats.lognorm.cdf(S_filtered, sigma, loc=0, scale=scale)
            abs_diff = np.abs(cdf_vals - empirical_cdf)
            mae = np.mean(abs_diff)
            mse = np.mean((cdf_vals - empirical_cdf) ** 2)
            ks_stat, pval = stats.kstest(S_filtered, 'lognorm', args=(sigma, 0, scale))
            h = 1 if pval < pval_settings["Lognormal"] else 0
            candidate_params['Lognormal'] = param_list
            candidate_errors['Lognormal'] = [mae, np.max(abs_diff), mse, h]
            candidate_T['Lognormal'] = (S_filtered, empirical_cdf, cdf_vals)
        except Exception as e:
            pass
        # Normal
        try:
            mu_norm, std_norm = stats.norm.fit(S_filtered)
            param_list = [mu_norm, std_norm, np.nan]
            cdf_vals = stats.norm.cdf(S_filtered, loc=mu_norm, scale=std_norm)
            abs_diff = np.abs(cdf_vals - empirical_cdf)
            mae = np.mean(abs_diff)
            mse = np.mean((cdf_vals - empirical_cdf) ** 2)
            ks_stat, pval = stats.kstest(S_filtered, 'norm', args=(mu_norm, std_norm))
            h = 1 if pval < pval_settings["Normal"] else 0
            candidate_params['Normal'] = param_list
            candidate_errors['Normal'] = [mae, np.max(abs_diff), mse, h]
            candidate_T['Normal'] = (S_filtered, empirical_cdf, cdf_vals)
        except Exception as e:
            pass
        # Exponential
        try:
            p = stats.expon.fit(S_filtered, floc=0)
            scale = p[1]
            param_list = [scale, np.nan, np.nan]
            cdf_vals = stats.expon.cdf(S_filtered, loc=0, scale=scale)
            abs_diff = np.abs(cdf_vals - empirical_cdf)
            mae = np.mean(abs_diff)
            mse = np.mean((cdf_vals - empirical_cdf) ** 2)
            ks_stat, pval = stats.kstest(S_filtered, 'expon', args=(0, scale))
            h = 1 if pval < pval_settings["Exponential"] else 0
            candidate_params['Exponential'] = param_list
            candidate_errors['Exponential'] = [mae, np.max(abs_diff), mse, h]
            candidate_T['Exponential'] = (S_filtered, empirical_cdf, cdf_vals)
        except Exception as e:
            pass
        # Generalized Pareto
        try:
            p = stats.genpareto.fit(S_filtered)
            gp_k, gp_loc, gp_scale = p
            param_list = [gp_k, gp_scale, gp_loc]
            cdf_vals = stats.genpareto.cdf(S_filtered, gp_k, loc=gp_loc, scale=gp_scale)
            abs_diff = np.abs(cdf_vals - empirical_cdf)
            mae = np.mean(abs_diff)
            mse = np.mean((cdf_vals - empirical_cdf) ** 2)
            ks_stat, pval = stats.kstest(S_filtered, 'genpareto', args=(gp_k, gp_loc, gp_scale))
            h = 1 if pval < pval_settings["GenPareto"] else 0
            candidate_params['GenPareto'] = param_list
            candidate_errors['GenPareto'] = [mae, np.max(abs_diff), mse, h]
            candidate_T['GenPareto'] = (S_filtered, empirical_cdf, cdf_vals)
        except Exception as e:
            pass
        # Half-Normal
        try:
            p = stats.halfnorm.fit(S_filtered, floc=0)
            hn_scale = p[1]
            param_list = [0, hn_scale, np.nan]
            cdf_vals = stats.halfnorm.cdf(S_filtered, loc=0, scale=hn_scale)
            abs_diff = np.abs(cdf_vals - empirical_cdf)
            mae = np.mean(abs_diff)
            mse = np.mean((cdf_vals - empirical_cdf) ** 2)
            ks_stat, pval = stats.kstest(S_filtered, 'halfnorm', args=(0, hn_scale))
            h = 1 if pval < pval_settings["HalfNormal"] else 0
            candidate_params['HalfNormal'] = param_list
            candidate_errors['HalfNormal'] = [mae, np.max(abs_diff), mse, h]
            candidate_T['HalfNormal'] = (S_filtered, empirical_cdf, cdf_vals)
        except Exception as e:
            pass
    
        # 6) 최적의 분포 선택 (평균 절대 오차 기준)
        best_dist = None
        best_mae = np.inf
        for dist_name, err_vals in candidate_errors.items():
            mae_val = err_vals[0]
            if mae_val < best_mae:
                best_mae = mae_val
                best_dist = dist_name
    
        if best_dist is None:
            st.warning(f"[{var_name}] 피팅 가능한 분포가 없습니다.")
            final_results['최소치'].append(round(np.min(S_sorted), 4))
            final_results['최대치'].append(round(np.max(S_sorted), 4))
            final_results['분포'].append('')
            final_results['모수1'].append('')
            final_results['모수2'].append('')
            final_results['모수3'].append('')
            continue
    
        picked_params = candidate_params[best_dist]
        picked_errors = candidate_errors[best_dist]
        picked_T = candidate_T[best_dist]
        # KS검정 결과: h=1이면 부적합, 0이면 적합 (여기서는 단순 정보로 활용)
        passed = 1 - picked_errors[3]
    
        min_val = round(np.min(S_sorted), 4)
        max_val = round(np.max(S_sorted), 4)
        final_results['최소치'].append(min_val)
        final_results['최대치'].append(max_val)
        final_results['분포'].append(best_dist)
        p1 = round(picked_params[0], 4) if not np.isnan(picked_params[0]) else ''
        p2 = round(picked_params[1], 4) if not np.isnan(picked_params[1]) else ''
        p3 = round(picked_params[2], 4) if len(picked_params) > 2 and not np.isnan(picked_params[2]) else ''
        final_results['모수1'].append(p1)
        final_results['모수2'].append(p2)
        final_results['모수3'].append(p3)
    
        results_detail.append({
            'Variable': var_name,
            'Best Distribution': best_dist,
            'Parameters': [p1, p2, p3],
            'Abs_Dev': best_mae,
            'Passed(KS)': passed
        })
    
        # 7) 그래프 생성 (경험적 CDF와 피팅된 분포 CDF 비교)
        fig = plt.figure(figsize=(10, 6), dpi=150)
        ax = fig.add_subplot(111)
        
        # 데이터 플롯
        ax.plot(picked_T[0], picked_T[1], 'k-', linewidth=2, label='Empirical CDF')
        ax.plot(picked_T[0], picked_T[2], color='#1E88E5', linestyle='--', linewidth=1.5, label=f'{best_dist} CDF')
        
        # 폰트 설정
        if platform.system() == 'Windows':
            font_path = r'C:\Windows\Fonts\malgun.ttf'
            font_prop = fm.FontProperties(fname=font_path, size=14)
        else:
            font_prop = fm.FontProperties(family='NanumGothic', size=14)
        
        # 타이틀 및 레이블 설정
        ax.set_title(f'{var_name}\n(Best Dist = {best_dist})', fontproperties=font_prop, pad=20)
        ax.set_xlabel('Value', labelpad=10)
        ax.set_ylabel('Cumulative Probability', labelpad=10)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 그래프 여백 조정
        plt.tight_layout()
        
        # BytesIO에 이미지 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graph_buffers[var_name] = buf
        plt.close()
    
    # 8) 최종 결과 DataFrame 생성 및 Excel 파일 생성 (메모리 내 BytesIO로 저장)
    df = pd.DataFrame(final_results)
    df_transposed = df.transpose()
    df_transposed.columns = VAR_Name
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df_transposed.to_excel(writer, index=True)
    excel_buffer.seek(0)
    
    return df_transposed, excel_buffer, results_detail, graph_buffers

def main():
    # 페이지 설정 및 CSS 스타일 적용
    st.set_page_config(
        page_title="데이터 분포 분석 시스템",
        page_icon="📊",
        layout="centered"
    )

    # 사이드바 옵션: Threshold Settings
    with st.sidebar.expander("Threshold Settings", expanded=False):
        default_thresholds_info = [
            (0.0, 0.9,  False, None),
            (0.0, 0.95, False, None),
            (0.0, 0.9,  False, None),
            (0.05, 0.95, False, 'maxS'),
            (0.0, 0.9,  False, None),
            (0.0, 0.9,  False, None),
            (0.0, 0.9,  False, None),
            (0.0, 1.0,  False, None),
            (0.0, 1.0,  True,  None),
            (0.05, 0.95, False, None),
            (0.0, 0.95, True,  None),
            (0.0, 0.9,  False, None)
        ]
        new_thresholds_info = []
        var_list = [
            'BIS자기자본비율', 'BIS비율변동성', '연체대출채권비율', '총자산순이익률',
            'ROA변동성', 'LCR', '요주의 대비 대손충당금 비율', '가계대출 분할상환 비중',
            '후순위채권 비중', '뱅크런취약예금 비중', '자금조달편중도', 'NSFR'
        ]
        for i, var in enumerate(var_list):
            st.markdown(f"**{var}**")
            lower_val = st.number_input(f"{var} - 하단 백분위 (%)", value=default_thresholds_info[i][0]*100, min_value=0.0, max_value=100.0, step=0.5, key=f"lower_{i}")
            upper_val = st.number_input(f"{var} - 상단 백분위 (%)", value=default_thresholds_info[i][1]*100, min_value=0.0, max_value=100.0, step=0.5, key=f"upper_{i}")
            remove_option = st.selectbox(f"{var} - 0 이하 제거", options=[False, True], index=1 if default_thresholds_info[i][2] else 0, key=f"remove_{i}")
            transform_option = st.selectbox(f"{var} - 변환 옵션", options=["None", "maxS", "minS"], 
                                              index=0 if default_thresholds_info[i][3] is None else (1 if default_thresholds_info[i][3]=="maxS" else 2), key=f"transform_{i}")
            transform_option_final = None if transform_option == "None" else transform_option
            new_thresholds_info.append((lower_val/100, upper_val/100, remove_option, transform_option_final))
    
    # 사이드바 옵션: P-Value Settings
    with st.sidebar.expander("P-Value Settings", expanded=False):
        burr_p = st.number_input("Burr 분포 p-value 임계치 (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, key="p_burr")
        weibull_p = st.number_input("Weibull 분포 p-value 임계치 (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, key="p_weibull")
        gamma_p = st.number_input("Gamma 분포 p-value 임계치 (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, key="p_gamma")
        lognorm_p = st.number_input("Lognormal 분포 p-value 임계치 (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, key="p_lognorm")
        normal_p = st.number_input("Normal 분포 p-value 임계치 (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, key="p_normal")
        exponential_p = st.number_input("Exponential 분포 p-value 임계치 (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, key="p_exponential")
        genpareto_p = st.number_input("GenPareto 분포 p-value 임계치 (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, key="p_genpareto")
        halfnorm_p = st.number_input("HalfNormal 분포 p-value 임계치 (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, key="p_halfnorm")
        pval_settings = {
            "Burr": burr_p / 100,
            "Weibull": weibull_p / 100,
            "Gamma": gamma_p / 100,
            "Lognormal": lognorm_p / 100,
            "Normal": normal_p / 100,
            "Exponential": exponential_p / 100,
            "GenPareto": genpareto_p / 100,
            "HalfNormal": halfnorm_p / 100,
        }
    
    # 메인 CSS 스타일 정의
    st.markdown("""
        <style>
        /* 전체 폰트 및 기본 스타일 */
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
        * {
            font-family: 'Pretendard', sans-serif;
        }
        
        /* 메인 헤더 */
        .main-title {
            font-weight: 700;
            font-size: 2.5rem;
            color: #191F28;
            margin-bottom: 0.5rem;
            line-height: 1.35;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #8B95A1;
            margin-bottom: 2rem;
            line-height: 1.5;
        }
        
        /* 섹션 헤더 */
        .section-header {
            font-weight: 600;
            font-size: 1.25rem;
            color: #191F28;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
        }
        
        /* 버튼 스타일 */
        .stButton>button {
            background: #333D4B;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.7rem 1.5rem;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.2s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background: #4E5968;
            transform: translateY(-1px);
        }
        
        /* 다운로드 섹션 */
        .download-section {
            background: #F8F9FA;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
        
        /* 결과 카드 */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        
        /* 데이터프레임 스타일링 */
        .dataframe {
            font-size: 0.9rem;
            border: none !important;
        }
        .dataframe th {
            background: #F8F9FA !important;
            font-weight: 600 !important;
            padding: 0.75rem !important;
        }
        .dataframe td {
            padding: 0.75rem !important;
        }
        
        /* 확장 패널 스타일 */
        .streamlit-expanderHeader {
            background: white !important;
            border-radius: 8px !important;
            border: 1px solid #E9ECEF !important;
        }
        .streamlit-expanderContent {
            border: none !important;
            background: white !important;
        }
        
        /* 상세 결과 스타일 */
        .detail-item {
            padding: 1rem;
            border-bottom: 1px solid #F1F3F5;
        }
        .detail-item:last-child {
            border-bottom: none;
        }
        .detail-title {
            font-weight: 600;
            color: #191F28;
            margin-bottom: 0.5rem;
        }
        .detail-content {
            color: #4E5968;
            font-size: 0.95rem;
        }
        
        /* 그래프 컨테이너 */
        .graph-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* 에러 메시지 */
        .stError {
            background: #FFF5F5;
            color: #E03131;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #FFE3E3;
        }
        </style>
    """, unsafe_allow_html=True)

    # 메인 헤더
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="main-title">은행 변수 데이터 분포 분석</h1>
            <p class="sub-title">
                CDF 매칭 및 KS검정 분석
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 파일 업로드 섹션
    st.markdown('<p class="section-header">📤 데이터 업로드</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("분석할 엑셀 파일을 선택해주세요. (시트 이름 sheet1의 데이터로 분석)", type=["xlsx", "xls"])
    output_filename = st.text_input("결과 파일명 설정", value="분석결과_분포_피팅.xlsx")
    
    if uploaded_file is not None:
        if st.button("분석 시작", key="run_analysis_btn"):
            with st.spinner("데이터를 분석하고 있습니다..."):
                result_df, excel_buffer, details, graph_buffers = run_analysis(uploaded_file, new_thresholds_info, pval_settings)
            
            if result_df is not None:
                # 분석 결과 출력
                st.markdown('<p class="section-header">📊 분석 결과</p>', unsafe_allow_html=True)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 상세 분석 결과
                st.markdown('<p class="section-header">📈 상세 분석</p>', unsafe_allow_html=True)
                with st.expander("상세 분포 피팅 결과", expanded=False):
                    for item in details:
                        st.markdown(f"""
                            <div class="detail-item">
                                <div class="detail-title">{item['Variable']}</div>
                                <div class="detail-content">
                                    <p>🎯 최적 분포: <strong>{item['Best Distribution']}</strong></p>
                                    <p>📊 파라미터: <strong>{item['Parameters']}</strong></p>
                                    <p>📉 절대 오차 (MAE): <strong>{item['Abs_Dev']:.4f}</strong></p>
                                    <p>✓ KS 검정: <strong>{'합격' if item['Passed(KS)'] == 1 else '불합격'}</strong></p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # 다운로드 섹션
                st.markdown('<div class="download-section">', unsafe_allow_html=True)
                st.markdown('<p class="section-header" style="margin-top: 0;">💾 결과 다운로드</p>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Excel 결과 다운로드",
                        data=excel_buffer,
                        file_name=output_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                with col2:
                    # ZIP 파일 생성
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for var, buf in graph_buffers.items():
                            buf.seek(0)
                            zipf.writestr(f"graph_{var}.png", buf.read())
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="전체 그래프 다운로드",
                        data=zip_buffer,
                        file_name="graphs.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 그래프 섹션
                st.markdown('<p class="section-header">📊 분석 그래프</p>', unsafe_allow_html=True)
                for var, buf in graph_buffers.items():
                    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                    buf.seek(0)
                    st.image(buf, caption=var, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.error("분석 중 오류가 발생했습니다. 파일 형식을 확인해 주세요.")

if __name__ == "__main__":
    main()
