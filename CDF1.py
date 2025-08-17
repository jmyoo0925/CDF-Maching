import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import (
    weibull_min, gamma, lognorm,
    norm, expon, genpareto
)
from scipy.optimize import minimize
import matplotlib.font_manager as fm
import platform, warnings
import os
import urllib.request

warnings.filterwarnings("ignore")

# ────────── 한글 폰트 설정 ──────────
@st.cache_resource
def install_korean_font():
    """
    깃허브/Streamlit Cloud 환경에서 한글 폰트가 깨지는 문제를 해결하기 위해
    앱 실행 시 나눔고딕 폰트를 다운로드하고 matplotlib의 폰트 캐시를 재설정합니다.
    @st.cache_resource 데코레이터 덕분에 이 함수는 세션당 한 번만 실행됩니다.
    """
    font_name = "NanumGothic"
    font_path = f"./{font_name}.ttf"

    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        try:
            with st.spinner("한글 폰트를 설정 중입니다... (최초 1회)"):
                urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            st.error(f"폰트 다운로드에 실패했습니다: {e}")
            return

    try:
        # Matplotlib의 폰트 매니저에 폰트 추가
        fm.fontManager.addfont(font_path)
    except Exception as e:
        st.error(f"폰트 설정에 실패했습니다: {e}")
        return

# 폰트 설정 함수 실행
install_korean_font()

# Matplotlib의 전역 폰트 설정
plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)


# ────────── Streamlit 페이지 설정 ──────────
st.set_page_config(page_title="CDF 매칭 분석기", layout="wide")
st.title("📊 은행 변수 데이터 분포 분석")
st.markdown("---")


# ────────── 분포 관련 공통 함수 ──────────
def burr_cdf(x, alpha, c, k):
    # x가 리스트나 배열이 아닐 경우를 대비하여 np.asarray 사용
    x = np.asarray(x)
    return 1 - (1 + (x/alpha)**c)**(-k)

def fit_burr_mle(data):
    init_list = [
        [1.0, 1.0, 1.0], [0.5, 2.0, 0.5], [2.0, 0.5, 2.0],
        [5.0, 1.0, 1.0], [0.1, 5.0, 0.5],
        [np.mean(data), 1.0, 1.0], [np.median(data), 2.0, 0.5]
    ]
    best, best_nll = None, np.inf
    def nll(params):
        a, c, k = params
        if a <= 0 or c <= 0 or k <= 0: return np.inf
        # 로그(0) 방지를 위해 매우 작은 값(epsilon)을 더합니다.
        log_data = np.log(np.maximum(data, 1e-9) / a)
        term1 = np.log(c*k/a)
        term2 = (c-1) * log_data
        term3 = -(k+1) * np.log(1 + np.exp(c * log_data))
        return -np.sum(term1 + term2 + term3)
    for ip in init_list:
        for m in ["Nelder-Mead", "Powell", "L-BFGS-B"]:
            bnds = [(1e-6, None)]*3 if m == "L-BFGS-B" else None
            try:
                res = minimize(nll, ip, method=m, bounds=bnds, options={'maxiter':1000})
                if res.success and res.fun < best_nll:
                    best, best_nll = res, res.fun
            except: pass
    if best is None:
        raise RuntimeError("Burr MLE 실패")
    return best.x  # alpha, c, k

def halfnormal_cdf_vec(x, mu, sigma):
    # loc 파라미터(mu)와 scale 파라미터(sigma)를 사용하는 halfnorm cdf
    return stats.halfnorm.cdf(x, loc=mu, scale=sigma)

# ────────── 데이터 처리 함수 ──────────
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
    return df

def construct_sample(numeric_df, indicator, lo, hi, exclude_zero, invert_sample):
    j_adj = indicator - 7
    S_full = numeric_df.iloc[:, j_adj].dropna().astype(float).values
    if exclude_zero:
        S_full = S_full[S_full > 0]
    
    if len(S_full) == 0:
        st.error("선택한 열에 유효한 데이터가 없습니다.")
        return None, None
        
    S_full = np.sort(S_full)
    p1 = int(np.ceil(len(S_full) * lo / 100))
    p2 = int(np.ceil(len(S_full) * hi / 100))
    S = S_full[p1:p2]

    if len(S) < 5:
        st.error(f"필터링 후 샘플 데이터가 너무 적습니다 ({len(S)}개). 범위를 조정해주세요.")
        return None, None

    # Invert 옵션에 따라 S와 emp_cdf를 조정
    if invert_sample:
        S = np.max(S) - S + 0.001
        S = np.sort(S) # 반전 후 다시 정렬
        
    emp_cdf = (np.arange(1, len(S) + 1) / len(S))
    
    # invert_sample 옵션은 데이터 자체를 변환하는 데 사용했으므로
    # emp_cdf는 항상 오름차순으로 유지합니다.
    # 그래프에서 1-CDF를 그릴지는 INV 리스트로 별도 판단합니다.

    return S, emp_cdf


# ────────── Streamlit UI ──────────
uploaded = st.file_uploader("엑셀 파일 업로드 (Sheet1)", type=["xlsx", "xls"])

if uploaded:
    df_raw = load_data(uploaded)
    st.subheader("📑 Sheet1 미리보기")
    st.dataframe(df_raw.head(), use_container_width=True)

    numeric = df_raw.iloc[:, 6:18].apply(pd.to_numeric, errors="coerce")

    VAR_LIST = [
        'BIS비율', 'BIS비율 변동성', '연체대출채권비율', 'ROA', 'ROA 변동성', 'LCR',
        '요주의 대비 대손충당금 비율', '가계대출 분할상환 비중', '후순위채권 비중',
        '뱅크런취약예금 비중', '자금조달편중도', 'NSFR'
    ]
    
    # 역방향(1-CDF)으로 그래프를 그릴 지표 인덱스 목록
    INV = [8, 9, 11, 16, 17]

    # --- 사이드바 ---
    st.sidebar.header("모드 선택")
    app_mode = st.sidebar.radio("", ["자동 분포 분석", "수동 분포 비교"], label_visibility="collapsed")
    
    # ----------------------------------------------------------------------
    # --------------------------- 자동 분포 분석 모드 ---------------------------
    # ----------------------------------------------------------------------
    if app_mode == "자동 분포 분석":
        st.sidebar.header("⚙️ 분석 옵션")
        indicator = st.sidebar.selectbox("🔎 지표 선택 (7~18열)", options=list(range(7,19)),
                                         format_func=lambda x: f"{x}: {VAR_LIST[x-7]}", key="auto_indicator")
        lo = st.sidebar.number_input("하단 백분위수(%)", 0, 100, 0, step=1, key="auto_lo")
        hi = st.sidebar.number_input("상단 백분위수(%)", 0, 100, 100, step=1, key="auto_hi")
        ks_alpha = st.sidebar.number_input("KS 통과 기준값(α, %)", 1, 100, 5, step=1, format="%d", key="auto_ks")/100
        exclude_zero = st.sidebar.checkbox("0 이하 값 제거", False, key="auto_ex0")
        invert_sample = st.sidebar.checkbox("샘플 데이터 반전 (Invert)", False, key="auto_invert")
        
        run_btn = st.sidebar.button("📈 분석 실행", use_container_width=True, key="auto_run")

        if run_btn:
            with st.spinner('데이터 처리 및 분포 피팅 중... 잠시만 기다려주세요.'):
                # 1. 샘플 구성
                S, emp_cdf = construct_sample(numeric, indicator, lo, hi, exclude_zero, invert_sample)
                
                if S is None:
                    st.stop()
                
                results = {}

                # 2. 분포 피팅
                # (Burr)
                try:
                    a,c,k = fit_burr_mle(S)
                    T = burr_cdf(S,a,c,k)
                    results["Burr"] = dict(params=[float(round(a,4)), float(round(c,4)), float(round(k,4))],
                                           mae=np.mean(abs(T-emp_cdf)),
                                           pval=stats.kstest(S, lambda x: burr_cdf(x,a,c,k))[1],
                                           cdf=T)
                except Exception as e:
                    st.warning(f"Burr 피팅 실패: {e}")

                # (Other distributions)
                dist_definitions = [
                    ("Weibull", weibull_min.fit, weibull_min.cdf, lambda p: [float(round(p[0],4)), float(round(p[2],4))]),
                    ("Gamma", gamma.fit, gamma.cdf, lambda p: [float(round(p[0],4)), float(round(p[2],4))]),
                    ("LogNormal", lambda data: lognorm.fit(data, floc=0), lognorm.cdf, lambda p: [float(round(p[0],4)), float(round(p[2],4))]),
                    ("Normal", norm.fit, norm.cdf, lambda p: [float(round(p[0],4)), float(round(p[1],4))]),
                    ("Exponential", expon.fit, expon.cdf, lambda p: [float(round(p[1],4))]),
                    ("Generalized Pareto", genpareto.fit, genpareto.cdf, lambda p: [float(round(p[0],4)), float(round(p[2],4))])
                ]
                for name, fit_func, cdf_func, param_fmt in dist_definitions:
                    try:
                        params = fit_func(S)
                        T = cdf_func(S, *params)
                        results[name] = dict(params=param_fmt(params),
                                             mae=np.mean(abs(T-emp_cdf)),
                                             pval=stats.kstest(S, cdf_func, args=params)[1],
                                             cdf=T)
                    except Exception as e:
                        st.warning(f"{name} 피팅 실패: {e}")

                # (Half-Normal)
                try:
                    # 코드1의 로직을 유지하기 위해 loc=0으로 고정하여 sigma만 추정
                    mu, sigma = stats.halfnorm.fit(S)
                    T = halfnormal_cdf_vec(S, mu, sigma)
                    results["Half-Normal"] = dict(params=[float(round(mu,4)), float(round(sigma,4))],
                                                  mae=np.mean(abs(T-emp_cdf)),
                                                  pval=stats.kstest(S, lambda x: halfnormal_cdf_vec(x, mu, sigma))[1],
                                                  cdf=T)
                except Exception as e:
                     st.warning(f"Half-Normal 피팅 실패: {e}")


                if not results:
                    st.error("⚠️ 모든 분포 피팅에 실패했습니다. 데이터나 옵션을 확인해주세요.")
                    st.stop()

                # 3. 결과 정리 및 출력
                best_name, best_info = max(results.items(), key=lambda x: x[1]["pval"])

                table = pd.DataFrame([
                    dict(
                        분포=k,
                        MAE=v["mae"],
                        KS_pvalue=round(v["pval"],4),
                        KS_Pass="O" if v["pval"] > ks_alpha else "X",
                        Parameters=v["params"],
                        Best="✅" if k == best_name else ""
                    ) for k, v in results.items()
                ]).sort_values(["Best", "KS_pvalue"], ascending=[False, False]).set_index("분포")
                
                st.subheader("📋 분포별 피팅 결과")
                st.dataframe(table, use_container_width=True)

                st.markdown(
                    f"### ⭐ 최적 분포: **{best_name}**\n"
                    f"- 파라미터: `{best_info['params']}`\n"
                    f"- Mean Abs Deviation: **{best_info['mae']:.6f}**\n"
                    f"- KS p-value: **{best_info['pval']:.4f}** "
                    f"({'<span style="color:green;">**통과**</span>' if best_info['pval'] > ks_alpha else '<span style="color:red;">**실패**</span>'})",
                    unsafe_allow_html=True
                )
                st.markdown("---")

                # 4. 그래프 시각화
                plot_survival = indicator in INV
                j_adj = indicator - 7
                
                if plot_survival:
                    st.subheader("📉 (1-CDF) 비교 그래프")
                    st.info(f"선택하신 지표 '{VAR_LIST[j_adj]}'({indicator}번)는 (1-CDF)로 표시됩니다.")
                    y_label = "Survival Probability (1-CDF)"
                else:
                    st.subheader("📉 누적 분포 함수(CDF) 비교 그래프")
                    y_label = "Cumulative Probability (CDF)"
                
                passed = {k:v for k,v in results.items() if v["pval"] > ks_alpha}
                
                if passed:
                    for name, v in passed.items():
                        fig, ax = plt.subplots(figsize=(8, 5))
                        plot_emp_cdf = 1 - emp_cdf if plot_survival else emp_cdf
                        plot_theory_cdf = 1 - v["cdf"] if plot_survival else v["cdf"]
                        
                        ax.plot(S, plot_emp_cdf, 'k-', linewidth=2, label="Empirical Data")
                        ax.plot(S, plot_theory_cdf, "g--", linewidth=2, label=f"{name} Fit")
                        
                        ax.set_title(f"{VAR_LIST[j_adj]} - {name} 분포 (KS 통과)", fontsize=14)
                        ax.set_xlabel("Sample Value")
                        ax.set_ylabel(y_label)
                        ax.legend()
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)
                else:
                    st.warning("KS 테스트를 통과한 분포가 없습니다. p-value가 가장 높은 분포를 표시합니다.")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    plot_emp_cdf = 1 - emp_cdf if plot_survival else emp_cdf
                    plot_theory_cdf = 1 - best_info["cdf"] if plot_survival else best_info["cdf"]
                    
                    ax.plot(S, plot_emp_cdf, 'k-', linewidth=2, label="Empirical Data")
                    ax.plot(S, plot_theory_cdf, "r--", linewidth=2, label=f"{best_name} Fit (KS 실패)")

                    ax.set_title(f"{VAR_LIST[j_adj]} - {best_name} 분포 (KS 실패)", fontsize=14)
                    ax.set_xlabel("Sample Value")
                    ax.set_ylabel(y_label)
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

                st.markdown("---")
                # 5. 구간별 MAE 비교
                st.subheader("📌 구간별 MAE 비교")
                n = len(S)
                idx_ranges = [
                    np.arange(0, n), np.arange(0, n//2), np.arange(n//2, n),
                    np.arange(0, int(n*0.3)), np.arange(int(n*0.3), int(n*0.7)), np.arange(int(n*0.7), n)
                ]
                range_names = ["전체", "하위 50%", "상위 50%", "하위 30%", "중위 40%", "상위 30%"]

                mae_table = pd.DataFrame([
                    dict(분포=k, **{
                        rn: round(np.mean(np.abs(v["cdf"][r] - emp_cdf[r])), 6)
                        for rn, r in zip(range_names, idx_ranges) if len(r) > 0
                    }) for k, v in results.items()
                ]).set_index("분포")
                st.dataframe(mae_table, use_container_width=True)

            st.success("✅ 분석 완료")

    # ----------------------------------------------------------------------
    # --------------------------- 수동 분포 비교 모드 ---------------------------
    # ----------------------------------------------------------------------
    elif app_mode == "수동 분포 비교":
        st.header("수동 분포 비교")
        st.info("데이터와 직접 입력한 분포 및 파라미터를 비교하는 그래프를 생성합니다.")
        
        st.sidebar.header("⚙️ 비교 옵션")
        indicator = st.sidebar.selectbox("🔎 지표 선택", options=list(range(7,19)), format_func=lambda x: f"{x}: {VAR_LIST[x-7]}", key="manual_indicator")
        lo = st.sidebar.number_input("하단 백분위수(%)", 0, 100, 0, step=1, key="manual_lo")
        hi = st.sidebar.number_input("상단 백분위수(%)", 0, 100, 100, step=1, key="manual_hi")
        exclude_zero = st.sidebar.checkbox("0 이하 값 제거", False, key="manual_ex0")
        invert_sample = st.sidebar.checkbox("샘플 데이터 반전 (Invert)", False, key="manual_invert")
        
        st.sidebar.markdown("---")
        
        dist_manual = st.sidebar.selectbox("분포 선택", ["Burr", "Weibull", "Gamma", "LogNormal", "Normal", "Exponential", "Generalized Pareto", "Half-Normal"])
        
        params_input = {}
        if dist_manual == "Burr":
            params_input['alpha'] = st.sidebar.number_input("alpha", value=1.0, format="%.4f")
            params_input['c'] = st.sidebar.number_input("c", value=1.0, format="%.4f")
            params_input['k'] = st.sidebar.number_input("k", value=1.0, format="%.4f")
        elif dist_manual == "Weibull":
            params_input['c'] = st.sidebar.number_input("c (shape)", value=1.0, format="%.4f")
            params_input['scale'] = st.sidebar.number_input("scale", value=1.0, format="%.4f")
        elif dist_manual == "Gamma":
            params_input['a'] = st.sidebar.number_input("a (shape)", value=1.0, format="%.4f")
            params_input['scale'] = st.sidebar.number_input("scale", value=1.0, format="%.4f")
        elif dist_manual == "LogNormal":
            params_input['s'] = st.sidebar.number_input("s (shape, sigma)", value=1.0, format="%.4f")
            params_input['scale'] = st.sidebar.number_input("scale (exp(mu))", value=1.0, format="%.4f")
        elif dist_manual == "Normal":
            params_input['loc'] = st.sidebar.number_input("loc (mean)", value=0.0, format="%.4f")
            params_input['scale'] = st.sidebar.number_input("scale (std)", value=1.0, format="%.4f")
        elif dist_manual == "Exponential":
            params_input['scale'] = st.sidebar.number_input("scale (1/lambda)", value=1.0, format="%.4f")
        elif dist_manual == "Generalized Pareto":
            params_input['c'] = st.sidebar.number_input("c (shape)", value=0.1, format="%.4f")
            params_input['loc'] = st.sidebar.number_input("loc (location)", value=0.0, format="%.4f")
            params_input['scale'] = st.sidebar.number_input("scale", value=1.0, format="%.4f")
        elif dist_manual == "Half-Normal":
            params_input['loc'] = st.sidebar.number_input("loc (mu)", value=0.0, format="%.4f")
            params_input['scale'] = st.sidebar.number_input("scale (sigma)", value=1.0, format="%.4f")

        plot_manual_btn = st.sidebar.button("📊 그래프 생성", use_container_width=True, key="manual_run")

        if plot_manual_btn:
            S, emp_cdf = construct_sample(numeric, indicator, lo, hi, exclude_zero, invert_sample)
            if S is not None:
                T = None
                try:
                    if dist_manual == "Burr": T = burr_cdf(S, **params_input)
                    elif dist_manual == "Weibull": T = weibull_min.cdf(S, **params_input)
                    elif dist_manual == "Gamma": T = gamma.cdf(S, **params_input)
                    elif dist_manual == "LogNormal": T = lognorm.cdf(S, **params_input)
                    elif dist_manual == "Normal": T = norm.cdf(S, **params_input)
                    elif dist_manual == "Exponential": T = expon.cdf(S, loc=0, **params_input) # expon은 loc=0이 기본
                    elif dist_manual == "Generalized Pareto": T = genpareto.cdf(S, **params_input)
                    elif dist_manual == "Half-Normal": T = halfnormal_cdf_vec(S, mu=params_input['loc'], sigma=params_input['scale'])
                except Exception as e:
                    st.error(f"CDF 계산 중 오류 발생: {e}")
                    st.stop()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                plot_survival = indicator in INV
                j_adj = indicator - 7
                
                plot_emp = 1 - emp_cdf if plot_survival else emp_cdf
                plot_theory = 1 - T if plot_survival else T
                
                ax.plot(S, plot_emp, 'k-', linewidth=2, label="Empirical Data")
                ax.plot(S, plot_theory, "b--", linewidth=2, label=f"Manual {dist_manual} Fit")
                
                title_text = f"{VAR_LIST[j_adj]} - Manual {dist_manual} Plot"
                ax.set_title(title_text, fontsize=14)
                ax.set_ylabel("Probability")
                ax.set_xlabel("Sample Value")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
