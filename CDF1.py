import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import (
    burr12, weibull_min, gamma, lognorm,
    norm, expon, genpareto
)
from scipy.optimize import minimize
import matplotlib.font_manager as fm
import platform, warnings
warnings.filterwarnings("ignore")

# ────────── 한글 폰트 설정 ──────────
if platform.system() == "Windows":
    font_path = r"C:\Windows\Fonts\malgun.ttf"
    plt.rc("font", family=fm.FontProperties(fname=font_path).get_name())
else:
    plt.rc("font", family="NanumGothic")
plt.rc("axes", unicode_minus=False)

# ────────── Streamlit 페이지 설정 ──────────
st.set_page_config(page_title="CDF 매칭 분석기")
st.title("📊 은행 변수 데이터 분포 분석")

# ────────── Burr / Half‑Normal 전용 함수 ──────────
def burr_cdf(x, alpha, c, k):
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
        term1 = np.log(c*k/a)
        term2 = (c-1)*np.log(data/a)
        term3 = -(k+1)*np.log(1+(data/a)**c)
        return -np.sum(term1+term2+term3)
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

def fit_halfnormal_sigma(data):
    return np.sqrt(np.sum(data**2)/len(data))

def halfnormal_cdf_vec(x, sigma):
    return 2*norm.cdf(x/sigma) - 1

# ────────── Streamlit UI ──────────
uploaded = st.file_uploader("엑셀 파일 업로드 (Sheet1)", type=["xlsx", "xls"])

if uploaded:
    df_raw = pd.read_excel(uploaded, sheet_name="Sheet1")
    st.subheader("📑 Sheet1 미리보기")
    st.dataframe(df_raw, use_container_width=True)

    numeric = df_raw.iloc[:, 6:18].apply(pd.to_numeric, errors="coerce")
    A = numeric.to_numpy(float)

    VAR_LIST = [
        'BIS비율', 'BIS비율 변동성', '연체대출채권비율', 'ROA', 'ROA 변동성', 'LCR',
        '요주의 대비 대손충당금 비율', '가계대출 분할상환 비중', '후순위채권 비중',
        '뱅크런취약예금 비중', '자금조달편중도', 'NSFR'
    ]

    col1, col2 = st.columns(2)
    with col1:
        indicator = st.selectbox("🔎 지표 선택 (7~18열)", options=list(range(7,19)),
                                 format_func=lambda x: f"{x}: {VAR_LIST[x-7]}")
        lo = st.number_input("하단 백분위수(%)", 0, 100, 0, step=1)
        hi = st.number_input("상단 백분위수(%)", 0, 100, 100, step=1)
        ks_alpha = st.number_input("KS 통과 기준값(α, %)", 1, 100, 5, step=1, format="%d")/100
    with col2:
        exclude_zero = st.checkbox("0 이하 값 제거", False)
        invert_sample = st.checkbox("샘플 반전", False)
        run_btn = st.button("📈 분석 실행")

    if run_btn:
        j_adj = indicator - 7
        S_full = numeric.iloc[:, j_adj].dropna().astype(float).values
        if exclude_zero:
            S_full = S_full[S_full > 0]
        S_full = np.sort(S_full)
        p1 = int(np.ceil(len(S_full)*lo/100))
        p2 = int(np.ceil(len(S_full)*hi/100))
        S = S_full[p1:p2]
        if invert_sample:
            S = np.max(S) - S + 0.001
        S = np.sort(S)
        emp_cdf = np.arange(1, len(S)+1)/len(S)
        results = {}

        try:
            a,c,k = fit_burr_mle(S)
            T = burr_cdf(S,a,c,k)
            results["Burr"] = dict(params=[float(round(a,4)), float(round(c,4)), float(round(k,4))],
                                   mae=np.mean(abs(T-emp_cdf)),
                                   pval=stats.kstest(S, lambda x: burr_cdf(x,a,c,k))[1],
                                   cdf=T)
        except Exception as e:
            st.warning(f"Burr 피팅 실패: {e}")

        for name, fit_func, cdf_func, param_fmt in [
            ("Weibull", weibull_min.fit, weibull_min.cdf,
             lambda p: [float(round(p[0],4)), float(round(p[2],4))]),
            ("Gamma", gamma.fit, gamma.cdf,
             lambda p: [float(round(p[0],4)), float(round(p[2],4))]),
            ("LogNormal", lambda data: lognorm.fit(data, floc=0), lognorm.cdf,
             lambda p: [float(round(p[0],4)), float(round(p[2],4))]),
            ("Normal", norm.fit, norm.cdf,
             lambda p: [float(round(p[0],4)), float(round(p[1],4))]),
            ("Exponential", expon.fit, expon.cdf,
             lambda p: [float(round(p[1],4))]),
            ("Generalized Pareto", genpareto.fit, genpareto.cdf,
             lambda p: [float(round(p[0],4)), float(round(p[2],4))])
        ]:
            try:
                params = fit_func(S)
                if name == "LogNormal":
                    shape, loc, scale = params

                T = cdf_func(S, *params)
                results[name] = dict(params=param_fmt(params),
                                     mae=np.mean(abs(T-emp_cdf)),
                                     pval=stats.kstest(S, cdf_func, args=params)[1],
                                     cdf=T)
            except Exception as e:
                st.warning(f"{name} 피팅 실패: {e}")

        try:
            sigma = fit_halfnormal_sigma(S)
            T = halfnormal_cdf_vec(S, sigma)
            results["Half‑Normal"] = dict(params=[float(round(sigma,4))],
                                          mae=np.mean(abs(T-emp_cdf)),
                                          pval=stats.kstest(S, lambda x: halfnormal_cdf_vec(x,sigma))[1],
                                          cdf=T)
        except: pass

        if not results:
            st.error("⚠️ 모든 분포 피팅 실패")
            st.stop()

        best_name, best_info = max(results.items(), key=lambda x: x[1]["pval"])

        table = pd.DataFrame([{
            "분포": k,
            "MAE": v["mae"],
            "KS_pvalue": round(v["pval"],4),
            "KS_Pass": "O" if v["pval"] > ks_alpha else "X",
            "Parameters": v["params"],
            "Best": "✅" if k == best_name else ""
        } for k, v in results.items()]).sort_values(["Best", "KS_pvalue"], ascending=[False, False])

        def highlight_ks_pass(row):
            return ['font-weight: bold' if row["KS_Pass"] == "O" else '' for _ in row]

        st.subheader("📋 분포별 피팅 결과")
        st.dataframe(table.style.apply(highlight_ks_pass, axis=1), use_container_width=True)

        st.markdown(
            f"### ⭐ 최적 분포: **{best_name}**\n"
            f"- 파라미터: `{best_info['params']}`\n"
            f"- Mean Abs Deviation: **{best_info['mae']:.6f}**\n"
            f"- KS p-value: **{best_info['pval']:.4f}** "
            f"({'통과' if best_info['pval'] > ks_alpha else '실패'})"
        )

        st.subheader("📉 CDF 비교 그래프")
        passed = {k:v for k,v in results.items() if v["pval"] > ks_alpha}
        if passed:
            for name, v in passed.items():
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.scatter(S, emp_cdf, s=25, color="blue", alpha=0.6, label="Empirical CDF")
                ax.plot(S, v["cdf"], "--", color="green", linewidth=2, label=f"{name} CDF")
                ax.set_title(f"{name} (KS Test Pass)")
                ax.set_xlabel("Sample Value")
                ax.set_ylabel("Cumulative Probability")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
        else:
            v = best_info
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(S, emp_cdf, s=25, color="blue", alpha=0.6, label="Empirical CDF")
            ax.plot(S, v["cdf"], "--", color="red", linewidth=2, label=f"{best_name} CDF (KS 실패)")
            ax.set_title(f"{best_name} (KS Test Fail)")
            ax.set_xlabel("Sample Value")
            ax.set_ylabel("Cumulative Probability")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.subheader("📌 구간별 MAE 비교")

        n = len(S)
        idx_ranges = [
            np.arange(0, n),
            np.arange(0, n//2),
            np.arange(n//2, n),
            np.arange(0, int(n*0.3)),
            np.arange(int(n*0.3), int(n*0.7)),
            np.arange(int(n*0.7), n)
        ]
        range_names = ["전체", "하위 50%", "상위 50%", "하위 30%", "중위 40%", "상위 30%"]

        # 순서에 맞게 정렬된 분포 이름 리스트
        sorted_dist_names = table["분포"].tolist()

        # 각 분포별로 구간별 MAE + KS통과여부
        mae_rows = []
        for name in sorted_dist_names:
            v = results[name]
            row = {
                "분포": name,
                "KS_Pass": "O" if v["pval"] > ks_alpha else "X"
            }
            for rn, r in zip(range_names, idx_ranges):
                row[rn] = round(np.mean(np.abs(v["cdf"][r] - emp_cdf[r])), 6)
            mae_rows.append(row)

        mae_table = pd.DataFrame(mae_rows)
        st.dataframe(mae_table, use_container_width=True)


        st.success("✅ 분석 완료")
