import re
import difflib
from io import StringIO
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st



def _approx_erf(x: float) -> float:
    x = float(x)
    sgn = np.sign(x)
    x = abs(x)
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-x * x))
    return sgn * y


def _simplex_projection(v: np.ndarray, total: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if total <= 0:
        raise ValueError("Total mass for simplex projection must be positive.")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    j = np.arange(1, v.size + 1)
    cond = u * j > (cssv - total)
    if not np.any(cond):
        return np.zeros_like(v)
    rho = j[cond][-1] - 1
    theta = (cssv[rho] - total) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _apply_weight_cap(weights: np.ndarray, cap: float | None) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if cap is None or cap >= 1.0:
        return _simplex_projection(w, 1.0)
    return _simplex_projection(np.minimum(w, cap), 1.0)


def _lognormal_terminal_prob(
    w: np.ndarray,
    mu_daily: np.ndarray,
    cov_daily: np.ndarray,
    horizon_days: int,
    initial_value: float,
    target_value: float,
) -> float:
    if initial_value <= 0 or target_value <= 0:
        return 0.0
    w = np.clip(np.asarray(w, float), 0.0, None)
    s = w.sum()
    if s <= 0:
        return 0.0
    w /= s
    mu_p = float(w @ mu_daily)
    var_p = float(w @ cov_daily @ w)
    growth_ratio = target_value / float(initial_value)
    drift = mu_p - 0.5 * var_p
    if var_p <= 0:
        return 1.0 if drift * horizon_days >= np.log(growth_ratio) else 0.0
    z = (drift * horizon_days - np.log(growth_ratio)) / np.sqrt(var_p * horizon_days)
    return 0.5 * (1.0 + _approx_erf(z / np.sqrt(2.0)))



class MonteCarloSimulation:
    def __init__(
        self,
        tickers,
        exp_returns,
        volatilities,
        corr_matrix,
        weights=None,
        initial_investment: float = 1.0,
    ):
        self.tickers = [str(t).upper() for t in tickers]
        self.n_assets = len(self.tickers)

        self.exp_returns_annual = np.asarray(exp_returns, dtype=float)
        self.vol_annual = np.asarray(volatilities, dtype=float)
        self.corr = np.asarray(corr_matrix, dtype=float)

        if self.corr.shape != (self.n_assets, self.n_assets):
            raise ValueError("Correlation matrix dimension mismatch.")

        if weights is None:
            w = np.ones(self.n_assets) / self.n_assets
        else:
            w = np.asarray(weights, dtype=float)
            if w.sum() == 0:
                raise ValueError("Weights must not sum to zero.")
            w = w / w.sum()

        self.weights = w
        self.initial_value = float(initial_investment)

        self.mu_daily = self.exp_returns_annual / 252.0
        self.sigma_daily = self.vol_annual / np.sqrt(252.0)
        self.cov_daily = np.outer(self.sigma_daily, self.sigma_daily) * self.corr

        self.port_mu_daily = float(self.weights @ self.mu_daily)
        self.port_var_daily = float(self.weights @ self.cov_daily @ self.weights)
        self.port_sigma_daily = float(np.sqrt(self.port_var_daily))

    def summary(self) -> dict:
        return {
            "Assets": ", ".join(self.tickers),
            "Expected Annual Return": f"{self.port_mu_daily * 252:.2%}",
            "Expected Annual Volatility": f"{self.port_sigma_daily * np.sqrt(252):.2%}",
            "Initial Investment": f"${self.initial_value:,.2f}",
        }

    def run(
        self,
        n_paths: int = 10_000,
        horizon_days: int = 2520,
        use_drift_correction: bool = True,
        random_seed: int | None = None,
    ):
        T = int(horizon_days)
        S = int(n_paths)

        if random_seed is not None:
            rng = np.random.default_rng(int(random_seed))
        else:
            rng = np.random.default_rng()

        drift = (
            self.port_mu_daily - 0.5 * self.port_var_daily
            if use_drift_correction
            else self.port_mu_daily
        )

        eps = 1e-10
        cov_psd = self.cov_daily + eps * np.eye(self.n_assets)
        try:
            L = np.linalg.cholesky(cov_psd)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov_psd)
            eigvals[eigvals < 0] = eps
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        z = rng.normal(size=(T, self.n_assets, S))
        correlated = (L @ z.reshape(self.n_assets, -1)).reshape(T, self.n_assets, S)

        path_logs = np.tensordot(correlated, self.weights, axes=([1], [0])) + drift
        log_cum = np.cumsum(path_logs, axis=0)
        paths = self.initial_value * np.exp(log_cum)
        terminal_values = paths[-1, :]
        return paths, terminal_values


def search_weights_for_target(
    mu_daily: np.ndarray,
    cov_daily: np.ndarray,
    horizon_days: int,
    initial_value: float,
    target_value: float,
    n_assets: int,
    seed: int = 123,
    n_candidates: int = 2000,
    n_iters: int = 4000,
    local_step: float = 0.035,
    cap: float | None = None,
):
    rng = np.random.default_rng(seed)

    w_best = np.ones(n_assets) / n_assets
    p_best = _lognormal_terminal_prob(
        w_best, mu_daily, cov_daily, horizon_days, initial_value, target_value
    )

    for _ in range(n_candidates):
        w = rng.dirichlet(np.ones(n_assets))
        w = _apply_weight_cap(w, cap)
        p = _lognormal_terminal_prob(
            w, mu_daily, cov_daily, horizon_days, initial_value, target_value
        )
        if p > p_best:
            w_best, p_best = w, p

    w = w_best.copy()
    p0 = p_best

    for _ in range(n_iters):
        i, j = rng.integers(0, n_assets, size=2)
        if i == j:
            continue

        d = min(local_step, w[j])
        trial = w.copy()
        trial[i] += d
        trial[j] -= d
        trial = _apply_weight_cap(trial, cap)
        p_trial = _lognormal_terminal_prob(
            trial, mu_daily, cov_daily, horizon_days, initial_value, target_value
        )
        if p_trial > p0:
            w, p0 = trial, p_trial
            continue

        d = min(local_step, w[i])
        trial = w.copy()
        trial[i] -= d
        trial[j] += d
        trial = _apply_weight_cap(trial, cap)
        p_trial = _lognormal_terminal_prob(
            trial, mu_daily, cov_daily, horizon_days, initial_value, target_value
        )
        if p_trial > p0:
            w, p0 = trial, p_trial

    return w, p0


def _normalize_label(text) -> str:
    if text is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _coerce_cell_to_float(x):
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if x is None:
        return np.nan
    s = str(x).strip().replace("−", "-")
    if s.lower() in {"", "none", "nan"}:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _force_psd_correlation(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    R = 0.5 * (R + R.T)
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, eps, None)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(R_psd), eps, None))
    R_corr = R_psd / np.outer(d, d)
    np.fill_diagonal(R_corr, 1.0)
    return np.clip(R_corr, -1.0, 1.0)


def _normalize_corr_frame(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    tickers = list(tickers)
    n = len(tickers)
    if set(tickers).issubset(df.columns) and set(tickers).issubset(df.index):
        df2 = df.loc[tickers, tickers].copy()
    else:
        df2 = df.iloc[:n, :n].copy()
        df2.index = tickers
        df2.columns = tickers
    df2 = df2.apply(pd.to_numeric, errors="coerce")
    if df2.shape != (n, n) or df2.isna().any().any():
        raise ValueError("Correlation matrix must be numeric and N×N.")
    return df2


def read_excel_correlation(file, tickers: list[str]) -> np.ndarray:
    tickers_clean = [t.strip().upper() for t in tickers]
    tickers_norm = [_normalize_label(t) for t in tickers_clean]
    n = len(tickers_clean)

    file.seek(0)
    xls = pd.ExcelFile(file)
    sheets = xls.sheet_names
    names_lower = [s.lower().strip() for s in sheets]
    default_index = names_lower.index("total corr") if "total corr" in names_lower else 0

    chosen_sheet = st.selectbox(
        "Correlation sheet",
        options=sheets,
        index=default_index,
        key="corr_sheet_choice",
    )

    file.seek(0)
    raw = pd.read_excel(file, sheet_name=chosen_sheet, header=None, dtype=object)

    st.caption(f"Preview of '{chosen_sheet}' (top-left block)")
    st.dataframe(raw.iloc[:12, :12])

    col_labels_raw = raw.iloc[0, 1:].tolist()
    row_labels_raw = raw.iloc[1:, 0].tolist()
    col_labels = ["" if v is None else str(v).strip().upper() for v in col_labels_raw]
    row_labels = ["" if v is None else str(v).strip().upper() for v in row_labels_raw]
    col_norm = [_normalize_label(c) for c in col_labels]
    row_norm = [_normalize_label(r) for r in row_labels]

    row_map = {row_norm[i]: i for i in range(len(row_norm)) if row_norm[i]}
    col_map = {col_norm[i]: i for i in range(len(col_norm)) if col_norm[i]}
    all_keys = sorted(set(row_map.keys()) | set(col_map.keys()))

    mapped_keys = []
    for u in tickers_norm:
        if u in row_map and u in col_map:
            mapped_keys.append(u)
            continue
        if u in row_map or u in col_map:
            mapped_keys.append(u)
            continue
        suggestion = difflib.get_close_matches(u, all_keys, n=1, cutoff=0.6)
        if suggestion:
            st.info(f"Mapped '{u}' to '{suggestion[0]}' using fuzzy matching.")
            mapped_keys.append(suggestion[0])
        else:
            choice = st.selectbox(
                f"Choose Excel header for '{u}':",
                options=["<pick>"] + row_labels + col_labels,
                key=f"map_{u}",
            )
            if choice == "<pick>":
                st.stop()
            mapped_keys.append(_normalize_label(choice))

    try:
        row_idx = [1 + row_map[k] for k in mapped_keys]
        col_idx = [1 + col_map[k] for k in mapped_keys]
    except KeyError:
        row_idx = []
        col_idx = []
        for k in mapped_keys:
            ri = row_map.get(k, None)
            ci = col_map.get(k, None)
            if ri is None:
                alt = difflib.get_close_matches(k, list(row_map.keys()), n=1, cutoff=0.4)
                if alt:
                    ri = row_map[alt[0]]
            if ci is None:
                alt = difflib.get_close_matches(k, list(col_map.keys()), n=1, cutoff=0.4)
                if alt:
                    ci = col_map[alt[0]]
            if ri is None or ci is None:
                st.error("Unable to match a ticker on both axes.")
                st.stop()
            row_idx.append(1 + ri)
            col_idx.append(1 + ci)

    mat = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            v = _coerce_cell_to_float(raw.iat[row_idx[i], col_idx[j]])
            if np.isnan(v) and i != j:
                v = _coerce_cell_to_float(raw.iat[row_idx[j], col_idx[i]])
            mat[i, j] = v

    np.fill_diagonal(mat, 1.0)
    if np.isnan(mat).any():
        st.warning("Some cells missing on both triangles; replaced with 0.")
        mat = np.where(np.isnan(mat), 0.0, mat)

    mat = 0.5 * (mat + mat.T)
    mat = np.clip(mat, -1.0, 1.0)

    try:
        np.linalg.cholesky(mat + 1e-12 * np.eye(n))
    except np.linalg.LinAlgError:
        st.warning("Uploaded matrix adjusted to nearest PSD correlation.")
        mat = _force_psd_correlation(mat)

    return mat


def correlation_input_widget(tickers: list[str]) -> np.ndarray:
    st.subheader("Correlation Matrix")
    mode = st.radio(
        "How do you want to provide correlations?",
        [
            "Upload Excel (labeled triangle)",
            "Manual entry",
            "Paste CSV",
            "Estimate from recent prices",
        ],
        horizontal=True,
        key="corr_mode",
    )

    n = len(tickers)

    if mode == "Manual entry":
        rows = []
        for i, ti in enumerate(tickers):
            cols = st.columns(n)
            cur_row = []
            for j, tj in enumerate(tickers):
                base = 1.0 if i == j else 0.30
                with cols[j]:
                    val = st.number_input(
                        f"ρ({ti},{tj})",
                        value=float(base),
                        step=0.05,
                        format="%.2f",
                        key=f"rho_{i}_{j}",
                    )
                cur_row.append(val)
            rows.append(cur_row)
        R = np.asarray(rows, dtype=float)

    elif mode == "Paste CSV":
        template = "\n".join(
            [
                ",".join(["1.00" if i == j else "0.30" for j in range(n)])
                for i in range(n)
            ]
        )
        csv_text = st.text_area(
            "N×N CSV matrix (with or without headers/index):",
            value=template,
            height=180,
            key="corr_csv_text",
        )
        df_csv = pd.read_csv(StringIO(csv_text), header=None)
        R = _normalize_corr_frame(df_csv, tickers).to_numpy(float)

    elif mode == "Estimate from recent prices":
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start date", value=date(2024, 1, 1))
        with c2:
            end = st.date_input("End date", value=date.today())
        if st.button("Fetch & compute correlations", key="estimate_corr_btn"):
            try:
                import yfinance as yf

                series_map = {}
                for t in tickers:
                    df_p = yf.download(
                        t,
                        start=start,
                        end=end,
                        progress=False,
                        auto_adjust=False,
                    )
                    if df_p is None or df_p.empty:
                        st.warning(f"No data for {t}; skipped.")
                        continue
                    col = "Adj Close" if "Adj Close" in df_p.columns else "Close"
                    s = df_p[col].dropna()
                    if s.empty:
                        st.warning(f"No close prices for {t}.")
                        continue
                    series_map[t] = s
                if not series_map:
                    st.error("Unable to build any price series.")
                    st.stop()
                wide = pd.concat(series_map, axis=1).dropna(how="any")
                rets = wide.pct_change().dropna(how="any")
                df_corr = rets.corr()
                R = _normalize_corr_frame(df_corr, tickers).to_numpy(float)
            except Exception as e:
                st.error(f"Correlation estimation failed: {e}")
                st.stop()
        else:
            st.stop()
    else:
        upload = st.file_uploader(
            "Excel file with correlation sheet (tickers on first row/column)",
            type=["xlsx", "xls"],
        )
        if upload is None:
            st.info("Upload an Excel file to proceed.")
            st.stop()
        R = read_excel_correlation(upload, tickers)

    R = np.asarray(R, dtype=float)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    if (R < -1.0001).any() or (R > 1.0001).any():
        st.error("Correlation entries must lie in [-1, 1].")
        st.stop()
    try:
        np.linalg.cholesky(R + 1e-12 * np.eye(n))
    except np.linalg.LinAlgError:
        st.warning("Adjusted to nearest PSD correlation.")
        R = _force_psd_correlation(R)

    st.success("Correlation matrix accepted.")
    st.dataframe(
        pd.DataFrame(R, index=tickers, columns=tickers).style.format("{:.2f}")
    )
    return R



def summarize_terminal_values(final_values: np.ndarray, initial_value: float) -> dict:
    fv = np.asarray(final_values, dtype=float)
    pct = np.percentile(fv, [5, 25, 50, 75, 95])
    return {
        "Mean": f"${fv.mean():,.2f}",
        "Median": f"${np.median(fv):,.2f}",
        "5th %": f"${pct[0]:,.2f}",
        "25th %": f"${pct[1]:,.2f}",
        "75th %": f"${pct[3]:,.2f}",
        "95th %": f"${pct[4]:,.2f}",
        "Initial": f"${initial_value:,.2f}",
    }


def render_path_chart(all_paths: np.ndarray, max_paths: int = 200):
    n_show = min(max_paths, all_paths.shape[1])
    df = pd.DataFrame(all_paths[:, :n_show])
    df.index.name = "Day"
    st.line_chart(df)


def render_histogram(final_values: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(final_values, bins=50)
    ax.set_title("Final Portfolio Value Distribution")
    ax.set_xlabel("Final Value ($)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


def sharpe_ratio_from_paths(
    all_paths: np.ndarray, risk_free_rate: float = 0.035, periods_per_year: int = 252
) -> float:
    daily_returns = all_paths[1:, :] / all_paths[:-1, :] - 1.0
    mu_d = daily_returns.mean()
    sd_d = daily_returns.std()
    if sd_d == 0:
        return np.nan
    return ((mu_d * periods_per_year) - risk_free_rate) / (
        sd_d * np.sqrt(periods_per_year)
    )


def max_drawdown_from_paths(all_paths: np.ndarray) -> float:
    peaks = np.maximum.accumulate(all_paths, axis=0)
    dd = (all_paths - peaks) / peaks
    return float(dd.min())


def sortino_ratio_from_paths(
    all_paths: np.ndarray, risk_free_rate: float = 0.035, periods_per_year: int = 252
) -> float:
    daily_returns = all_paths[1:, :] / all_paths[:-1, :] - 1.0
    mu_d = daily_returns.mean()
    downside = daily_returns[daily_returns < 0]
    if downside.size == 0:
        return np.nan
    sd_neg = downside.std()
    if sd_neg == 0:
        return np.nan
    return ((mu_d * periods_per_year) - risk_free_rate) / (
        sd_neg * np.sqrt(periods_per_year)
    )



def main():
    st.set_page_config(page_title="Forward-Looking Monte Carlo", layout="wide")
    st.title("Forward-Looking Portfolio Monte Carlo")

    st.write(
        """
This tool simulates portfolio outcomes based on your own assumptions:
annual return, volatility, and a correlation matrix that **you** specify
(rather than fitting everything directly from history).
"""
    )

    st.header("1. Assets and Weights")
    raw_tickers = st.text_area(
        "Tickers (comma-separated):",
        value="AAPL, NVDA, MSFT",
    )
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    if not tickers:
        st.error("Please provide at least one ticker.")
        st.stop()
    n = len(tickers)
    st.write(f"Assets in portfolio: **{n}**")

    default_w = 1.0 / n
    weights_df = pd.DataFrame(
        {"Ticker": tickers, "Weight": [default_w] * n}
    )
    st.info("Edit weights so they sum to 1.0.")
    weights_df = st.data_editor(
        weights_df,
        num_rows="fixed",
        key="weights_editor",
    )
    total_w = float(weights_df["Weight"].sum())
    if abs(total_w - 1.0) > 1e-6:
        st.error(f"Weights must sum to 1.0 (current total: {total_w:.4f}).")
        st.stop()
    user_weights = weights_df["Weight"].to_numpy(float)

    initial_investment = st.number_input(
        "Initial investment ($):",
        value=100000.0,
        min_value=0.0,
        step=1000.0,
    )

    st.header("2. Return and Volatility Assumptions (annualized)")
    c1, c2 = st.columns(2)
    exp_returns = []
    volatilities = []
    for t in tickers:
        with c1:
            r = (
                st.number_input(
                    f"Expected return ({t}) %", value=8.0, step=0.5, key=f"ret_{t}"
                )
                / 100.0
            )
        with c2:
            v = (
                st.number_input(
                    f"Volatility ({t}) %", value=20.0, step=0.5, key=f"vol_{t}"
                )
                / 100.0
            )
        exp_returns.append(r)
        volatilities.append(v)

    exp_returns = np.asarray(exp_returns, dtype=float)
    volatilities = np.asarray(volatilities, dtype=float)

    st.header("3. Correlation Structure")
    corr_matrix = correlation_input_widget(tickers)

    mu_daily = exp_returns / 252.0
    sigma_daily = volatilities / np.sqrt(252.0)
    cov_daily = np.outer(sigma_daily, sigma_daily) * corr_matrix

    st.header("4. Simulation Settings")
    num_sim = st.number_input(
        "Number of Monte Carlo paths",
        value=10000,
        min_value=100,
        step=100,
    )
    horizon_days = st.number_input(
        "Horizon (trading days)",
        value=2520,
        min_value=1,
        step=1,
        help="Example: 252 ≈ 1 year, 2520 ≈ 10 years.",
    )
    use_drift_corr = st.checkbox("Use drift correction (μ − ½σ²)", value=True)
    seed_value = st.number_input("Random seed (optional)", value=42, step=1)

    st.header("5. Optimize for Target Final Value")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        target_final = st.number_input(
            "Target final value ($)",
            value=200000.0,
            min_value=0.0,
            step=1000.0,
        )
    with col_b:
        weight_cap = st.number_input(
            "Per-asset cap (0–1)",
            value=1.0,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )
        cap = float(weight_cap) if weight_cap < 1.0 else None
    with col_c:
        opt_seed = st.number_input("Optimizer seed", value=123, step=1)

    if st.button("Search for weights", type="primary"):
        try:
            w_opt, p_opt = search_weights_for_target(
                mu_daily=mu_daily,
                cov_daily=cov_daily,
                horizon_days=int(horizon_days),
                initial_value=float(initial_investment),
                target_value=float(target_final),
                n_assets=n,
                seed=int(opt_seed),
                n_candidates=2000,
                n_iters=4000,
                local_step=0.02,
                cap=cap,
            )
            st.success(
                f"Estimated probability of reaching ${target_final:,.0f}: {p_opt:.2%}"
            )
            st.subheader("Suggested weights")
            st.dataframe(
                pd.DataFrame({"Ticker": tickers, "Weight": w_opt}).style.format(
                    {"Weight": "{:.4f}"}
                )
            )
            st.session_state["opt_weights"] = w_opt
            st.session_state["opt_prob"] = float(p_opt)
            st.session_state["opt_target"] = float(target_final)
        except Exception as e:
            st.error(f"Weight search failed: {e}")

    use_opt_weights = st.toggle("Use suggested weights in simulation")
    sim_weights = user_weights
    if use_opt_weights and "opt_weights" in st.session_state:
        sim_weights = np.asarray(st.session_state["opt_weights"], dtype=float)

    if st.button("Run Monte Carlo simulation", key="run_mc", type="secondary"):
        try:
            engine = MonteCarloSimulation(
                tickers=tickers,
                exp_returns=exp_returns,
                volatilities=volatilities,
                corr_matrix=corr_matrix,
                weights=sim_weights,
                initial_investment=initial_investment,
            )
        except Exception as e:
            st.error(f"Simulation setup error: {e}")
            st.stop()

        all_paths, final_values = engine.run(
            n_paths=int(num_sim),
            horizon_days=int(horizon_days),
            use_drift_correction=bool(use_drift_corr),
            random_seed=int(seed_value),
        )

        sharpe = sharpe_ratio_from_paths(all_paths, risk_free_rate=0.035)
        sortino = sortino_ratio_from_paths(all_paths, risk_free_rate=0.035)
        mdd = max_drawdown_from_paths(all_paths)
        target_used = st.session_state.get("opt_target", 1_500_000.0)
        prob_reach = float(np.mean(final_values >= target_used))

        st.subheader("Risk / Performance Metrics")
        st.write(f"Sharpe ratio: **{sharpe:.3f}**")
        st.write(f"Sortino ratio: **{sortino:.3f}**")
        st.write(f"Max drawdown: **{mdd:.2%}**")
        st.write(
            f"Probability of finishing above ${target_used:,.0f}: **{prob_reach:.2%}**"
        )

        st.success(
            f"Monte Carlo complete: {num_sim:,} paths over {horizon_days:,} trading days."
        )

        st.subheader("Portfolio summary")
        for k, v in engine.summary().items():
            st.write(f"**{k}**: {v}")
        if use_opt_weights and "opt_prob" in st.session_state and "opt_target" in st.session_state:
            st.info(
                f"Analytic probability of reaching ${st.session_state['opt_target']:,.0f}: "
                f"{st.session_state['opt_prob']:.2%}"
            )

        st.subheader("Sample paths")
        render_path_chart(all_paths, max_paths=200)

        st.subheader("Final value distribution")
        render_histogram(final_values)

        st.subheader("Key percentiles")
        for k, v in summarize_terminal_values(
            final_values, float(initial_investment)
        ).items():
            st.write(f"**{k}**: {v}")


if __name__ == "__main__":
    main()
