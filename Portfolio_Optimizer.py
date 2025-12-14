import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
import json
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
np.random.seed(42)

KNOWN_ETFS = {
    'ARKK', 'GRID', 'FAN', 'PAVE', 'TAN', 'PHO', 'PBW', 'IBB', 'DGRO', 'ESGU', 'ICLN', 'INDA', 'EWW', 'EWT', 'ITA',
    'BBCA', 'XLU',
    'ESGV', 'VWO', 'VHT', 'VNQ', 'VTI', 'VT', 'VSS', 'VNQI', 'MSOS', 'IPO', 'JEPI', 'COWZ', 'LCTU', 'XLC', 'XLP', 'XLE',
    'XLF', 'XLV',
    'XLI', 'RSP', 'ESGE', 'EWA', 'EWZ', 'MCHI', 'DSI', 'USMV', 'QUAL', 'ESGD', 'MOAT', 'VEU', 'VEA', 'VGK', 'VOO',
    'VXUS', 'XLB', 'XLY',
    'QQQ', 'MTUM', 'IWF', 'XLK', 'SMH', 'VUG', 'VGT', 'IWD', 'NOBL', 'SCHD', 'VIG', 'VYM', 'VTV', 'IJH', 'VO', 'IJR',
    'IWM', 'AVUV', 'AGG',
    'JPST', 'BSV', 'BND', 'BNDX', 'FTSL', 'HYLS', 'IGSB', 'FALN', 'HYG', 'LQD', 'JNK', 'VCIT', 'VCSH', 'SHY', 'TLT',
    'IEF', 'EMB', 'SHV', 'GOVT',
    'BIL', 'VGSH', 'USFR', 'EMLC', 'VTIP', 'TIP', 'MBB', 'MUB', 'VTEB'
}


class ImpliedVolatilityPortfolioOptimizer:
    def __init__(
            self,
            tickers,
            lookback_years=3,
            risk_free_rate=0.04,
            iv_multiplier=0.7,
            blend_alpha=0.6,
            use_black_litterman=False,
            bl_tau=0.05,
            min_weight=0.0,
            max_weight=0.35,
            jitter=1e-8,
            target_return=None,
            optimize_method="max_sharpe",
            use_intrinsic_value=True,
    ):
        self.tickers = list(tickers)
        self.lookback_years = lookback_years
        self.risk_free_rate = risk_free_rate
        self.iv_multiplier = iv_multiplier
        self.blend_alpha = blend_alpha
        self.use_black_litterman = use_black_litterman
        self.bl_tau = bl_tau
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.jitter = jitter
        self.target_return = target_return
        self.optimize_method = optimize_method
        self.use_intrinsic_value = use_intrinsic_value

        self.prices = None
        self.returns = None
        self.mu_hist = None
        self.sigma_hist = None
        self.cov_matrix = None
        self.implied_vols = None
        self.iv_skew = None
        self.intrinsic_upsides = None
        self.mu_final = None
        self.weights = None
        self.diagnostics = {}
        self.bl_diagnostics = {}

    def fetch_price_data(self):
        print(f"Fetching {self.lookback_years}yr price data for {len(self.tickers)} tickers")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_years * 365)

        raw = yf.download(self.tickers, start=start_date, end=end_date, progress=False, threads=True)
        if raw.empty:
            raise ValueError("No price data returned from yfinance")

        if isinstance(raw.columns, pd.MultiIndex):
            if "Adj Close" in raw.columns.levels[0]:
                prices = raw["Adj Close"].copy()
            elif "Close" in raw.columns.levels[0]:
                prices = raw["Close"].copy()
            else:
                prices = raw.iloc[:, raw.columns.get_level_values(0) == raw.columns.levels[0][0]]
                prices = prices.iloc[:, 0::raw.shape[1] // len(self.tickers)]
        else:
            if "Adj Close" in raw.columns:
                prices = raw["Adj Close"].copy()
            elif "Close" in raw.columns:
                prices = raw["Close"].copy()
            else:
                prices = raw.copy()

        prices = prices.reindex(columns=self.tickers)
        prices = prices.sort_index().ffill().bfill().dropna(how="all")

        if prices.shape[0] < 10:
            raise ValueError("Too few price rows after cleaning.")

        self.prices = prices
        self.returns = prices.pct_change().dropna()
        print(f"  {len(self.returns)} days of returns")

    def compute_historical_statistics(self):
        print("Computing historical stats")
        trading_days = 252.0
        self.mu_hist = self.returns.mean() * trading_days
        self.sigma_hist = self.returns.std() * np.sqrt(trading_days)

        try:
            lw = LedoitWolf()
            lw.fit(self.returns.values)
            cov_ann = lw.covariance_ * trading_days
            cov_ann += np.eye(len(self.tickers)) * max(self.jitter, 1e-12)
            self.cov_matrix = pd.DataFrame(cov_ann, index=self.tickers, columns=self.tickers)
        except Exception as e:
            print(f"  LedoitWolf failed: {e}, using sample covariance")
            cov_ann = self.returns.cov().values * trading_days
            cov_ann += np.eye(len(self.tickers)) * max(self.jitter, 1e-8)
            self.cov_matrix = pd.DataFrame(cov_ann, index=self.tickers, columns=self.tickers)

        print(
            f"  mu: [{self.mu_hist.min():.2%}, {self.mu_hist.max():.2%}], sigma: [{self.sigma_hist.min():.2%}, {self.sigma_hist.max():.2%}]")

    def implied_vol_from_chain(self, calls, puts, spot_price, return_skew=False):
        try:
            iv_by_type = {'calls': np.nan, 'puts': np.nan}

            for chain_type, chain in [('calls', calls), ('puts', puts)]:
                if chain is None or chain.empty:
                    continue
                chain = chain.copy()
                chain["distance"] = np.abs(chain["strike"] - spot_price)
                atm_idx = chain["distance"].idxmin()
                pos = chain.index.get_loc(atm_idx)
                idxs = [chain.index[j] for k in (-1, 0, 1) if 0 <= (j := pos + k) < len(chain)]
                sample = chain.loc[idxs]
                ivs = sample["impliedVolatility"].dropna().values
                if len(ivs) > 0:
                    iv_by_type[chain_type] = np.median(ivs)

            valid_ivs = [v for v in iv_by_type.values() if np.isfinite(v)]
            if len(valid_ivs) == 0:
                return (np.nan, np.nan) if return_skew else np.nan

            final_iv = float(np.clip(np.median(valid_ivs), 0.02, 3.0))

            if return_skew:
                if np.isfinite(iv_by_type['calls']) and np.isfinite(iv_by_type['puts']):
                    avg_iv = (iv_by_type['calls'] + iv_by_type['puts']) / 2
                    if avg_iv > 0.001:
                        skew = float(np.clip((iv_by_type['calls'] - iv_by_type['puts']) / avg_iv, -1.0, 1.0))
                    else:
                        skew = 0.0
                else:
                    skew = 0.0
                return final_iv, skew

            return final_iv
        except Exception:
            return (np.nan, np.nan) if return_skew else np.nan

    def interpolate_to_30d(self, iv_by_days, skew_by_days=None):
        if len(iv_by_days) == 0:
            return (np.nan, np.nan) if skew_by_days is not None else np.nan

        days = np.array(sorted(iv_by_days.keys()))
        vols = np.array([iv_by_days[d] for d in days], dtype=float)

        mask = np.isfinite(vols)
        if mask.sum() == 0:
            return (np.nan, np.nan) if skew_by_days is not None else np.nan

        days_clean, vols_clean = days[mask], vols[mask]
        vars_ = (vols_clean ** 2) * days_clean

        if len(days_clean) == 1:
            var_30 = vars_[0] * (30.0 / days_clean[0])
        else:
            coeffs = np.polyfit(days_clean, vars_, 1)
            var_30 = max(np.polyval(coeffs, 30.0), 1e-8)

        vol_30 = float(np.clip(np.sqrt(var_30 / 30.0), 0.02, 3.0))

        if skew_by_days is not None:
            skew_values = np.array([skew_by_days.get(d, np.nan) for d in days_clean], dtype=float)
            skew_mask = np.isfinite(skew_values)

            if skew_mask.sum() == 0:
                skew_30 = 0.0
            elif skew_mask.sum() == 1:
                skew_30 = float(skew_values[skew_mask][0])
            else:
                days_skew, skews = days_clean[skew_mask], skew_values[skew_mask]
                if len(days_skew) >= 2:
                    coeffs_skew = np.polyfit(days_skew, skews, 1)
                    skew_30 = float(np.clip(np.polyval(coeffs_skew, 30.0), -1.0, 1.0))
                else:
                    skew_30 = float(np.clip(np.mean(skews), -1.0, 1.0))
            return vol_30, skew_30

        return vol_30

    def fetch_implied_volatilities(self):
        print("Fetching implied volatilities")
        iv_dict, skew_dict = {}, {}
        today = datetime.utcnow()

        for ticker in self.tickers:
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period="5d")
                spot = float(hist["Close"].iloc[-1]) if not hist.empty else float(self.prices[ticker].iloc[-1])

                exp_dates = tk.options
                if not exp_dates:
                    iv_fallback = float(self.sigma_hist.get(ticker, 0.2))
                    iv_dict[ticker] = float(np.clip(iv_fallback if np.isfinite(iv_fallback) else 0.2, 0.02, 3.0))
                    skew_dict[ticker] = 0.0
                    print(f"  {ticker}: no options, using hist vol {iv_dict[ticker]:.1%}")
                    continue

                iv_by_days, skew_by_days = {}, {}
                for ed in exp_dates[:8]:
                    try:
                        opt_chain = tk.option_chain(ed)
                        dt = pd.to_datetime(ed).to_pydatetime()
                        days = max(1, (dt - today).days)
                        iv, skew = self.implied_vol_from_chain(opt_chain.calls, opt_chain.puts, spot, return_skew=True)
                        if np.isfinite(iv):
                            iv_by_days[days] = iv
                            if np.isfinite(skew):
                                skew_by_days[days] = skew
                    except Exception:
                        continue

                result = self.interpolate_to_30d(iv_by_days, skew_by_days)
                if isinstance(result, tuple):
                    iv30, skew30 = result
                else:
                    iv30, skew30 = result, 0.0

                if not np.isfinite(iv30):
                    iv30 = float(np.clip(self.sigma_hist.get(ticker, 0.2), 0.02, 3.0))
                    skew30 = 0.0
                    print(f"  {ticker}: IV failed, using hist vol {iv30:.1%}")
                else:
                    print(f"  {ticker}: IV={iv30:.1%}, skew={skew30:+.2f}")

                iv_dict[ticker] = iv30
                skew_dict[ticker] = skew30 if np.isfinite(skew30) else 0.0

            except Exception as e:
                iv_fb = float(np.clip(self.sigma_hist.get(ticker, 0.2), 0.02, 3.0))
                iv_dict[ticker] = iv_fb
                skew_dict[ticker] = 0.0
                print(f"  {ticker}: error ({e}), using hist vol {iv_dict[ticker]:.1%}")

        self.implied_vols = pd.Series(iv_dict)
        self.iv_skew = pd.Series(skew_dict)

    def intrinsic_value_calculator(self, ticker, terminal_growth=0.01, years=10, max_growth_cap=0.20):
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        # Financial services use earnings-based model
        if ("Financial Services" in sector) or ("Insurance" in sector) or ("Reinsurance" in industry):
            try:
                price = stock.history(period="1d")["Close"].iloc[-1]
            except Exception:
                price = None

            net_income = info.get("netIncomeToCommon") or 0.0
            shares_outstanding = info.get("sharesOutstanding")
            book_value_per_share = info.get("bookValue") or 0.0

            eps_reported = info.get("trailingEps") or info.get("forwardEps")
            try:
                eps_reported = float(eps_reported) if eps_reported is not None else None
            except Exception:
                eps_reported = None

            eps_from_totals = None
            try:
                if net_income and shares_outstanding and float(shares_outstanding) > 1000000:
                    eps_from_totals = float(net_income) / float(shares_outstanding)
            except Exception:
                pass

            if eps_reported and np.isfinite(eps_reported) and eps_reported > 0:
                eps = float(eps_reported)
            elif eps_from_totals and np.isfinite(eps_from_totals) and eps_from_totals > 0:
                eps = float(eps_from_totals)
            else:
                raise ValueError(f"{ticker}: Unable to determine reliable EPS")

            growth_rate = float(info.get("earningsGrowth", 0.05) or 0.05)
            growth_rate = max(float(np.clip(growth_rate, -0.1, max_growth_cap)), 0.01)

            risk_free, market_return = 0.045, 0.09
            beta = float(info.get("beta", 1.0) or 1.0)
            cost_of_equity = np.clip(risk_free + beta * (market_return - risk_free), 0.08, 0.10)

            projected_eps = []
            last_eps = float(eps)
            for t in range(1, years + 1):
                year_growth = growth_rate - (growth_rate - terminal_growth) * (t / years)
                last_eps = last_eps * (1.0 + year_growth)
                projected_eps.append(last_eps / ((1.0 + cost_of_equity) ** t))

            payout_ratio = 0.35
            denom = max(cost_of_equity - terminal_growth, 0.0001)
            justified_pe = (1.0 - payout_ratio) / max(denom, 0.01)
            terminal_pe = float(np.clip(justified_pe, 5, 14))
            terminal_value = (last_eps * terminal_pe) / ((1.0 + cost_of_equity) ** years)
            intrinsic_per_share = sum(projected_eps) + terminal_value

            try:
                book_value_per_share = float(book_value_per_share) if book_value_per_share else 0.0
            except Exception:
                book_value_per_share = 0.0

            if price and book_value_per_share > 0 and price > 0:
                ratio = book_value_per_share / price
                if ratio > 50:
                    possible_factor = int(round(ratio))
                    if 2 <= possible_factor <= 5000:
                        book_value_per_share = book_value_per_share / possible_factor

            retention = 1.0 - payout_ratio
            implied_roe = growth_rate / max(retention, 0.2)
            sustainable_roe = float(np.clip(implied_roe, 0.03, 0.12))
            book_value_future = book_value_per_share * ((1.0 + sustainable_roe * retention) ** years)

            book_component = min(0.1 * book_value_future, intrinsic_per_share * 3) if intrinsic_per_share > 0 else 0
            float_uplift = min(0.05 * intrinsic_per_share, 0.5 * book_value_per_share)
            final_intrinsic = 0.60 * intrinsic_per_share + book_component + float_uplift

            return 100 * (final_intrinsic - price) / price

        else:
            # Non-financial companies use FCF-based DCF
            def get_free_cash_flow(stock):
                cashflow = stock.cashflow
                if not cashflow.empty:
                    op_cf = None
                    for c in ["Total Cash From Operating Activities", "Operating Cash Flow", "Operating Cashflow"]:
                        if c in cashflow.index:
                            s = cashflow.loc[c].dropna()
                            op_cf = s.iloc[0:2].mean() if len(s) >= 2 else (s.iloc[0] if len(s) > 0 else None)
                            if op_cf is not None:
                                break
                    capex = 0
                    for c in ["Capital Expenditures", "Capex"]:
                        if c in cashflow.index:
                            s = cashflow.loc[c].dropna()
                            capex = s.iloc[0:2].mean() if len(s) >= 2 else (s.iloc[0] if len(s) > 0 else 0)
                            break
                    if op_cf is not None:
                        return float(op_cf) - float(capex or 0)

                fcf = stock.info.get("freeCashflow")
                if fcf:
                    return float(fcf)
                op_cf = stock.info.get("operatingCashflow")
                if op_cf:
                    return float(op_cf) - float(stock.info.get("capitalExpenditures", 0) or 0)
                net_income = stock.info.get("netIncomeToCommon")
                if net_income:
                    return float(net_income) * 0.8
                return None

            fcf = get_free_cash_flow(stock)
            shares_outstanding = stock.info.get("sharesOutstanding")
            try:
                shares_outstanding = float(shares_outstanding) if shares_outstanding else None
            except Exception:
                shares_outstanding = None

            balance_sheet = stock.balance_sheet
            if balance_sheet.empty:
                total_debt, cash = 0.0, 0.0
            else:
                total_debt = balance_sheet.loc["Total Debt"].iloc[0] if "Total Debt" in balance_sheet.index else 0.0
                cash = balance_sheet.loc["Cash"].iloc[0] if "Cash" in balance_sheet.index else 0.0
                total_debt = 0.0 if total_debt is None or np.isnan(total_debt) else total_debt
                cash = 0.0 if cash is None or np.isnan(cash) else cash
            net_debt = total_debt - cash

            income_stmt = stock.financials
            auto_growth = 0.05
            if not income_stmt.empty and "Total Revenue" in income_stmt.index:
                revenues = income_stmt.loc["Total Revenue"].dropna().values
                if len(revenues) >= 2:
                    years_span = min(3, len(revenues) - 1)
                    newest, earliest = revenues[0], revenues[years_span]
                    if earliest > 0 and newest > 0:
                        try:
                            auto_growth = (newest / earliest) ** (1.0 / years_span) - 1.0
                        except Exception:
                            pass

            analyst_growth = stock.info.get("earningsGrowth")
            try:
                analyst_growth = float(analyst_growth) if analyst_growth else None
            except Exception:
                analyst_growth = None

            growth_rate = (auto_growth + analyst_growth) / 2.0 if analyst_growth and analyst_growth > 0 else auto_growth
            if not np.isfinite(growth_rate):
                growth_rate = 0.05
            growth_rate = max(min(growth_rate, max_growth_cap), -0.6)

            risk_free, market_premium = 0.045, 0.045
            market_cap = float(info.get("marketCap") or 0)
            beta = float(stock.info.get("beta", 1.0) or 1.0)
            adj_beta = 0.67 * beta + 0.33 * 1.0
            capm = risk_free + adj_beta * market_premium

            if any(k in sector for k in ["Utility", "Defensive"]):
                base_low, base_high = 0.07, 0.08
            elif "Consumer Defensive" in sector:
                base_low, base_high = 0.075, 0.09
            elif any(k in sector for k in ["Health", "Industrial"]):
                base_low, base_high = 0.09, 0.11
            elif any(k in sector for k in ["Technology", "Semiconductor", "Software", "Communication"]):
                base_low, base_high = 0.095, 0.12
            elif any(k in sector for k in ["Energy", "Materials"]):
                base_low, base_high = 0.10, 0.14
            else:
                base_low, base_high = 0.09, 0.12

            cap_adj = -0.01 if market_cap > 200_000_000_000 else (0.01 if market_cap < 2_000_000_000 else 0.0)
            position = np.clip((adj_beta - 0.8) / 0.8, 0, 1)
            sector_anchor = base_low + position * (base_high - base_low)
            discount_rate = float(np.clip(0.6 * capm + 0.4 * sector_anchor + cap_adj, 0.05, 0.16))

            if not all([fcf and np.isfinite(fcf), shares_outstanding and shares_outstanding > 0,
                        discount_rate > 0, np.isfinite(terminal_growth)]):
                raise ValueError("Invalid inputs for valuation")

            denom = discount_rate - terminal_growth
            if denom <= 0:
                raise ValueError("Discount rate must exceed terminal growth")

            projected_fcfs = []
            last_fcf = float(fcf)
            for t in range(1, years + 1):
                fade = np.exp(-t / (years / 2))
                year_growth = terminal_growth + (growth_rate - terminal_growth) * fade
                last_fcf *= (1.0 + year_growth)
                projected_fcfs.append(last_fcf / ((1.0 + discount_rate) ** t))

            terminal_fcf = last_fcf * (1.0 + terminal_growth)
            terminal_value = terminal_fcf / max(denom, 0.005)
            discounted_terminal_value = min(terminal_value / ((1.0 + discount_rate) ** years), sum(projected_fcfs) * 3)

            enterprise_value = np.clip(sum(projected_fcfs) + discounted_terminal_value, 0, 50 * fcf)
            dcf_value = (enterprise_value - net_debt) / shares_outstanding

            eps = stock.info.get("forwardEps") or stock.info.get("trailingEps")
            try:
                eps = float(eps) if eps else None
            except Exception:
                eps = None

            try:
                price = stock.history(period="1d")["Close"].iloc[-1]
            except Exception:
                price = None

            pe_val, ev_ebitda_val, p_fcf_val, industry_pe_val, hist_pe_val = None, None, None, None, None

            forward_pe = stock.info.get("forwardPE")
            trailing_pe = stock.info.get("trailingPE")
            chosen_pe = forward_pe or trailing_pe
            if eps and chosen_pe:
                pe_val = chosen_pe * eps

            ev_to_ebitda = stock.info.get("enterpriseToEbitda")
            if ev_to_ebitda and ev_to_ebitda > 0:
                ebitda = stock.financials.loc["EBITDA"].iloc[0] if "EBITDA" in stock.financials.index else None
                if ebitda and shares_outstanding > 0:
                    ev_ebitda_val = (ebitda * ev_to_ebitda - net_debt) / shares_outstanding

            if fcf and shares_outstanding > 0 and price > 0:
                p_fcf = (price * shares_outstanding) / fcf
                if p_fcf > 0:
                    p_fcf_val = (fcf * p_fcf) / shares_outstanding

            sector_pe_map = {"Technology": 25, "Communication Services": 20, "Consumer Cyclical": 20,
                             "Consumer Defensive": 18, "Healthcare": 18, "Financial Services": 14,
                             "Industrials": 16, "Energy": 12, "Utilities": 10,
                             "Basic Materials": 14, "Real Estate": 15}
            sector_pe = sector_pe_map.get(sector, chosen_pe or 15)
            if eps and sector_pe:
                industry_pe_val = sector_pe * eps

            hist = stock.history(period="5y")
            if not hist.empty and eps and eps > 0:
                hist_pe = (hist["Close"] / eps).median()
                if hist_pe and np.isfinite(hist_pe):
                    hist_pe_val = hist_pe * eps

            rel_current = [v for v in [pe_val, ev_ebitda_val, p_fcf_val] if v and np.isfinite(v)]
            rel_anchors = [v for v in [industry_pe_val, hist_pe_val] if v and np.isfinite(v)]

            if rel_current or rel_anchors:
                current_val = np.median(rel_current) if rel_current else None
                anchor_val = np.median(rel_anchors) if rel_anchors else None
                if current_val and anchor_val:
                    relative_value = 0.6 * current_val + 0.4 * anchor_val
                else:
                    relative_value = current_val or anchor_val
            else:
                relative_value = dcf_value

            intrinsic_value = dcf_value * 0.5 + relative_value * 0.5
            return 100 * (intrinsic_value - price) / price

    def fetch_intrinsic_values(self):
        if not self.use_intrinsic_value:
            return

        print("Calculating intrinsic values")
        upside_dict = {}

        for ticker in self.tickers:
            try:
                if ticker.upper() in KNOWN_ETFS:
                    print(f"  {ticker}: ETF, skipped")
                    upside_dict[ticker] = 0.0
                    continue

                try:
                    tk = yf.Ticker(ticker)
                    quote_type = tk.info.get('quoteType', '').upper()
                    if quote_type in ['ETF', 'MUTUALFUND', 'INDEX', 'CURRENCY', 'CRYPTOCURRENCY', 'FUTURE', 'OPTION']:
                        print(f"  {ticker}: {quote_type}, skipped")
                        upside_dict[ticker] = 0.0
                        continue
                except Exception:
                    pass

                upside_pct = self.intrinsic_value_calculator(ticker)
                upside_decimal = upside_pct / 100.0
                upside_decimal_clipped = float(np.clip(upside_decimal, -0.5, 0.7))
                upside_dict[ticker] = upside_decimal_clipped

                if abs(upside_decimal - upside_decimal_clipped) > 0.001:
                    print(f"  {ticker}: {upside_pct:+.1f}% (clipped to {upside_decimal_clipped * 100:+.1f}%)")
                else:
                    print(f"  {ticker}: {upside_pct:+.1f}%")
            except Exception as e:
                print(f"  {ticker}: error - {e}")
                upside_dict[ticker] = 0.0

        self.intrinsic_upsides = pd.Series(upside_dict)

    def compute_blended_returns(self):
        print("Blending expected returns")
        rf = self.risk_free_rate
        mu_hist = self.mu_hist.copy()

        skew_factor = 0.5
        if self.iv_skew is not None:
            direction_multiplier = (1.0 + skew_factor * self.iv_skew).clip(0.5, 1.5)
            mu_iv = rf + self.iv_multiplier * self.implied_vols * direction_multiplier
        else:
            mu_iv = rf + self.iv_multiplier * self.implied_vols

        if self.use_intrinsic_value and self.intrinsic_upsides is not None:
            realization_period = 3.0
            capture_rate = 0.5
            w_iv_fixed = 0.40
            w_remaining = 0.60

            final_returns = pd.Series(index=self.tickers, dtype=float)
            self._per_ticker_weights = {}

            for ticker in self.tickers:
                hist_ret = mu_hist[ticker]
                iv_ret = mu_iv[ticker]
                upside = self.intrinsic_upsides[ticker]

                intrinsic_implied = rf + (upside * capture_rate / realization_period) + 0.05

                # Dynamic weighting based on historical extremity
                abs_hist = abs(hist_ret)
                dynamic_intrinsic_pct = np.clip(0.10 + 0.60 * min(abs_hist / 0.50, 1.0), 0.10, 0.70)

                w_hist_ticker = w_remaining * (1 - dynamic_intrinsic_pct)
                w_intrinsic_ticker = w_remaining * dynamic_intrinsic_pct

                self._per_ticker_weights[ticker] = {'w_hist': w_hist_ticker, 'w_iv': w_iv_fixed,
                                                    'w_intrinsic': w_intrinsic_ticker}
                final_returns[
                    ticker] = w_hist_ticker * hist_ret + w_iv_fixed * iv_ret + w_intrinsic_ticker * intrinsic_implied

            self._intrinsic_implied = rf + (self.intrinsic_upsides * capture_rate / realization_period) + 0.05
            self.mu_final = final_returns

            avg_w_hist = np.mean([w['w_hist'] for w in self._per_ticker_weights.values()])
            avg_w_intrinsic = np.mean([w['w_intrinsic'] for w in self._per_ticker_weights.values()])
            self._print_return_breakdown(mu_hist, mu_iv, avg_w_hist, w_iv_fixed, avg_w_intrinsic)
        else:
            self.mu_final = self.blend_alpha * mu_hist + (1.0 - self.blend_alpha) * mu_iv

        self.mu_final = pd.Series(np.clip(self.mu_final.values, -0.5, 1.5), index=self.tickers)
        print(f"  Final expected returns: [{self.mu_final.min():.1%}, {self.mu_final.max():.1%}]")

    def _print_return_breakdown(self, mu_hist, mu_iv, avg_w_hist, w_iv_fixed, avg_w_intrinsic):
        print("\nReturn Breakdown:")
        print(f"{'Ticker':<7} {'Type':<5} {'Hist':>7} {'IV':>6} {'Skew':>6} {'IVRet':>7} {'Intr':>7} {'Final':>7}")

        for ticker in self.tickers:
            is_etf = ticker.upper() in KNOWN_ETFS
            t = "ETF" if is_etf else "STK"

            hist_ret = mu_hist[ticker]
            iv_ret = mu_iv[ticker]
            iv_raw = self.implied_vols[ticker]
            skew = self.iv_skew[ticker] if self.iv_skew is not None else 0
            intr_implied = self._intrinsic_implied[ticker]

            weights = self._per_ticker_weights.get(ticker, {'w_hist': avg_w_hist, 'w_iv': w_iv_fixed,
                                                            'w_intrinsic': avg_w_intrinsic})
            final_ret = weights['w_hist'] * hist_ret + weights['w_iv'] * iv_ret + weights['w_intrinsic'] * intr_implied
            intr_str = "-" if is_etf else f"{intr_implied:>6.1%}"

            print(
                f"{ticker:<7} {t:<5} {hist_ret:>6.1%} {iv_raw:>6.1%} {skew:>+5.2f} {iv_ret:>6.1%} {intr_str:>7} {final_ret:>6.1%}")

        print(
            f"{'Avg':<7} {'':5} {mu_hist.mean():>6.1%} {self.implied_vols.mean():>6.1%} {self.iv_skew.mean():>+5.2f} {mu_iv.mean():>6.1%} {self._intrinsic_implied.mean():>7.1%} {self.mu_final.mean():>6.1%}")
        print()

    def fetch_market_caps(self):
        print("Fetching market caps for equilibrium weights")
        market_caps = {}

        for ticker in self.tickers:
            try:
                tk = yf.Ticker(ticker)
                info = tk.info
                quote_type = info.get('quoteType', '').upper()
                is_etf = quote_type == 'ETF' or ticker.upper() in KNOWN_ETFS

                cap, source = None, None

                if is_etf:
                    total_assets = info.get('totalAssets')
                    net_assets = info.get('netAssets')

                    if total_assets and total_assets > 0:
                        cap, source = float(total_assets), "totalAssets"
                    elif net_assets and net_assets > 0:
                        cap, source = float(net_assets), "netAssets"
                    else:
                        shares = info.get('sharesOutstanding')
                        if shares and shares > 0 and self.prices is not None and ticker in self.prices.columns:
                            cap, source = float(self.prices[ticker].iloc[-1] * shares), "price*shares"
                        else:
                            try:
                                hist = tk.history(period="5d")
                                implied_shares = info.get('impliedSharesOutstanding')
                                if not hist.empty and implied_shares and implied_shares > 0:
                                    cap, source = float(hist['Close'].iloc[-1] * implied_shares), "price*impliedShares"
                            except:
                                pass
                else:
                    cap = info.get('marketCap')
                    if cap and cap > 0:
                        source = "marketCap"
                    else:
                        shares = info.get('sharesOutstanding')
                        if shares and self.prices is not None and ticker in self.prices.columns:
                            cap, source = float(self.prices[ticker].iloc[-1] * shares), "price*shares"

                if cap and cap > 0:
                    market_caps[ticker] = cap
                    print(f"  {ticker}: {'ETF' if is_etf else 'STK'} - ${cap / 1e9:.1f}B ({source})")
                else:
                    market_caps[ticker] = None
                    print(f"  {ticker}: {'ETF' if is_etf else 'STK'} - no cap found, will use median")

            except Exception as e:
                market_caps[ticker] = None
                print(f"  {ticker}: error fetching cap ({e})")

        valid_caps = [c for c in market_caps.values() if c is not None and c > 0]
        median_cap = np.median(valid_caps) if valid_caps else 10e9

        for ticker in self.tickers:
            if market_caps[ticker] is None or market_caps[ticker] <= 0:
                market_caps[ticker] = median_cap
                print(f"  {ticker}: using median cap ${median_cap / 1e9:.1f}B")

        total_cap = sum(market_caps.values())
        print(f"\n  Equilibrium weights:")
        for ticker in sorted(self.tickers, key=lambda t: market_caps[t], reverse=True):
            print(f"    {ticker}: {market_caps[ticker] / total_cap:.1%}")

        return market_caps

    def apply_black_litterman(self, market_caps=None):
        # Covariance adjustment for diversification
        if not self.use_black_litterman:
            return

        print("Applying Black-Litterman (covariance adjustment only)")
        n = len(self.tickers)
        Σ = self.cov_matrix.values.copy()
        tau = self.bl_tau

        if market_caps is None:
            market_caps = self.fetch_market_caps()

        w_arr = np.array([market_caps.get(t, 1.0) for t in self.tickers], dtype=float)
        w_mkt = w_arr / w_arr.sum() if w_arr.sum() > 0 else np.ones(n) / n

        confidence = np.zeros(n)

        for i, ticker in enumerate(self.tickers):
            hist_ret = self.mu_hist[ticker]
            iv = self.implied_vols[ticker]
            skew = self.iv_skew[ticker] if self.iv_skew is not None else 0
            skew_adjustment = np.clip(1.0 + 0.5 * skew, 0.5, 1.5)
            iv_ret = self.risk_free_rate + self.iv_multiplier * iv * skew_adjustment

            if self.use_intrinsic_value and self.intrinsic_upsides is not None:
                intrinsic_ret = self.risk_free_rate + (self.intrinsic_upsides[ticker] * 0.5 / 3.0) + 0.05
            else:
                intrinsic_ret = hist_ret

            signals = [hist_ret, iv_ret, intrinsic_ret]
            signal_std = np.std(signals)
            signal_mean = np.mean(signals)
            cv = signal_std / abs(signal_mean) if abs(signal_mean) > 0.01 else signal_std / 0.01

            confidence[i] = 1.0 / (1.0 + cv * 2)

        print(f"  Signal confidence range: [{confidence.min():.2f}, {confidence.max():.2f}]")

        vols = np.sqrt(np.diag(Σ))
        vol_outer = np.outer(vols, vols)
        corr = Σ / (vol_outer + 1e-10)

        avg_corr = (corr.sum() - n) / (n * n - n)
        target_corr = np.full((n, n), avg_corr)
        np.fill_diagonal(target_corr, 1.0)
        target_cov = target_corr * vol_outer

        print(f"  Average correlation: {avg_corr:.3f}")

        shrinkage_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                pair_confidence = min(confidence[i], confidence[j])
                shrinkage_matrix[i, j] = tau * 10 * (1 - pair_confidence)

        shrinkage_matrix = np.clip(shrinkage_matrix, 0, 0.5)
        Σ_posterior = (1 - shrinkage_matrix) * Σ + shrinkage_matrix * target_cov

        min_eig = np.min(np.linalg.eigvalsh(Σ_posterior))
        if min_eig < 1e-8:
            Σ_posterior += np.eye(n) * (1e-8 - min_eig)

        old_cov = self.cov_matrix.copy()
        self.cov_matrix = pd.DataFrame(Σ_posterior, index=self.tickers, columns=self.tickers)

        def compute_portfolio_stats(mu, cov):
            try:
                Σ_inv = np.linalg.inv(cov + np.eye(n) * 1e-8)
                w_opt = np.clip(Σ_inv @ (mu - self.risk_free_rate), 0, None)
                w_opt = w_opt / w_opt.sum() if w_opt.sum() > 0 else np.ones(n) / n
                port_ret = w_opt @ mu
                port_vol = np.sqrt(w_opt @ cov @ w_opt)
                sharpe = (port_ret - self.risk_free_rate) / max(port_vol, 1e-8)
                concentration = np.sum(w_opt ** 2)
                return sharpe, port_vol, concentration
            except:
                return 0, 0, 1

        mu = self.mu_final.values
        sharpe_old, vol_old, conc_old = compute_portfolio_stats(mu, old_cov.values)
        sharpe_new, vol_new, conc_new = compute_portfolio_stats(mu, Σ_posterior)

        cov_correlation = np.corrcoef(old_cov.values.flatten(), Σ_posterior.flatten())[0, 1]
        avg_shrinkage = shrinkage_matrix.mean()

        print(f"\n  Covariance adjustment results:")
        print(f"    Average shrinkage applied: {avg_shrinkage:.1%}")
        print(f"    Covariance correlation (old vs new): {cov_correlation:.3f}")
        print(f"\n  Implied portfolio comparison (same returns, different cov):")
        print(f"    Old cov: Sharpe={sharpe_old:.2f}, Vol={vol_old:.1%}, Concentration={conc_old:.3f}")
        print(f"    New cov: Sharpe={sharpe_new:.2f}, Vol={vol_new:.1%}, Concentration={conc_new:.3f}")
        print(f"\n  Expected returns PRESERVED (not modified by BL)")
        print(f"    Returns range: [{self.mu_final.min():.1%}, {self.mu_final.max():.1%}]")

        self.bl_diagnostics = {
            'mode': 'covariance_only',
            'avg_shrinkage': avg_shrinkage,
            'avg_correlation_target': avg_corr,
            'cov_correlation': cov_correlation,
            'confidence': pd.Series(confidence, index=self.tickers),
            'sharpe_old_cov': sharpe_old,
            'sharpe_new_cov': sharpe_new,
            'vol_old': vol_old,
            'vol_new': vol_new,
            'concentration_old': conc_old,
            'concentration_new': conc_new,
            'market_cap_weights': pd.Series(w_mkt, index=self.tickers),
        }

    def neg_sharpe(self, w, mu, cov, rf):
        w = np.array(w)
        ret = float(w.dot(mu))
        vol = float(np.sqrt(max(1e-12, w.dot(cov).dot(w))))
        return -(ret - rf) / vol

    def volatility(self, w, cov):
        return float(np.sqrt(max(0.0, np.array(w).dot(cov).dot(np.array(w)))))

    def optimize_portfolio(self):
        print("Optimizing portfolio")
        n = len(self.tickers)
        mu = self.mu_final.values
        cov = self.cov_matrix.values
        rf = self.risk_free_rate

        if np.all(mu - rf <= 0) and self.optimize_method == "max_sharpe":
            print("  All excess returns <= 0, using min-variance instead")
            method = "min_variance"
        else:
            method = self.optimize_method

        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        x0 = np.ones(n) / n

        if method == "max_sharpe":
            res = minimize(self.neg_sharpe, x0, args=(mu, cov, rf), method="SLSQP",
                           bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
        elif method == "min_variance":
            res = minimize(lambda w: self.volatility(w, cov), x0, method="SLSQP",
                           bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
            print(f"  Min-variance optimization {'succeeded' if res.success else 'failed'}")
        elif method == "min_variance_target":
            if self.target_return is None:
                raise ValueError("target_return required for min_variance_target")
            cons2 = cons + [{"type": "ineq", "fun": lambda w: w.dot(mu) - self.target_return}]
            res = minimize(lambda w: self.volatility(w, cov), x0, method="SLSQP",
                           bounds=bounds, constraints=cons2, options={"maxiter": 1000, "ftol": 1e-10})
        else:
            raise ValueError(f"Unknown method: {method}. Options: max_sharpe, min_variance, min_variance_target")

        w_opt = np.clip(res.x if res.success else x0, self.min_weight, self.max_weight)
        w_opt = w_opt / w_opt.sum() if w_opt.sum() > 0 else np.ones(n) / n

        self.weights = pd.Series(w_opt, index=self.tickers)
        print(f"  {(w_opt > 1e-6).sum()}/{n} non-zero weights")

    def compute_diagnostics(self):
        w, mu, cov = self.weights.values, self.mu_final.values, self.cov_matrix.values
        exp_return = float(w.dot(mu))
        vol = float(np.sqrt(max(0.0, w.dot(cov).dot(w))))
        self.diagnostics = {
            "expected_return": exp_return,
            "volatility": vol,
            "sharpe_ratio": float((exp_return - self.risk_free_rate) / max(vol, 1e-12)),
            "effective_n_assets": float(1.0 / np.sum(w ** 2)),
            "top3_concentration": float(np.sum(np.sort(w)[-3:])),
        }

    def run_optimization(self, market_caps=None):
        print("Running portfolio optimization")
        self.fetch_price_data()
        self.compute_historical_statistics()
        self.fetch_implied_volatilities()
        self.fetch_intrinsic_values()
        self.compute_blended_returns()
        if self.use_black_litterman:
            self.apply_black_litterman(market_caps=market_caps)
        self.optimize_portfolio()
        self.compute_diagnostics()
        print("Done\n")

    def export_results(self, filename="portfolio_results.json"):
        results = {
            "tickers": self.tickers,
            "optimization_date": datetime.now().isoformat(),
            "weights": self.weights.to_dict(),
            "expected_returns": self.mu_final.to_dict(),
            "implied_vols": self.implied_vols.to_dict(),
            "iv_skew": self.iv_skew.to_dict() if self.iv_skew is not None else {},
            "intrinsic_upsides": self.intrinsic_upsides.to_dict() if self.intrinsic_upsides is not None else {},
            "diagnostics": self.diagnostics,
            "bl_diagnostics": {k: v.to_dict() if isinstance(v, pd.Series) else v
                               for k, v in self.bl_diagnostics.items()} if self.bl_diagnostics else {},
        }
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {filename}")
        return results

    def display_results(self):
        print("Portfolio Allocation:")
        print(f"{'Ticker':<7} {'Type':<5} {'Weight':>8} {'ExpRet':>8} {'Vol':>7} {'IV':>7} {'Skew':>6} {'Intr':>7}")

        for ticker in sorted(self.tickers, key=lambda t: self.weights[t], reverse=True):
            t = "ETF" if ticker.upper() in KNOWN_ETFS else "STK"
            w = self.weights[ticker]
            er = self.mu_final[ticker]
            vol = self.sigma_hist[ticker]
            iv = self.implied_vols[ticker]
            skew = self.iv_skew[ticker] if self.iv_skew is not None else 0
            intr = self.intrinsic_upsides[ticker] if self.intrinsic_upsides is not None else 0
            intr_str = "-" if ticker.upper() in KNOWN_ETFS else f"{intr * 100:>+5.1f}%"

            print(f"{ticker:<7} {t:<5} {w:>7.1%} {er:>7.1%} {vol:>6.1%} {iv:>6.1%} {skew:>+5.2f} {intr_str:>7}")

        d = self.diagnostics
        print(f"\nExpected Return: {d['expected_return']:.2%}")
        print(f"Volatility: {d['volatility']:.2%}")
        print(f"Sharpe Ratio: {d['sharpe_ratio']:.2f}")

        if self.bl_diagnostics and 'error' not in self.bl_diagnostics:
            print(f"\nBlack-Litterman Diagnostics:")
            if self.bl_diagnostics.get('mode') == 'covariance_only':
                print(f"  Mode: Covariance adjustment only (returns preserved)")
                print(f"  Average shrinkage: {self.bl_diagnostics['avg_shrinkage']:.1%}")
                print(f"  Covariance correlation: {self.bl_diagnostics['cov_correlation']:.3f}")
                print(f"  Vol change: {self.bl_diagnostics['vol_old']:.1%} → {self.bl_diagnostics['vol_new']:.1%}")
            else:
                print(f"  Risk aversion (delta): {self.bl_diagnostics.get('delta', 'N/A')}")
                print(f"  BL weight: {self.bl_diagnostics.get('bl_weight', 'N/A'):.1%}")
                print(f"  View correlation: {self.bl_diagnostics.get('view_correlation', 'N/A'):.2f}")


if __name__ == "__main__":
    tickers = [
        "EXC", "UNH", "META", "FCX", "VZ", "QQQ", "IWM", "VTI",
        "VXUS", "TLT", "IEF", "SHY", "AGG", "LQD", "ARKK", "IPO", "HIMS"
    ]

    optimizer = ImpliedVolatilityPortfolioOptimizer(
        tickers=tickers,
        lookback_years=3,
        risk_free_rate=0.04,
        iv_multiplier=0.7,
        blend_alpha=0.6,
        use_black_litterman=True,
        bl_tau=0.05,
        min_weight=0.0,
        max_weight=1.0,
        jitter=1e-8,
        target_return=0.13,
        optimize_method="max_sharpe",
        use_intrinsic_value=True,
    )

    optimizer.run_optimization()
    optimizer.display_results()
    optimizer.export_results("portfolio_results.json")