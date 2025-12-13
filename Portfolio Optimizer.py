"""
Portfolio optimizer using historical returns, implied volatility, and intrinsic value.
"""

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
            intrinsic_weight=0.25,
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
        self.intrinsic_weight = intrinsic_weight

        self.prices = None
        self.returns = None
        self.mu_hist = None
        self.sigma_hist = None
        self.cov_matrix = None
        self.implied_vols = None
        self.intrinsic_upsides = None
        self.mu_final = None
        self.weights = None
        self.diagnostics = {}

    # Data fetching & cleaning
    def fetch_price_data(self):
        # Fetch adjusted close prices and compute returns

        print(f"Fetching {self.lookback_years}yr price data for {len(self.tickers)} tickers...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_years * 365)

        raw = yf.download(self.tickers, start=start_date, end=end_date, progress=False, threads=True)
        if raw.empty:
            raise ValueError("No price data returned from yfinance")

        if isinstance(raw.columns, pd.MultiIndex):
            if ("Adj Close" in raw.columns.levels[0]):
                prices = raw["Adj Close"].copy()
            else:
                if "Close" in raw.columns.levels[0]:
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
        prices = prices.sort_index()
        prices = prices.ffill().bfill()
        prices = prices.dropna(how="all")

        if prices.shape[0] < 10:
            raise ValueError("Too few price rows after cleaning.")

        self.prices = prices
        self.returns = prices.pct_change().dropna()
        print(f"  {len(self.returns)} days of returns")

    def compute_historical_statistics(self):
        # Annualize mean returns and volatilities, compute covariance matrix

        print("Computing historical stats")
        trading_days = 252.0
        self.mu_hist = self.returns.mean() * trading_days
        self.sigma_hist = self.returns.std() * np.sqrt(trading_days)

        try:
            lw = LedoitWolf()
            lw.fit(self.returns.values)
            cov_daily = lw.covariance_
            cov_ann = cov_daily * trading_days
            cov_ann += np.eye(len(self.tickers)) * max(self.jitter, 1e-12)
            self.cov_matrix = pd.DataFrame(cov_ann, index=self.tickers, columns=self.tickers)
        except Exception as e:
            print(f" LedoitWolf failed: {e}, using sample covariance")
            cov_ann = self.returns.cov().values * trading_days
            cov_ann += np.eye(len(self.tickers)) * max(self.jitter, 1e-8)
            self.cov_matrix = pd.DataFrame(cov_ann, index=self.tickers, columns=self.tickers)

        print(
            f" mu: [{self.mu_hist.min():.2%}, {self.mu_hist.max():.2%}], sigma: [{self.sigma_hist.min():.2%}, {self.sigma_hist.max():.2%}]")

    # Implied volatility extraction

    def implied_vol_from_chain(self, calls, puts, spot_price, return_skew=False):
        # Compute ATM mid implied vol given calls and puts tables. If return_skew=True, also returns the put-call IV skew.

        try:
            iv_by_type = {'calls': np.nan, 'puts': np.nan}

            for chain_type, chain in [('calls', calls), ('puts', puts)]:
                if chain is None or chain.empty:
                    continue
                chain = chain.copy()
                chain["distance"] = np.abs(chain["strike"] - spot_price)
                atm_idx = chain["distance"].idxmin()
                idxs = []
                pos = chain.index.get_loc(atm_idx)
                for k in (-1, 0, 1):
                    j = pos + k
                    if 0 <= j < len(chain):
                        idxs.append(chain.index[j])
                sample = chain.loc[idxs]
                ivs = sample["impliedVolatility"].dropna().values
                if len(ivs) > 0:
                    iv_median = np.median(ivs)
                    iv_by_type[chain_type] = iv_median

            # Calculate combined IV (average of calls and puts)
            valid_ivs = [v for v in iv_by_type.values() if np.isfinite(v)]
            if len(valid_ivs) == 0:
                if return_skew:
                    return np.nan, np.nan
                return np.nan

            final_iv = float(np.median(valid_ivs))
            final_iv = float(np.clip(final_iv, 0.02, 3.0))

            if return_skew:
                # Calculate skew: (call_IV - put_IV) / average_IV
                # Positive skew = bullish (calls more expensive)
                # Negative skew = bearish (puts more expensive)
                if np.isfinite(iv_by_type['calls']) and np.isfinite(iv_by_type['puts']):
                    avg_iv = (iv_by_type['calls'] + iv_by_type['puts']) / 2
                    if avg_iv > 0.001:
                        skew = (iv_by_type['calls'] - iv_by_type['puts']) / avg_iv
                        skew = float(np.clip(skew, -1.0, 1.0))  # Clip extreme skews
                    else:
                        skew = 0.0
                else:
                    skew = 0.0  # Can't calculate skew without both
                return final_iv, skew

            return final_iv
        except Exception:
            if return_skew:
                return np.nan, np.nan
            return np.nan

    def interpolate_to_30d(self, iv_by_days, skew_by_days=None):
        # Interpolate variance to 30 days. If skew_by_days provided, also interpolates skew to 30 days.

        if len(iv_by_days) == 0:
            if skew_by_days is not None:
                return np.nan, np.nan
            return np.nan

        days = np.array(sorted(iv_by_days.keys()))
        vols = np.array([iv_by_days[d] for d in days], dtype=float)

        mask = np.isfinite(vols)
        if mask.sum() == 0:
            if skew_by_days is not None:
                return np.nan, np.nan
            return np.nan
        days_clean = days[mask]
        vols_clean = vols[mask]

        vars_ = (vols_clean ** 2) * days_clean

        if len(days_clean) == 1:
            var_30 = vars_[0] * (30.0 / days_clean[0])
        else:
            coeffs = np.polyfit(days_clean, vars_, 1)
            var_30 = np.polyval(coeffs, 30.0)
            var_30 = max(var_30, 1e-8)

        vol_30 = np.sqrt(var_30 / 30.0)
        vol_30 = float(np.clip(vol_30, 0.02, 3.0))

        # Interpolate skew if provided
        if skew_by_days is not None:
            skew_values = np.array([skew_by_days.get(d, np.nan) for d in days_clean], dtype=float)
            skew_mask = np.isfinite(skew_values)

            if skew_mask.sum() == 0:
                skew_30 = 0.0
            elif skew_mask.sum() == 1:
                skew_30 = float(skew_values[skew_mask][0])
            else:
                # Simple linear interpolation for skew
                days_skew = days_clean[skew_mask]
                skews = skew_values[skew_mask]
                if len(days_skew) >= 2:
                    coeffs_skew = np.polyfit(days_skew, skews, 1)
                    skew_30 = np.polyval(coeffs_skew, 30.0)
                else:
                    skew_30 = float(np.mean(skews))
                skew_30 = float(np.clip(skew_30, -1.0, 1.0))

            return vol_30, skew_30

        return vol_30

    def fetch_implied_volatilities(self):
        # Retrieve an ATM implied volatility (interpolated to 30-day) for each asset. Also calculates put-call skew for directional signal.

        print("Fetching implied volatilities")
        iv_dict = {}
        skew_dict = {}
        today = datetime.utcnow()

        for ticker in self.tickers:
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period="5d")
                if hist.empty:
                    spot = float(self.prices[ticker].iloc[-1]) if (
                                self.prices is not None and ticker in self.prices.columns) else np.nan
                else:
                    spot = float(hist["Close"].iloc[-1])

                iv_by_days = {}
                skew_by_days = {}
                exp_dates = tk.options
                if not exp_dates:
                    iv_fallback = float(self.sigma_hist.get(ticker, np.nan))
                    iv_dict[ticker] = float(np.clip(iv_fallback if np.isfinite(iv_fallback) else 0.2, 0.02, 3.0))
                    skew_dict[ticker] = 0.0
                    print(f"  {ticker}: no options, using hist vol {iv_dict[ticker]:.1%}")
                    continue

                for ed in exp_dates[:8]:
                    try:
                        opt_chain = tk.option_chain(ed)
                        calls = opt_chain.calls
                        puts = opt_chain.puts
                        dt = pd.to_datetime(ed).to_pydatetime()
                        days = max(1, (dt - today).days)
                        iv, skew = self.implied_vol_from_chain(calls, puts, spot, return_skew=True)
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
                    iv30 = result
                    skew30 = 0.0

                if not np.isfinite(iv30):
                    iv30 = float(self.sigma_hist.get(ticker, np.nan))
                    iv30 = float(np.clip(iv30 if np.isfinite(iv30) else 0.2, 0.02, 3.0))
                    skew30 = 0.0
                    print(f"  {ticker}: IV failed, using hist vol {iv30:.1%}")
                else:
                    print(f"  {ticker}: IV={iv30:.1%}, skew={skew30:+.2f}")

                iv_dict[ticker] = iv30
                skew_dict[ticker] = skew30 if np.isfinite(skew30) else 0.0

            except Exception as e:
                iv_fb = float(self.sigma_hist.get(ticker, np.nan))
                iv_dict[ticker] = float(np.clip(iv_fb if np.isfinite(iv_fb) else 0.2, 0.02, 3.0))
                skew_dict[ticker] = 0.0
                print(f"  {ticker}: error ({e}), using hist vol {iv_dict[ticker]:.1%}")

        self.implied_vols = pd.Series(iv_dict)
        self.iv_skew = pd.Series(skew_dict)

    # -------------------------
    # NEW: Intrinsic Value Integration
    # -------------------------
    def intrinsic_value_calculator(self, ticker, terminal_growth=0.01, years=10, max_growth_cap=0.20):
        """
        Calculate intrinsic value upside for a given ticker.
        Returns the percentage upside (e.g., 25.0 for 25% undervalued).

        FIX: Added 'self' as first parameter to make this a proper instance method.
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        # If company is in Financials, Insurance, or Reinsurance, Intrinsic value has to be calculated differently because of their unique business model
        if ("Financial Services" in sector) or ("Insurance" in sector) or ("Reinsurance" in industry):
            # Price
            try:
                price = stock.history(period="1d")["Close"].iloc[-1]
            except Exception:
                price = None

            net_income = info.get("netIncomeToCommon") or 0.0
            shares_outstanding = info.get("sharesOutstanding") or None
            book_value_per_share = info.get("bookValue") or 0.0

            # Reported EPS
            eps_reported = info.get("trailingEps") or info.get("forwardEps")
            try:
                eps_reported = float(eps_reported) if eps_reported is not None else None
            except Exception:
                eps_reported = None

            # Derive EPS from totals only if shares_outstanding looks reasonable
            eps_from_totals = None
            try:
                if net_income and shares_outstanding and float(
                        shares_outstanding) > 1000000:  # makes sure the shares outstanding isn't some bs because yfinance might return 1.2 billion as 1.2 or something like that
                    eps_from_totals = float(net_income) / float(shares_outstanding)
            except Exception:
                eps_from_totals = None

            if eps_reported and np.isfinite(eps_reported) and eps_reported > 0:
                eps = float(eps_reported)
                eps_source = "reported_eps"
            elif eps_from_totals and np.isfinite(eps_from_totals) and eps_from_totals > 0:
                eps = float(eps_from_totals)
                eps_source = "net_income / shares_outstanding"
            else:
                raise ValueError(
                    f"{ticker}: Unable to determine reliable EPS (net_income={net_income}, shares_out={shares_outstanding}, trailingEps={eps_reported})")

            # Validation shares_outstanding
            try:
                shares_outstanding = float(shares_outstanding) if shares_outstanding else None
            except Exception:
                shares_outstanding = None

            # Growth & discount with restrictions to keep this model reasonable
            growth_rate = float(info.get("earningsGrowth", 0.05) or 0.05)
            growth_rate = float(np.clip(growth_rate, -0.1, max_growth_cap))
            growth_rate = max(growth_rate, 0.01)  # floor 1%

            # CAPM
            risk_free = 0.045
            market_return = 0.09
            try:
                beta = float(info.get("beta", 1.0) or 1.0)
            except Exception:
                beta = 1.0
            cost_of_equity = risk_free + beta * (market_return - risk_free)
            cost_of_equity = np.clip(cost_of_equity, 0.08, 0.10)  # restricts it to industry standard

            # Project EPS (per-share)
            projected_eps = []
            last_eps = float(eps)
            for t in range(1, years + 1):
                year_growth = growth_rate - (growth_rate - terminal_growth) * (t / years)
                last_eps = last_eps * (1.0 + year_growth)
                discounted_eps = last_eps / ((1.0 + cost_of_equity) ** t)
                projected_eps.append(discounted_eps)

            # Terminal P/E also with restrictions
            payout_ratio = 0.35
            denom = cost_of_equity - terminal_growth
            if denom <= 0.0001:
                denom = 0.0001
            justified_pe = (1.0 - payout_ratio) / max(denom,
                                                      0.01)  # stops the terminal value from exploding if cost of equity is close to the growth rate
            terminal_pe = float(np.clip(justified_pe, 5,
                                        14))  # Use 14x as cap because historically, most matured firms that's in the financials/insurance/reinsurance has a one of around 10-13 but I use 14 just to be conservative
            terminal_value = (last_eps * terminal_pe) / ((1.0 + cost_of_equity) ** years)
            intrinsic_per_share = sum(projected_eps) + terminal_value

            # Error handling of book value per share
            try:
                book_value_per_share = float(book_value_per_share) if book_value_per_share else 0.0
            except Exception:
                book_value_per_share = 0.0

            scaled = False
            scale_factor = None
            if price and book_value_per_share > 0 and price > 0:
                ratio = book_value_per_share / price
                if ratio > 50:  # only trigger on extreme mismatch just in case
                    possible_factor = int(round(ratio))
                    if 2 <= possible_factor <= 5000:
                        book_value_per_share = book_value_per_share / possible_factor
                        scaled = True
                        scale_factor = possible_factor

            # Sustainable ROE & book future
            retention = 1.0 - payout_ratio
            implied_roe = growth_rate / max(retention,
                                            0.2)  # reverses the standard sustainable growth rate equation to calculate for ROE
            sustainable_roe = float(np.clip(implied_roe, 0.03, 0.12))
            book_value_future = book_value_per_share * ((1.0 + sustainable_roe * retention) ** years)

            # More conservative weights(10%) compared to the short term + stronger cap(*3) vs intrinsic per share because its over 10 years instead of 2
            if intrinsic_per_share > 0:
                book_component = 0.1 * book_value_future
            else:
                book_component = 0
            book_component = min(book_component, intrinsic_per_share * 3)

            # float uplift capped absolutely (no % of intrinsic runaway)
            float_uplift = min(0.05 * intrinsic_per_share, 0.5 * book_value_per_share)
            # Combine everything together
            final_intrinsic = 0.60 * intrinsic_per_share + book_component + float_uplift

            return (100 * (final_intrinsic - price) / price)

        else:  # If company is not in financials/insurance/reinsurance
            # Free Cash Flow (average of 2 years)
            def get_free_cash_flow(stock):
                cashflow = stock.cashflow
                if not cashflow.empty:
                    op_cf = None
                    for c in ["Total Cash From Operating Activities", "Operating Cash Flow", "Operating Cashflow"]:
                        if c in cashflow.index:
                            s = cashflow.loc[c].dropna()
                            if len(s) >= 2:  # average of 2 yrs
                                op_cf = s.iloc[0:2].mean()
                                break
                            elif len(s) > 0:
                                op_cf = s.iloc[0]
                                break
                    capex = 0
                    for c in ["Capital Expenditures", "Capex"]:
                        if c in cashflow.index:
                            s = cashflow.loc[c].dropna()
                            if len(s) >= 2:
                                capex = s.iloc[0:2].mean()
                                break
                            elif len(s) > 0:
                                capex = s.iloc[0]
                                break
                    if op_cf is not None:
                        return float(op_cf) - float(capex or 0)

                # Fallbacks
                fcf = stock.info.get("freeCashflow", None)
                if fcf:
                    try:
                        return float(fcf)
                    except Exception:
                        pass

                op_cf = stock.info.get("operatingCashflow", None)
                capex = stock.info.get("capitalExpenditures", 0)
                if op_cf:
                    try:
                        return float(op_cf) - float(capex or 0)
                    except Exception:
                        pass

                net_income = stock.info.get("netIncomeToCommon", None)
                if net_income:
                    try:
                        return float(net_income) * 0.8
                    except Exception:
                        pass
                return None

            fcf = get_free_cash_flow(stock)

            # Shares & Net Debt
            shares_outstanding = stock.info.get("sharesOutstanding", None)
            try:
                shares_outstanding = float(shares_outstanding) if shares_outstanding else None
            except Exception:
                shares_outstanding = None

            balance_sheet = stock.balance_sheet
            if balance_sheet.empty:
                total_debt = cash = 0.0
            else:
                total_debt = balance_sheet.loc["Total Debt"].iloc[0] if "Total Debt" in balance_sheet.index else 0.0
                cash = balance_sheet.loc["Cash"].iloc[0] if "Cash" in balance_sheet.index else 0.0
                total_debt = 0.0 if total_debt is None or np.isnan(total_debt) else total_debt
                cash = 0.0 if cash is None or np.isnan(cash) else cash

            net_debt = total_debt - cash

            # Growth rate (auto + analyst, capped at 50%)
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

            analyst_growth = stock.info.get("earningsGrowth", None)
            try:
                analyst_growth = float(analyst_growth) if analyst_growth else None
            except Exception:
                analyst_growth = None

            growth_rate = (auto_growth + analyst_growth) / 2.0 if analyst_growth and analyst_growth > 0 else auto_growth
            if not np.isfinite(growth_rate):
                growth_rate = 0.05
            growth_rate = max(min(growth_rate, max_growth_cap), -0.6)

            # Discount Rate (CAPM)

            risk_free, market_return, market_premium = 0.045, 0.09, 0.045
            market_cap = float(info.get("marketCap") or 0)
            beta = stock.info.get("beta", 1.0) or 1.0
            try:
                beta = float(beta)
            except Exception:
                beta = 1.0

            # Adjusted (Blume) beta
            adj_beta = 0.67 * beta + 0.33 * 1.0

            # Base CAPM
            capm = risk_free + adj_beta * market_premium

            # Sector-based rates
            if any(k in sector for k in ["Utility", "Defensive"]):
                base_low, base_high = 0.07, 0.08
            elif any(k in sector for k in ["Consumer Defensive", "Consumer Defensive"]):
                base_low, base_high = 0.075, 0.09
            elif any(k in sector for k in ["Health", "Industrial"]):
                base_low, base_high = 0.09, 0.11
            elif any(k in sector for k in ["Technology", "Semiconductor", "Software", "Communication"]):
                base_low, base_high = 0.095, 0.12
            elif any(k in sector for k in ["Energy", "Materials"]):
                base_low, base_high = 0.10, 0.14
            else:
                base_low, base_high = 0.09, 0.12

            # Adjust for market cap
            if market_cap > 200_000_000_000:
                cap_adj = -0.01
            elif market_cap < 2_000_000_000:
                cap_adj = 0.01
            else:
                cap_adj = 0.0

            # Blend everything: move within sector band based on beta
            position = np.clip((adj_beta - 0.8) / 0.8, 0, 1)  # normalize 0.8–1.6 beta range
            sector_anchor = base_low + position * (base_high - base_low)
            discount_rate = 0.6 * capm + 0.4 * sector_anchor
            discount_rate = discount_rate + cap_adj

            # Keep result in a reasonable corridor
            discount_rate = float(np.clip(discount_rate, 0.05, 0.16))

            # Checks
            if not all([fcf and np.isfinite(fcf), shares_outstanding and shares_outstanding > 0,
                        discount_rate and discount_rate > 0, np.isfinite(terminal_growth)]):
                raise ValueError("Invalid inputs for valuation")

            denom = discount_rate - terminal_growth
            if denom <= 0:
                raise ValueError("Discount rate must exceed terminal growth")

            # DCF Projection (linear growth decay)
            projected_fcfs = []
            last_fcf = float(fcf)
            start_growth, end_growth = growth_rate, terminal_growth + 0.01

            for t in range(1, years + 1):
                # Use exponential decay because it's on a 10 year time frame
                fade = np.exp(-t / (years / 2))
                year_growth = terminal_growth + (start_growth - terminal_growth) * fade
                last_fcf *= (1.0 + year_growth)
                discounted = last_fcf / ((1.0 + discount_rate) ** t)
                projected_fcfs.append(discounted)

            terminal_fcf = last_fcf * (1.0 + terminal_growth)
            denom = max(denom, 0.005)  # makes it so that the terminal value doesn't explode
            terminal_value = terminal_fcf / denom

            discounted_terminal_value = terminal_value / ((1.0 + discount_rate) ** years)
            discounted_terminal_value = min(discounted_terminal_value, sum(projected_fcfs) * 3)

            enterprise_value = sum(projected_fcfs) + discounted_terminal_value
            enterprise_value = np.clip(enterprise_value, 0, 50 * fcf)

            equity_value = enterprise_value - net_debt

            dcf_value = equity_value / shares_outstanding

            # Relative Valuation Calculations:

            # EPS and price
            eps = stock.info.get("forwardEps") or stock.info.get("trailingEps")
            try:
                eps = float(eps) if eps else None
            except Exception:
                eps = None

            try:
                price = stock.history(period="1d")["Close"].iloc[-1]
            except Exception:
                price = None

            # Multiples
            pe_val = None
            ev_ebitda_val = None
            p_fcf_val = None
            industry_pe_val = None
            hist_pe_val = None

            # P/E multiple
            forward_pe = stock.info.get("forwardPE")
            trailing_pe = stock.info.get("trailingPE")
            chosen_pe = forward_pe or trailing_pe
            if eps and chosen_pe:
                pe_val = chosen_pe * eps

            # EV/EBITDA
            ev_to_ebitda = stock.info.get("enterpriseToEbitda")
            if ev_to_ebitda and ev_to_ebitda > 0:
                ebitda = stock.financials.loc["EBITDA"].iloc[0] if "EBITDA" in stock.financials.index else None
                if ebitda and shares_outstanding > 0:
                    ev_ebitda_val = (ebitda * ev_to_ebitda - net_debt) / shares_outstanding

            # P/FCF
            if fcf and shares_outstanding > 0 and price > 0:
                p_fcf = (price * shares_outstanding) / fcf
                if p_fcf > 0:
                    p_fcf_val = (fcf * p_fcf) / shares_outstanding

            # Sector P/Es
            sector = stock.info.get("sector", "Unknown")
            sector_pe_map = {"Technology": 25, "Communication Services": 20, "Consumer Cyclical": 20,
                             "Consumer Defensive": 18, "Healthcare": 18, "Financial Services": 14,
                             "Industrials": 16, "Energy": 12, "Utilities": 10,
                             "Basic Materials": 14, "Real Estate": 15}
            sector_pe = sector_pe_map.get(sector, chosen_pe or 15)
            if eps and sector_pe:
                industry_pe_val = sector_pe * eps

            # Historical P/E
            hist = stock.history(period="5y")
            if not hist.empty and eps and eps > 0:
                hist["PE"] = hist["Close"] / eps
                hist_pe = hist["PE"].median()
                if hist_pe and np.isfinite(hist_pe):
                    hist_pe_val = hist_pe * eps

            # Average everything
            # Group 1: current multiples
            rel_current = [v for v in [pe_val, ev_ebitda_val, p_fcf_val] if v and np.isfinite(v)]
            # Group 2: Anchors(makes sure the relative value has doesn't explode)
            rel_anchors = [v for v in [industry_pe_val, hist_pe_val] if v and np.isfinite(v)]

            if rel_current or rel_anchors:
                current_val = np.median(rel_current) if rel_current else None
                anchor_val = np.median(rel_anchors) if rel_anchors else None

                if current_val and anchor_val:
                    relative_value = 0.6 * current_val + 0.4 * anchor_val
                elif current_val:
                    relative_value = current_val
                else:
                    relative_value = anchor_val
            else:
                relative_value = dcf_value

            # Weights(changed it to 50 50 because DCF is going to matter more in the long run)
            intrinsic_value = dcf_value * 0.5 + relative_value * 0.5

            return (100 * (intrinsic_value - price) / price)

    def fetch_intrinsic_values(self):
        # Calculate intrinsic value upside for each ticker

        if not self.use_intrinsic_value:
            return

        print("Calculating intrinsic values")
        upside_dict = {}

        known_etfs = {
            'ARKK', 'GRID', 'FAN', 'PAVE', 'TAN', 'PHO', 'PBW', 'IBB', 'DGRO', 'ESGU', 'ICLN', 'INDA', 'EWW', 'EWT', 'ITA', 'BBCA', 'XLU',
            'ESGV', 'VWO', 'VHT', 'VNQ', 'VTI', 'VT', 'VSS', 'VNQI', 'MSOS', 'IPO', 'JEPI', 'COWZ', 'LCTU', 'XLC', 'XLP', 'XLE', 'XLF', 'XLV',
            'XLI', 'RSP', 'ESGE', 'EWA', 'EWZ', 'MCHI', 'DSI', 'USMV', 'QUAL', 'ESGD', 'MOAT', 'VEU', 'VEA', 'VGK', 'VOO', 'VXUS', 'XLB', 'XLY',
            'QQQ', 'MTUM', 'IWF', 'XLK', 'SMH', 'VUG', 'VGT', 'IWD', 'NOBL', 'SCHD', 'VIG', 'VYM', 'VTV', 'IJH', 'VO', 'IJR', 'IWM', 'AVUV', 'AGG',
            'JPST', 'BSV', 'BND', 'BNDX', 'FTSL', 'HYLS', 'IGSB', 'FALN', 'HYG', 'LQD', 'JNK', 'VCIT', 'VCSH', 'SHY', 'TLT', 'IEF', 'EMB', 'SHV', 'GOVT',
            'BIL', 'VGSH', 'USFR', 'EMLC', 'VTIP', 'TIP', 'MBB', 'MUB', 'VTEB'
        }

        for ticker in self.tickers:
            try:
                if ticker.upper() in known_etfs:
                    print(f"  {ticker}: ETF, skipped")
                    upside_dict[ticker] = 0.0
                    continue

                try:
                    tk = yf.Ticker(ticker)
                    info = tk.info
                    quote_type = info.get('quoteType', '').upper()

                    if quote_type in ['ETF', 'MUTUALFUND', 'INDEX', 'CURRENCY', 'CRYPTOCURRENCY', 'FUTURE', 'OPTION']:
                        print(f"  {ticker}: {quote_type}, skipped because it's not a stock")
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
        # Three-way blend: historical, IV-based, and intrinsic value signals

        print("Blending expected returns")
        rf = self.risk_free_rate
        mu_hist = self.mu_hist.copy()

        # IV-based returns with directional skew adjustment
        skew_factor = 0.5
        if hasattr(self, 'iv_skew') and self.iv_skew is not None:
            direction_multiplier = (1.0 + skew_factor * self.iv_skew).clip(0.5, 1.5)
            mu_iv = rf + self.iv_multiplier * self.implied_vols * direction_multiplier
        else:
            mu_iv = rf + self.iv_multiplier * self.implied_vols

        if self.use_intrinsic_value and self.intrinsic_upsides is not None:
            realization_period = 3.0
            capture_rate = 0.5
            w_iv_fixed = 0.40
            w_remaining = 0.60

            mu_intrinsic = pd.Series(index=self.tickers, dtype=float)
            intrinsic_blend_weights = pd.Series(index=self.tickers, dtype=float)
            final_returns = pd.Series(index=self.tickers, dtype=float)
            per_ticker_weights = {}

            for ticker in self.tickers:
                hist_ret = mu_hist[ticker]
                iv_ret = mu_iv[ticker]
                upside = self.intrinsic_upsides[ticker]

                intrinsic_return_adjustment = upside * capture_rate / realization_period
                intrinsic_implied = rf + intrinsic_return_adjustment + 0.05

                abs_hist = abs(hist_ret)
                dynamic_intrinsic_pct = np.clip(0.10 + 0.60 * min(abs_hist / 0.50, 1.0), 0.10, 0.70) # Intrinsic value model isn't perfect and will sometimes output a very large number, so clip it to 70%
                intrinsic_blend_weights[ticker] = dynamic_intrinsic_pct

                mu_intrinsic[ticker] = (
                                                   1 - dynamic_intrinsic_pct) * hist_ret + dynamic_intrinsic_pct * intrinsic_implied

                w_hist_ticker = w_remaining * (1 - dynamic_intrinsic_pct)
                w_intrinsic_ticker = w_remaining * dynamic_intrinsic_pct

                per_ticker_weights[ticker] = {'w_hist': w_hist_ticker, 'w_iv': w_iv_fixed,
                                              'w_intrinsic': w_intrinsic_ticker}
                final_returns[
                    ticker] = w_hist_ticker * hist_ret + w_iv_fixed * iv_ret + w_intrinsic_ticker * intrinsic_implied

            self._intrinsic_blend_weights = intrinsic_blend_weights
            self._intrinsic_implied = rf + (self.intrinsic_upsides * capture_rate / realization_period) + 0.05
            self._per_ticker_weights = per_ticker_weights
            self._w_iv_fixed = w_iv_fixed
            self.mu_final = final_returns

            avg_w_hist = np.mean([w['w_hist'] for w in per_ticker_weights.values()])
            avg_w_intrinsic = np.mean([w['w_intrinsic'] for w in per_ticker_weights.values()])

            self._return_components = {
                'historical': mu_hist, 'iv_based': mu_iv, 'intrinsic': mu_intrinsic,
                'intrinsic_implied': self._intrinsic_implied,
                'weights': {'historical': avg_w_hist, 'iv_based': w_iv_fixed, 'intrinsic': avg_w_intrinsic}
            }

            self.print_compound_return_breakdown(mu_hist, mu_iv, mu_intrinsic, avg_w_hist, w_iv_fixed, avg_w_intrinsic)
        else:
            self.mu_final = self.blend_alpha * mu_hist + (1.0 - self.blend_alpha) * mu_iv
            self._return_components = {
                'historical': mu_hist, 'iv_based': mu_iv, 'intrinsic': None,
                'weights': {'historical': self.blend_alpha, 'iv_based': 1 - self.blend_alpha, 'intrinsic': 0}
            }

        self.mu_final = pd.Series(np.clip(self.mu_final.values, -0.5, 1.5), index=self.tickers)
        print(f"  Final expected returns: [{self.mu_final.min():.1%}, {self.mu_final.max():.1%}]")

    def print_compound_return_breakdown(self, mu_hist, mu_iv, mu_intrinsic, avg_w_hist, w_iv_fixed, avg_w_intrinsic):
        #Print return breakdown table

        # A list of the 100 tradable ETFs from the wharton rules website
        known_etfs = {
            'ARKK', 'GRID', 'FAN', 'PAVE', 'TAN', 'PHO', 'PBW', 'IBB', 'DGRO', 'ESGU', 'ICLN', 'INDA', 'EWW', 'EWT', 'ITA', 'BBCA', 'XLU',
            'ESGV', 'VWO', 'VHT', 'VNQ', 'VTI', 'VT', 'VSS', 'VNQI', 'MSOS', 'IPO', 'JEPI', 'COWZ', 'LCTU', 'XLC', 'XLP', 'XLE', 'XLF', 'XLV',
            'XLI', 'RSP', 'ESGE', 'EWA', 'EWZ', 'MCHI', 'DSI', 'USMV', 'QUAL', 'ESGD', 'MOAT', 'VEU', 'VEA', 'VGK', 'VOO', 'VXUS', 'XLB', 'XLY',
            'QQQ', 'MTUM', 'IWF', 'XLK', 'SMH', 'VUG', 'VGT', 'IWD', 'NOBL', 'SCHD', 'VIG', 'VYM', 'VTV', 'IJH', 'VO', 'IJR', 'IWM', 'AVUV', 'AGG',
            'JPST', 'BSV', 'BND', 'BNDX', 'FTSL', 'HYLS', 'IGSB', 'FALN', 'HYG', 'LQD', 'JNK', 'VCIT', 'VCSH', 'SHY', 'TLT', 'IEF', 'EMB', 'SHV', 'GOVT',
            'BIL', 'VGSH', 'USFR', 'EMLC', 'VTIP', 'TIP', 'MBB', 'MUB', 'VTEB'
        }

        print("\nReturn Breakdown:")
        print(f"{'Ticker':<7} {'Type':<5} {'Hist':>7} {'IV':>6} {'Skew':>6} {'IVRet':>7} {'Intr':>7} {'Final':>7}")

        for ticker in self.tickers:
            is_etf = ticker.upper() in known_etfs
            t = "ETF" if is_etf else "STK"

            hist_ret = mu_hist[ticker]
            iv_ret = mu_iv[ticker]
            iv_raw = self.implied_vols[ticker] if hasattr(self, 'implied_vols') else 0
            skew = self.iv_skew[ticker] if hasattr(self, 'iv_skew') else 0
            intr_implied = self._intrinsic_implied[ticker] if hasattr(self, '_intrinsic_implied') else 0

            if hasattr(self, '_per_ticker_weights') and ticker in self._per_ticker_weights:
                weights = self._per_ticker_weights[ticker]
                w_h, w_i, w_intr = weights['w_hist'], weights['w_iv'], weights['w_intrinsic']
            else:
                w_h, w_i, w_intr = avg_w_hist, w_iv_fixed, avg_w_intrinsic

            final_ret = w_h * hist_ret + w_i * iv_ret + w_intr * intr_implied
            intr_str = "-" if is_etf else f"{intr_implied:>6.1%}"

            print(
                f"{ticker:<7} {t:<5} {hist_ret:>6.1%} {iv_raw:>6.1%} {skew:>+5.2f} {iv_ret:>6.1%} {intr_str:>7} {final_ret:>6.1%}")

        avg_final = self.mu_final.mean()
        print(
            f"{'Avg':<7} {'':5} {mu_hist.mean():>6.1%} {self.implied_vols.mean():>6.1%} {self.iv_skew.mean():>+5.2f} {mu_iv.mean():>6.1%} {self._intrinsic_implied.mean():>7.1%} {avg_final:>6.1%}")
        print()

    def apply_black_litterman(self, market_caps=None):
        # Black-Litterman implementation
        if not self.use_black_litterman:
            return

        print("Applying Black-Litterman")
        n = len(self.tickers)
        Σ = self.cov_matrix.values
        tau = self.bl_tau

        if market_caps is not None:
            w_list = []
            for t in self.tickers:
                w_list.append(market_caps.get(t, 0.0))
            w_arr = np.array(w_list, dtype=float)
            if w_arr.sum() <= 0:
                print("Warning: market_caps sum non-positive(using equal weight instead)")
                w_mkt = np.ones(n) / n
            else:
                w_mkt = w_arr / w_arr.sum()
        else:
            print("Warning: no market caps supplied(using equal weight as proxy)")
            w_mkt = np.ones(n) / n

        μ_hist_arr = self.mu_hist.values
        mkt_return = w_mkt @ μ_hist_arr
        mkt_var = w_mkt @ Σ @ w_mkt
        if mkt_var <= 0:
            delta = 2.5
        else:
            delta = max(0.1, (mkt_return - self.risk_free_rate) / mkt_var)

        pi = delta * (Σ @ w_mkt)

        # Views: use the already-blended returns (which may include intrinsic value)
        Q = self.mu_final.values.reshape(-1, 1)
        P = np.eye(n)

        tau_sigma = tau * Σ
        Omega_diag = np.diag(P @ tau_sigma @ P.T).copy()
        Omega_diag = np.maximum(Omega_diag, 1e-8) * 1.0
        Omega = np.diag(Omega_diag)

        try:
            middle = P @ tau_sigma @ P.T + Omega
            inv_middle = np.linalg.inv(middle)
            adjustment = tau * Σ @ P.T @ inv_middle @ (Q - P @ pi.reshape(-1, 1))
            mu_bl = pi.reshape(-1, 1) + adjustment
            mu_bl = mu_bl.flatten()
            mu_final_arr = self.blend_alpha * μ_hist_arr + (1.0 - self.blend_alpha) * mu_bl
            self.mu_final = pd.Series(np.clip(mu_final_arr, -0.5, 1.5), index=self.tickers)
            print(f"Black-Litterman applied: μ [{self.mu_final.min():.3f}, {self.mu_final.max():.3f}]")
        except np.linalg.LinAlgError as e:
            print("Black-Litterman linear algebra error:", e)
            print("Falling back to blended returns without BL.")

    # Optimization (weights directly)
    def neg_sharpe(self, w, mu, cov, rf):
        w = np.array(w)
        ret = float(w.dot(mu))
        vol = float(np.sqrt(max(1e-12, w.dot(cov).dot(w))))
        return - (ret - rf) / vol

    def volatility(self, w, cov):
        w = np.array(w)
        return float(np.sqrt(max(0.0, w.dot(cov).dot(w))))

    def optimize_portfolio(self):
        # Optimize weights
        print("Optimizing portfolio")
        n = len(self.tickers)
        mu = self.mu_final.values
        cov = self.cov_matrix.values
        rf = self.risk_free_rate

        mu_excess = mu - rf
        if np.all(mu_excess <= 0):
            print("All excess returns <= 0, using min-variance instead")
            method = "min_variance_target"
        else:
            method = self.optimize_method

        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if method == "max_sharpe":
            x0 = np.ones(n) / n
            res = minimize(fun=self.neg_sharpe, x0=x0, args=(mu, cov, rf), method="SLSQP",
                           bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
            w_opt = res.x if res.success else np.ones(n) / n

        elif method == "min_variance_target":
            if self.target_return is None:
                raise ValueError("target_return required for min_variance_target")
            cons2 = cons + [{"type": "ineq", "fun": lambda w: w.dot(mu) - self.target_return}]
            x0 = np.ones(n) / n
            res = minimize(fun=lambda w: self.volatility(w, cov), x0=x0, method="SLSQP",
                           bounds=bounds, constraints=cons2, options={"maxiter": 1000, "ftol": 1e-10})
            w_opt = res.x if res.success else np.ones(n) / n
        else:
            raise ValueError("Unknown method: " + method)

        w_opt = np.clip(np.array(w_opt, dtype=float), self.min_weight, self.max_weight)
        w_opt = w_opt / w_opt.sum() if w_opt.sum() > 0 else np.ones(n) / n

        self.weights = pd.Series(w_opt, index=self.tickers)
        print(f"  {(w_opt > 1e-6).sum()}/{n} non-zero weights")

    def compute_diagnostics(self):
        w = self.weights.values
        mu = self.mu_final.values
        cov = self.cov_matrix.values
        rf = self.risk_free_rate

        exp_return = float(w.dot(mu))
        volatility = float(np.sqrt(max(0.0, w.dot(cov).dot(w))))
        sharpe = float((exp_return - rf) / (volatility if volatility > 0 else 1e-12))

        self.diagnostics = {
            "expected_return": exp_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
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
            "iv_skew": self.iv_skew.to_dict() if hasattr(self, 'iv_skew') else {},
            "intrinsic_upsides": self.intrinsic_upsides.to_dict() if self.intrinsic_upsides is not None else {},
            "diagnostics": self.diagnostics,
        }
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {filename}")
        return results

    def display_results(self):
        known_etfs = {
            'ARKK', 'GRID', 'FAN', 'PAVE', 'TAN', 'PHO', 'PBW', 'IBB', 'DGRO', 'ESGU', 'ICLN', 'INDA', 'EWW', 'EWT', 'ITA', 'BBCA', 'XLU',
            'ESGV', 'VWO', 'VHT', 'VNQ', 'VTI', 'VT', 'VSS', 'VNQI', 'MSOS', 'IPO', 'JEPI', 'COWZ', 'LCTU', 'XLC', 'XLP', 'XLE', 'XLF', 'XLV',
            'XLI', 'RSP', 'ESGE', 'EWA', 'EWZ', 'MCHI', 'DSI', 'USMV', 'QUAL', 'ESGD', 'MOAT', 'VEU', 'VEA', 'VGK', 'VOO', 'VXUS', 'XLB', 'XLY',
            'QQQ', 'MTUM', 'IWF', 'XLK', 'SMH', 'VUG', 'VGT', 'IWD', 'NOBL', 'SCHD', 'VIG', 'VYM', 'VTV', 'IJH', 'VO', 'IJR', 'IWM', 'AVUV', 'AGG',
            'JPST', 'BSV', 'BND', 'BNDX', 'FTSL', 'HYLS', 'IGSB', 'FALN', 'HYG', 'LQD', 'JNK', 'VCIT', 'VCSH', 'SHY', 'TLT', 'IEF', 'EMB', 'SHV', 'GOVT',
            'BIL', 'VGSH', 'USFR', 'EMLC', 'VTIP', 'TIP', 'MBB', 'MUB', 'VTEB'
        }

        print("Portfolio Allocation:")
        print(f"{'Ticker':<7} {'Type':<5} {'Weight':>8} {'ExpRet':>8} {'Vol':>7} {'IV':>7} {'Skew':>6} {'Intr':>7}")

        sorted_tickers = sorted(self.tickers, key=lambda t: self.weights[t], reverse=True)
        for ticker in sorted_tickers:
            t = "ETF" if ticker.upper() in known_etfs else "STK"
            w = self.weights[ticker]
            er = self.mu_final[ticker]
            vol = self.sigma_hist[ticker]
            iv = self.implied_vols[ticker]
            skew = self.iv_skew[ticker] if hasattr(self, 'iv_skew') else 0
            intr = self.intrinsic_upsides[ticker] if self.intrinsic_upsides is not None else 0
            intr_str = "-" if ticker.upper() in known_etfs else f"{intr * 100:>+5.1f}%"

            print(f"{ticker:<7} {t:<5} {w:>7.1%} {er:>7.1%} {vol:>6.1%} {iv:>6.1%} {skew:>+5.2f} {intr_str:>7}")

        d = self.diagnostics
        print(f"Expected Return: {d['expected_return']:.2%}")
        print(f"Volatility: {d['volatility']:.2%}")
        print(f"Sharpe Ratio: {d['sharpe_ratio']:.2f}")

# Main body
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
        use_black_litterman=False,
        bl_tau=0.05,
        min_weight=0.0,
        max_weight=1.0,
        jitter=1e-8,
        target_return=0.13,
        optimize_method="max_sharpe",
        use_intrinsic_value=True,
        intrinsic_weight=0.25,
    )

    optimizer.run_optimization()
    optimizer.display_results()
    optimizer.export_results("portfolio_results.json")