import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import yfinance as yf
import numpy as np

def intrinsic_value_calculator(ticker, terminal_growth=0.01, years=10, max_growth_cap=0.15):
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
            if net_income and shares_outstanding and float(shares_outstanding) > 1000000:#makes sure the shares outstanding isn't some bs because yfinance might return 1.2 billion as 1.2 or something like that
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

        #CAPM
        risk_free = 0.045
        market_return = 0.09
        try:
            beta = float(info.get("beta", 1.0) or 1.0)
        except Exception:
            beta = 1.0
        cost_of_equity = risk_free + beta * (market_return - risk_free)

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
        justified_pe = (1.0 - payout_ratio) / max(denom, 0.01)#stops the terminal value from exploding if cost of equity is close to the growth rate
        terminal_pe = float(np.clip(justified_pe, 5, 14))#Use 14x as cap because historically, most matured firms that's in the financials/insurance/reinsurance has a one of around 10-13 but I use 14 just to be conservative
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
        implied_roe = growth_rate / max(retention, 0.2)#reverses the standard sustainable growth rate equation to calculate for ROE
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

        return {"Intrinsic Value(estimate)": final_intrinsic}

    else: #If company is not in financials/insurance/reinsurance
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
        risk_free, market_return = 0.045, 0.09
        beta = stock.info.get("beta", 1.0) or 1.0
        try:
            beta = float(beta)
        except Exception:
            beta = 1.0
        discount_rate = risk_free + beta * (market_return - risk_free)

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
        denom = max(denom, 0.005) # makes it so that the terminal value doesn't explode
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

        return {"Intrinsic Value(estimate)": intrinsic_value}

# Goes through each ticker and runs the intrinsic value calc on it then return a sorted data frame with length of 50
def run_screen(tickers):
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")["Close"].iloc[-1]
            intrinsic = intrinsic_value_calculator(ticker)["Intrinsic Value(estimate)"]
            upside = ((intrinsic-price)/price)*100  #%upside of the stock
            results.append({"Ticker": ticker, "Price": price, "Intrinsic": intrinsic, "% Upside":upside})
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue
    df = pd.DataFrame(results)
    df = df.sort_values("% Upside", ascending=False).head(50)
    return df


# S&P 500 tickers
import urllib.request
import pandas as pd

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    with urllib.request.urlopen(req) as response:
        tables = pd.read_html(response.read())
    # Extract symbols as a list of strings
    sp500 = tables[0]["Symbol"].tolist()
    # Some symbols on wikipedia have dots in between them but yfinance must use dashes instead.
    sp500 = [s.replace(".", "-") for s in sp500]
    return sp500


sp500 = get_sp500_tickers()
top50 = run_screen(sp500)
print(top50)