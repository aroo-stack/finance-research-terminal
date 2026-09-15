#!/usr/bin/env python3
"""
Multi-Agent Financial Research Terminal — Full Pipeline
==========================================================
Step 1: Shared ResearchState + Data Ingestion Agent
Step 2: Calculation Agent
Step 3: Synthesis Agent

Each agent reads from and writes to the shared ResearchState,
forming a clean pipeline of financial analysis.
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ANSI color codes for polished terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    BG_BLACK = "\033[40m"

    @staticmethod
    def clear():
        print("\033[H\033[J", end="")

import yfinance as yf

# ---------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ResearchTerminal")


# ---------------------------------------------------------------------------
# 1. SHARED DATA STRUCTURE — ResearchState
# ---------------------------------------------------------------------------
@dataclass
class ResearchState:
    """
    Shared data container passed between agents in the research pipeline.

    Fields
    ------
    ticker : str
        The stock ticker symbol under investigation.
    raw_financial_data : Dict[str, Any]
        Raw financial figures fetched from the data source (Agent 1).
    calculated_metrics : Dict[str, float]
        Derived metrics computed by the calculation agent (Agent 2).
    final_report : Dict[str, Any]
        The synthesized report produced by the synthesis agent (Agent 3).
    """

    ticker: str = ""
    raw_financial_data: Dict[str, Any] = field(default_factory=dict)
    calculated_metrics: Dict[str, float] = field(default_factory=dict)
    final_report: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ResearchState(ticker={self.ticker!r}, "
            f"raw_keys={list(self.raw_financial_data.keys())}, "
            f"metrics_count={len(self.calculated_metrics)}, "
            f"report_ready={bool(self.final_report)})"
        )


# ---------------------------------------------------------------------------
# 2. AGENT 1 — Data Ingestion Agent
# ---------------------------------------------------------------------------
class DataIngestionAgent:
    """
    Agent responsible for fetching raw financial data from Yahoo Finance.

    Populates ``raw_financial_data`` with:
      - Net Income
      - Operating Cash Flow
      - Capital Expenditures (CapEx)
      - Total Shares Outstanding
    """

    def __init__(self, state: ResearchState) -> None:
        self.state = state
        self.ticker_obj: Optional[yf.Ticker] = None

    def ingest(self, ticker: str) -> ResearchState:
        """
        Orchestrate the full data ingestion pipeline for a given ticker.

        Parameters
        ----------
        ticker : str
            The stock ticker symbol (e.g., "AAPL", "MSFT").

        Returns
        -------
        ResearchState
            The populated state object ready for downstream agents.
        """
        self.state.ticker = ticker.upper()
        logger.info("=" * 55)
        logger.info("AGENT 1 — DATA INGESTION AGENT")
        logger.info("=" * 55)
        logger.info("Step 1/4: Initialising ResearchState for ticker '%s'…", self.state.ticker)

        self._fetch_ticker_object()
        logger.info("Step 2/5: Fetching financial statements…")
        self._fetch_raw_financials()
        logger.info("Step 3/5: Fetching extended data…")
        self._fetch_extended_data()
        logger.info("Step 4/5: Extracting key metrics…")
        self._extract_key_metrics()
        logger.info("Step 5/5: Finalising ingestion — state updated.")
        logger.info("INGESTION COMPLETE — %s", self.state)
        logger.info("=" * 55)

        return self.state

    def _fetch_ticker_object(self) -> None:
        """Create a yfinance Ticker object and log the action."""
        self.ticker_obj = yf.Ticker(self.state.ticker)
        logger.info("  → yf.Ticker('%s') initialised.", self.state.ticker)

    def _fetch_raw_financials(self) -> None:
        """
        Pull annual income statement, cash-flow statement, and share info.
        All data is stored directly into ``self.state.raw_financial_data``.
        """
        # --- Income statement (for Net Income) ---
        logger.info("  → Fetching annual income statement…")
        income_stmt = self.ticker_obj.financials
        if income_stmt is not None and not income_stmt.empty:
            latest_year = income_stmt.columns[0]
            self.state.raw_financial_data["net_income"] = float(
                income_stmt.loc["Net Income", latest_year]
            )
            self.state.raw_financial_data["net_income_year"] = str(latest_year.year)
            logger.info(
                "     Net Income (%s): $%.2fM",
                latest_year.year,
                self.state.raw_financial_data["net_income"] / 1e6,
            )
        else:
            logger.warning("     Income statement unavailable for %s.", self.state.ticker)
            self.state.raw_financial_data["net_income"] = None

        # --- Cash-flow statement (for Operating Cash Flow & CapEx) ---
        logger.info("  → Fetching annual cash-flow statement…")
        cash_flow = self.ticker_obj.cashflow
        if cash_flow is not None and not cash_flow.empty:
            latest_year = cash_flow.columns[0]
            self.state.raw_financial_data["operating_cash_flow"] = float(
                cash_flow.loc["Operating Cash Flow", latest_year]
            )
            self.state.raw_financial_data["operating_cash_flow_year"] = str(latest_year.year)
            logger.info(
                "     Operating Cash Flow (%s): $%.2fM",
                latest_year.year,
                self.state.raw_financial_data["operating_cash_flow"] / 1e6,
            )

            capex_key = None
            for key in ["Capital Expenditure", "Capital Expenditures", "Capital Expend."]:
                if key in cash_flow.index:
                    capex_key = key
                    break
            if capex_key:
                self.state.raw_financial_data["capital_expenditures"] = float(
                    cash_flow.loc[capex_key, latest_year]
                )
                logger.info(
                    "     Capital Expenditures (%s): $%.2fM",
                    latest_year.year,
                    self.state.raw_financial_data["capital_expenditures"] / 1e6,
                )
            else:
                logger.warning("     CapEx line item not found in cash-flow statement.")
                self.state.raw_financial_data["capital_expenditures"] = None
        else:
            logger.warning("     Cash-flow statement unavailable for %s.", self.state.ticker)
            self.state.raw_financial_data["operating_cash_flow"] = None
            self.state.raw_financial_data["capital_expenditures"] = None

        # --- Share info (for Total Shares Outstanding) ---
        logger.info("  → Fetching share information…")
        info = self.ticker_obj.info
        if info:
            shares = info.get("sharesOutstanding")
            if shares:
                self.state.raw_financial_data["shares_outstanding"] = float(shares)
                logger.info("     Shares Outstanding: %.2fM", shares / 1e6)
            else:
                logger.warning("     sharesOutstanding not available in ticker info.")
                self.state.raw_financial_data["shares_outstanding"] = None
            self.state.raw_financial_data["company_name"] = info.get("shortName", "N/A")
            self.state.raw_financial_data["market_cap"] = info.get("marketCap")
            logger.info(
                "     Company: %s | Market Cap: $%.2fB",
                self.state.raw_financial_data["company_name"],
                info.get("marketCap", 0) / 1e9 if info.get("marketCap") else 0,
            )
        else:
            logger.warning("     Ticker info unavailable for %s.", self.state.ticker)
            self.state.raw_financial_data["shares_outstanding"] = None
            self.state.raw_financial_data["company_name"] = "N/A"

# ── NEW: Extended data fetching ──
    def _fetch_extended_data(self) -> None:
        """Fetch income statement details, balance sheet, and market context."""
        # --- Income Statement Details ---
        logger.info("  → Fetching extended income statement…")
        income_stmt = self.ticker_obj.financials
        if income_stmt is not None and not income_stmt.empty:
            latest_year = income_stmt.columns[0]
            for key in ["Total Revenue", "Gross Profit", "Operating Income",
                         "EBITDA", "Research Development", "Selling General and Administrative"]:
                if key in income_stmt.index:
                    self.state.raw_financial_data[key.lower().replace(" ", "_")] = float(
                        income_stmt.loc[key, latest_year]
                    )
            logger.info("  ✓ Income statement details extracted")

        # --- Balance Sheet ---
        logger.info("  → Fetching balance sheet…")
        balance_sheet = self.ticker_obj.balance_sheet
        if balance_sheet is not None and not balance_sheet.empty:
            latest_year = balance_sheet.columns[0]
            for key in ["Total Debt", "Stockholders Equity", "Total Assets",
                         "Total Liabilities Net Minority Interest", "Cash And Cash Equivalents"]:
                if key in balance_sheet.index:
                    self.state.raw_financial_data[key.lower().replace(" ", "_")] = float(
                        balance_sheet.loc[key, latest_year]
                    )
            logger.info("  ✓ Balance sheet extracted")

        # --- Market Context from Info ---
        logger.info("  → Fetching market context…")
        info = self.ticker_obj.info
        if info:
            market_fields = {
                "beta": "beta",
                "52week_high": "fiftyTwoWeekHigh",
                "52week_low": "fiftyTwoWeekLow",
                "target_mean": "targetMeanPrice",
                "current_price": "currentPrice",
                "trailing_pe": "trailingPE",
                "forward_pe": "forwardPE",
                "price_to_sales": "priceToSalesTrailing12Months",
                "price_to_book": "priceToBook",
                "enterprise_value": "enterpriseValue",
                "ebitda": "ebitda",
                "dividend_yield": "dividendYield",
                "profit_margins": "profitMargins",
                "operating_margins": "operatingMargins",
                "return_on_equity": "returnOnEquity",
                "return_on_assets": "returnOnAssets",
                "total_revenue": "totalRevenue",
                "revenue_growth": "revenueGrowth",
                "earnings_growth": "earningsGrowth",
                "free_cashflow": "freeCashflow",
                "operating_cashflow": "operatingCashflow",
            }
            for key, info_key in market_fields.items():
                val = info.get(info_key)
                if val is not None:
                    self.state.raw_financial_data[key] = float(val)
            logger.info("  ✓ Market context extracted")

        # --- Multi-year FCF Trend ---
        logger.info("  → Fetching multi-year FCF trend…")
        cash_flow = self.ticker_obj.cashflow
        if cash_flow is not None and not cash_flow.empty:
            fcf_history = {}
            for col_idx, col in enumerate(cash_flow.columns[:5]):
                year_str = str(col.year)
                ocf = cash_flow.loc["Operating Cash Flow", col] if "Operating Cash Flow" in cash_flow.index else None
                capex_key = None
                for k in ["Capital Expenditure", "Capital Expenditures", "Capital Expend."]:
                    if k in cash_flow.index:
                        capex_key = k
                        break
                capex = cash_flow.loc[capex_key, col] if capex_key else None
                if ocf is not None and capex is not None:
                    fcf_history[year_str] = float(ocf + capex)
            if fcf_history:
                self.state.raw_financial_data["fcf_history"] = fcf_history
                logger.info("  ✓ FCF trend (%d years) extracted", len(fcf_history))

    def _extract_key_metrics(self) -> None:
        """Log a summary of all extracted raw metrics."""
        logger.info("  → Summary of extracted raw financial data:")
        for key, value in self.state.raw_financial_data.items():
            if isinstance(value, float) and abs(value) > 1e6:
                logger.info("     %s: $%.2fB", key, value / 1e9)
            elif isinstance(value, float):
                logger.info("     %s: %.2f", key, value)
            elif isinstance(value, dict):
                logger.info("     %s: %d years of data", key, len(value))
            else:
                logger.info("     %s: %s", key, value)


# ---------------------------------------------------------------------------
# 4. NEW CALCULATION METHODS
# ---------------------------------------------------------------------------
# 3. AGENT 2 — Calculation Agent
# ---------------------------------------------------------------------------
class CalculationAgent:
    """
    Agent responsible for deriving key financial metrics from raw data.

    Computes the following metrics and stores them in ``calculated_metrics``:
      - Free Cash Flow (FCF)
      - FCF Margin
      - FCF Yield
      - Earnings Per Share (EPS)
      - Return on Equity proxy (FCF / Market Cap)
    """

    def __init__(self, state: ResearchState) -> None:
        self.state = state

    def calculate(self) -> ResearchState:
        """
        Run the full calculation pipeline on the ingested raw data.

        Returns
        -------
        ResearchState
            The state object now containing ``calculated_metrics``.
        """
        logger.info("=" * 55)
        logger.info("AGENT 2 — CALCULATION AGENT")
        logger.info("=" * 55)
        logger.info("Step 1/3: Validating raw data availability…")

        if not self.state.raw_financial_data:
            logger.error("No raw financial data to calculate from. Run Agent 1 first.")
            return self.state

        self._validate_inputs()
        logger.info("Step 2/3: Computing derived metrics…")
        self._compute_metrics()
        logger.info("Step 3/3: Finalising calculations.")
        logger.info("CALCULATION COMPLETE — %s", self.state)
        logger.info("=" * 55)

        return self.state

    def _validate_inputs(self) -> None:
        """Check that all required raw data fields are present and non-null."""
        required = ["net_income", "operating_cash_flow", "capital_expenditures", "shares_outstanding"]
        for field in required:
            value = self.state.raw_financial_data.get(field)
            if value is None:
                logger.warning("  ⚠ Required field '%s' is missing or None.", field)
            else:
                logger.info("  ✓ Field '%s' validated.", field)

    def _compute_metrics(self) -> None:
        """Derive financial metrics and store them in ``calculated_metrics``."""
        raw = self.state.raw_financial_data

        net_income = raw.get("net_income")
        op_cash_flow = raw.get("operating_cash_flow")
        capex = raw.get("capital_expenditures")
        shares = raw.get("shares_outstanding")
        market_cap = raw.get("market_cap")
        company_name = raw.get("company_name", "Unknown")

        # --- Free Cash Flow = Operating Cash Flow + CapEx (CapEx is negative) ---
        if op_cash_flow is not None and capex is not None:
            fcf = op_cash_flow + capex  # capex is already negative
            self.state.calculated_metrics["free_cash_flow"] = fcf
            logger.info("  → Free Cash Flow: $%.2fB", fcf / 1e9)

            # --- FCF Margin = FCF / Revenue proxy (Net Income * 10 for rough revenue) ---
            # Using Net Income * 10 as a rough revenue estimate when revenue is unavailable
            revenue_estimate = net_income * 10 if net_income else None
            if revenue_estimate and revenue_estimate != 0:
                fcf_margin = fcf / revenue_estimate
                self.state.calculated_metrics["fcf_margin"] = fcf_margin
                logger.info("  → FCF Margin (est.): %.2f%%", fcf_margin * 100)

            # --- FCF Yield = FCF / Market Cap ---
            if market_cap and market_cap != 0:
                fcf_yield = fcf / market_cap
                self.state.calculated_metrics["fcf_yield"] = fcf_yield
                logger.info("  → FCF Yield: %.2f%%", fcf_yield * 100)

        # --- Earnings Per Share = Net Income / Shares Outstanding ---
        if net_income is not None and shares is not None and shares != 0:
            eps = net_income / shares
            self.state.calculated_metrics["eps"] = eps
            logger.info("  → Earnings Per Share (EPS): $%.2f", eps)

        # --- FCF Per Share ---
        if "free_cash_flow" in self.state.calculated_metrics and shares is not None and shares != 0:
            fcf_per_share = self.state.calculated_metrics["free_cash_flow"] / shares
            self.state.calculated_metrics["fcf_per_share"] = fcf_per_share
            logger.info("  → FCF Per Share: $%.2f", fcf_per_share)

        # ── NEW: Extended Calculations ──
        self._compute_extended_metrics()

        logger.info("  → All calculated metrics stored for %s (%s).", company_name, self.state.ticker)

    def _compute_extended_metrics(self) -> None:
        """Compute additional financial ratios and valuation multiples."""
        raw = self.state.raw_financial_data
        calc = self.state.calculated_metrics
        shares = raw.get("shares_outstanding")
        current_price = raw.get("current_price", 0)
        market_cap = raw.get("market_cap")
        net_income = raw.get("net_income")
        op_cash = raw.get("operating_cash_flow")

        # --- Revenue & Margins ---
        revenue = raw.get("total_revenue") or raw.get("total_revenue")
        gross_profit = raw.get("gross_profit")
        operating_income = raw.get("operating_income")
        ebitda = raw.get("ebitda")
        profit_margins = raw.get("profit_margins")
        operating_margins = raw.get("operating_margins")

        if revenue and revenue > 0:
            calc["revenue"] = float(revenue)
            logger.info("  → Revenue: $%.2fB", float(revenue) / 1e9)
            if gross_profit and gross_profit > 0:
                calc["gross_margin"] = float(gross_profit) / float(revenue)
                logger.info("  → Gross Margin: %.2f%%", calc["gross_margin"] * 100)
            if operating_income and operating_income > 0:
                calc["operating_margin"] = float(operating_income) / float(revenue)
                logger.info("  → Operating Margin: %.2f%%", calc["operating_margin"] * 100)

        if profit_margins is not None:
            calc["profit_margin"] = float(profit_margins)
            logger.info("  → Profit Margin: %.2f%%", calc["profit_margin"] * 100)
        if operating_margins is not None:
            calc["operating_margin_yfinance"] = float(operating_margins)
            logger.info("  → Operating Margin (yfinance): %.2f%%", calc["operating_margin_yfinance"] * 100)

        # --- Balance Sheet Ratios ---
        total_debt = raw.get("total_debt")
        total_equity = raw.get("stockholders_equity")
        total_assets = raw.get("total_assets")
        total_liabilities = raw.get("total_liabilities_net_minority_interest")
        cash = raw.get("cash_and_cash_equivalents")

        if total_debt is not None and total_equity and total_equity != 0:
            de_ratio = float(total_debt) / float(total_equity)
            calc["debt_to_equity"] = de_ratio
            logger.info("  → Debt-to-Equity: %.2f", de_ratio)
        if total_debt is not None and total_assets and total_assets != 0:
            calc["debt_to_assets"] = float(total_debt) / float(total_assets)
            logger.info("  → Debt-to-Assets: %.2f", calc["debt_to_assets"])
        if cash is not None and total_debt is not None:
            net_debt = float(total_debt) - float(cash)
            calc["net_debt"] = net_debt
            logger.info("  → Net Debt: $%.2fB", net_debt / 1e9)

        # --- Profitability Ratios ---
        if net_income and total_equity and total_equity != 0 and net_income > 0:
            roe = float(net_income) / float(total_equity)
            calc["roe"] = roe
            logger.info("  → ROE: %.2f%%", roe * 100)
        if net_income and total_assets and total_assets != 0 and net_income > 0:
            roa = float(net_income) / float(total_assets)
            calc["roa"] = roa
            logger.info("  → ROA: %.2f%%", roa * 100)

        # --- Valuation Multiples ---
        if current_price > 0 and net_income and shares and shares > 0:
            eps_calc = float(net_income) / float(shares)
            if eps_calc != 0:
                pe = current_price / eps_calc
                calc["pe_ratio"] = pe
                logger.info("  → P/E Ratio: %.2f", pe)

        trailing_pe = raw.get("trailing_pe")
        if trailing_pe is not None:
            calc["pe_trailing"] = float(trailing_pe)
            logger.info("  → P/E (trailing): %.2f", float(trailing_pe))

        forward_pe = raw.get("forward_pe")
        if forward_pe is not None:
            calc["pe_forward"] = float(forward_pe)
            logger.info("  → P/E (forward): %.2f", float(forward_pe))

        if current_price > 0 and revenue and revenue > 0 and shares and shares > 0:
            revenue_per_share = float(revenue) / float(shares)
            if revenue_per_share != 0:
                ps = current_price / revenue_per_share
                calc["ps_ratio"] = ps
                logger.info("  → P/S Ratio: %.2f", ps)

        ev = raw.get("enterprise_value")
        if ev and ev > 0 and ebitda and ebitda > 0:
            calc["ev_ebitda"] = float(ev) / float(ebitda)
            logger.info("  → EV/EBITDA: %.2f", calc["ev_ebitda"])

        pb = raw.get("price_to_book")
        if pb is not None:
            calc["pb_ratio"] = float(pb)
            logger.info("  → P/B Ratio: %.2f", float(pb))

        # --- Growth ---
        revenue_growth = raw.get("revenue_growth")
        earnings_growth = raw.get("earnings_growth")
        if revenue_growth is not None:
            calc["revenue_growth_yfinance"] = float(revenue_growth) * 100
            logger.info("  → Revenue Growth: %.2f%%", calc["revenue_growth_yfinance"])
        if earnings_growth is not None:
            calc["earnings_growth_yfinance"] = float(earnings_growth) * 100
            logger.info("  → Earnings Growth: %.2f%%", calc["earnings_growth_yfinance"])

        # --- Market Context ---
        beta = raw.get("beta")
        if beta is not None:
            calc["beta"] = float(beta)
            logger.info("  → Beta: %.2f", float(beta))

        target_mean = raw.get("target_mean")
        if target_mean is not None and current_price > 0:
            upside = ((float(target_mean) - current_price) / current_price) * 100
            calc["analyst_upside"] = upside
            logger.info("  → Analyst Target Upside: %.2f%%", upside)

        # --- Dividend ---
        div_yield = raw.get("dividend_yield")
        if div_yield is not None:
            calc["dividend_yield"] = float(div_yield) * 100
            logger.info("  → Dividend Yield: %.2f%%", calc["dividend_yield"])

        # --- FCF Trend ---
        fcf_history = raw.get("fcf_history")
        if fcf_history and len(fcf_history) >= 2:
            years = sorted(fcf_history.keys())
            if len(years) >= 2:
                latest_fcf = fcf_history[years[-1]]
                prev_fcf = fcf_history[years[-2]]
                if prev_fcf != 0 and prev_fcf > 0:
                    fcf_growth = ((latest_fcf - prev_fcf) / abs(prev_fcf)) * 100
                    calc["fcf_growth_yoy"] = fcf_growth
                    logger.info("  → FCF YoY Growth: %.2f%%", fcf_growth)


# ---------------------------------------------------------------------------
# 5. HARDENED BUYBACK DCF FUNCTION
# ---------------------------------------------------------------------------
def run_hardened_buyback_dcf(research_state, discount_rate=0.10, terminal_growth=0.02, projection_years=5, annual_buyback_rate=0.025):
    """
    Executes a hardened DCF model that factorizes a 2.5% annual reduction
    in outstanding shares to mimic real-world capital allocation strategies.

    The compounding buyback shrinks the denominator over time, increasing
    per-share intrinsic value relative to a static-share model.

    Parameters
    ----------
    research_state : ResearchState
        The shared state object containing raw financial data.
    discount_rate : float
        The rate used to discount future cash flows (default: 10%).
    terminal_growth : float
        The perpetual growth rate for terminal value (default: 2%).
    projection_years : int
        Number of years to explicitly project (default: 5).
    annual_buyback_rate : float
        Annual reduction rate in shares outstanding (default: 2.5%).

    Returns
    -------
    ResearchState
        The state object now containing ``dcf_scenarios`` in calculated_metrics.
    """
    fcf_base = research_state.raw_financial_data.get('operating_cash_flow', 0) + \
               research_state.raw_financial_data.get('capital_expenditures', 0)
    starting_shares = research_state.raw_financial_data.get('shares_outstanding', 1)

    scenarios = {
        "Conservative": 0.03,
        "Base Case": 0.07,
        "Aggressive": 0.12,
    }

    dcf_results = {}

    print(f"\n[⚡ CALCULATION AGENT] Running Hardened Model ({annual_buyback_rate*100}% Annual Buyback Factor)...")
    print(f" -> Base FCF: ${fcf_base / 1e9:.2f}B | Starting Shares: {starting_shares / 1e6:.2f}M")

    for name, growth_rate in scenarios.items():
        pv_future_cash_flows = 0
        current_fcf = fcf_base
        projected_shares = starting_shares

        # Project and discount while simultaneously reducing share pool
        for year in range(1, projection_years + 1):
            current_fcf *= (1 + growth_rate)
            projected_shares *= (1 - annual_buyback_rate)

            discount_factor = (1 + discount_rate) ** year
            pv_future_cash_flows += current_fcf / discount_factor

        # Calculate Terminal Value (Gordon Growth Model)
        terminal_fcf = current_fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)

        total_intrinsic_value = pv_future_cash_flows + pv_terminal_value

        # Divide by the optimized final share pool
        estimated_price_per_share = total_intrinsic_value / projected_shares

        dcf_results[name] = {
            "growth_rate": growth_rate,
            "estimated_stock_price": estimated_price_per_share,
        }

        print(f" -> [{name}] New Target: ${estimated_price_per_share:.2f} per share")

    # Save cleanly into the shared terminal memory state
    research_state.calculated_metrics['dcf_scenarios'] = dcf_results
    return research_state


# ---------------------------------------------------------------------------
# 6. AGENT 3 — Synthesis Agent
# ---------------------------------------------------------------------------
class SynthesisAgent:
    """
    Agent responsible for synthesizing raw data and calculated metrics
    into a human-readable final report.

    Produces ``final_report`` with a narrative summary, key takeaways,
    and a quality assessment.
    """

    def __init__(self, state: ResearchState) -> None:
        self.state = state

    def synthesize(self) -> ResearchState:
        """
        Build the final report from raw data and calculated metrics.

        Returns
        -------
        ResearchState
            The state object now containing ``final_report``.
        """
        logger.info("=" * 55)
        logger.info("AGENT 3 — SYNTHESIS AGENT")
        logger.info("=" * 55)
        logger.info("Step 1/3: Gathering all available data…")

        if not self.state.calculated_metrics:
            logger.warning("No calculated metrics found. Running Calculation Agent first.")
            calc_agent = CalculationAgent(self.state)
            calc_agent.calculate()

        self._build_summary()
        logger.info("Step 2/3: Generating key takeaways…")
        self._generate_takeaways()
        logger.info("Step 3/3: Compiling final report.")
        logger.info("SYNTHESIS COMPLETE — %s", self.state)
        logger.info("=" * 55)

        return self.state

    def _build_summary(self) -> None:
        """Create a structured summary dict and store it in final_report."""
        raw = self.state.raw_financial_data
        calc = self.state.calculated_metrics
        ticker = self.state.ticker
        company_name = raw.get("company_name", "Unknown")

        # Fetch current market price for pricing context
        try:
            ticker_obj = yf.Ticker(ticker)
            current_price = float(ticker_obj.info.get("currentPrice", 0))
        except Exception:
            current_price = 0.0

        # Read DCF scenarios if present
        dcf_scenarios = calc.get("dcf_scenarios", {})

        report = {
            "ticker": ticker,
            "company_name": company_name,
            "fiscal_year": raw.get("net_income_year", "N/A"),
            "current_price": current_price,
            "raw_data": {
                "Net Income": raw.get("net_income"),
                "Operating Cash Flow": raw.get("operating_cash_flow"),
                "Capital Expenditures": raw.get("capital_expenditures"),
                "Shares Outstanding": raw.get("shares_outstanding"),
                "Market Cap": raw.get("market_cap"),
            },
            "calculated_metrics": dict(calc),
            "dcf_scenarios": dcf_scenarios,
        }
        self.state.final_report = report
        logger.info("  ✓ Summary compiled for %s.", company_name)
        logger.info("  ✓ Current market price: $%.2f", current_price)
        if dcf_scenarios:
            logger.info("  ✓ DCF scenarios loaded: %s", list(dcf_scenarios.keys()))

    def _generate_takeaways(self) -> None:
        """Generate narrative takeaways and a quality score, then append to final_report."""
        raw = self.state.raw_financial_data
        calc = self.state.calculated_metrics
        ticker = self.state.ticker
        company_name = raw.get("company_name", "Unknown")

        takeaways = []
        quality_score = 0

        # --- FCF Health Check ---
        fcf = calc.get("free_cash_flow")
        if fcf is not None and fcf > 0:
            takeaways.append(f"✅ {company_name} generates positive Free Cash Flow (${fcf/1e9:.2f}B).")
            quality_score += 1
        elif fcf is not None and fcf < 0:
            takeaways.append(f"⚠️ {company_name} has negative Free Cash Flow (${fcf/1e9:.2f}B).")
        else:
            takeaways.append(f"❓ FCF data unavailable for {company_name}.")

        # --- Earnings Quality ---
        net_income = raw.get("net_income")
        op_cash_flow = raw.get("operating_cash_flow")
        if net_income is not None and op_cash_flow is not None and net_income != 0:
            accrual_ratio = (op_cash_flow - net_income) / abs(net_income)
            if accrual_ratio > 0:
                takeaways.append(f"✅ Operating Cash Flow exceeds Net Income by {accrual_ratio*100:.1f}% — strong earnings quality.")
                quality_score += 1
            else:
                takeaways.append(f"⚠️ Net Income exceeds Operating Cash Flow — potential accrual concerns.")

        # --- Capital Efficiency ---
        fcf_yield = calc.get("fcf_yield")
        if fcf_yield is not None:
            if fcf_yield > 0.05:
                takeaways.append(f"✅ FCF Yield is {fcf_yield*100:.2f}% — attractive valuation signal.")
                quality_score += 1
            elif fcf_yield > 0:
                takeaways.append(f"ℹ️ FCF Yield is {fcf_yield*100:.2f}% — modest but positive.")
            else:
                takeaways.append(f"❌ FCF Yield is negative or zero.")

        # --- EPS Check ---
        eps = calc.get("eps")
        if eps is not None:
            takeaways.append(f"ℹ️ Earnings Per Share (EPS): ${eps:.2f}")

        # --- CapEx Intensity ---
        capex = raw.get("capital_expenditures")
        op_cf = raw.get("operating_cash_flow")
        if capex is not None and op_cf is not None and op_cf != 0:
            capex_intensity = abs(capex) / op_cf
            if capex_intensity < 0.3:
                takeaways.append(f"✅ CapEx intensity is {capex_intensity*100:.1f}% — asset-light model.")
            else:
                takeaways.append(f"ℹ️ CapEx intensity is {capex_intensity*100:.1f}% — capital-heavy operations.")

        # --- Assemble takeaways ---
        self.state.final_report["takeaways"] = takeaways
        self.state.final_report["quality_score"] = f"{quality_score}/3"

        logger.info("  ✓ Generated %d key takeaways (Quality Score: %s).", len(takeaways), quality_score)

    def print_report(self) -> None:
        """Pretty-print the final report to the terminal, including DCF markdown table."""
        report = self.state.final_report
        if not report:
            logger.warning("No report to display. Run the full pipeline first.")
            return

        raw = report.get("raw_data", {})
        calc = report.get("calculated_metrics", {})
        takeaways = report.get("takeaways", [])
        quality = report.get("quality_score", "N/A")
        dcf_scenarios = report.get("dcf_scenarios", {})
        current_price = report.get("current_price", 0.0)
        ticker = report.get("ticker", "N/A")
        company_name = report.get("company_name", "Unknown")

        # --- Tracking variables for explicit state ---
        report_lines: list[str] = []
        table_rows_built = False
        dcf_included = bool(dcf_scenarios)

        # ── Header ──
        print("\n" + "=" * 65)
        print(f"  📊 FINAL RESEARCH REPORT — {ticker} ({company_name})")
        print("=" * 65)

        # ── Raw Financial Data ──
        print("\n  📈 RAW FINANCIAL DATA")
        print(f"  ├─ Net Income:           ${raw.get('Net Income', 0) / 1e9:.2f}B" if raw.get('Net Income') else "  ├─ Net Income:           N/A")
        print(f"  ├─ Operating Cash Flow:  ${raw.get('Operating Cash Flow', 0) / 1e9:.2f}B" if raw.get('Operating Cash Flow') else "  ├─ Operating Cash Flow:  N/A")
        print(f"  ├─ Capital Expenditures: ${raw.get('Capital Expenditures', 0) / 1e9:.2f}B" if raw.get('Capital Expenditures') else "  ├─ Capital Expenditures: N/A")
        print(f"  ├─ Shares Outstanding:   {raw.get('Shares Outstanding', 0) / 1e6:.2f}M" if raw.get('Shares Outstanding') else "  ├─ Shares Outstanding:   N/A")
        print(f"  └─ Market Cap:           ${raw.get('Market Cap', 0) / 1e9:.2f}B" if raw.get('Market Cap') else "  └─ Market Cap:           N/A")

        # ── Calculated Metrics ──
        print("\n  🧮 CALCULATED METRICS")
        for key, val in calc.items():
            if isinstance(val, dict):
                continue
            if key in ("fcf_margin", "fcf_yield"):
                print(f"  ├─ {key.replace('_', ' ').title():<25}: {val*100:.2f}%")
            elif key in ("eps", "fcf_per_share"):
                print(f"  ├─ {key.replace('_', ' ').title():<25}: ${val:.2f}")
            else:
                print(f"  ├─ {key.replace('_', ' ').title():<25}: ${val / 1e9:.2f}B")

        # ── DCF Multi-Scenario Table ──
        if dcf_included:
            print("\n  💰 MULTI-SCENARIO DCF — INTRINSIC VALUE PER SHARE")
            print("  ┌─────────────────┬──────────────┬──────────────────┬──────────────────┐")
            print("  │ Scenario        │ Growth Rate  │ Intrinsic Value  │ vs Current Price │")
            print("  ├─────────────────┼──────────────┼──────────────────┼──────────────────┤")

            for scenario_name, data in dcf_scenarios.items():
                growth_pct = f"{data['growth_rate']*100:.0f}%"
                intrinsic_val = f"${data['estimated_stock_price']:.2f}"

                if current_price > 0:
                    premium_discount = ((data['estimated_stock_price'] - current_price) / current_price) * 100
                    vs_current = f"{premium_discount:+.1f}%"
                else:
                    vs_current = "N/A"

                print(f"  │ {scenario_name:<15} │ {growth_pct:<12} │ {intrinsic_val:<16} │ {vs_current:<16} │")
                table_rows_built = True

            print("  └─────────────────┴──────────────┴──────────────────┴──────────────────┘")

            if current_price > 0:
                print(f"\n  📌 Current Market Price: ${current_price:.2f}")
                print(f"  📌 Raw FCF Base: ${calc.get('free_cash_flow', 0) / 1e9:.2f}B")
                print(f"  📌 Shares Outstanding: {raw.get('Shares Outstanding', 0) / 1e6:.2f}M")
                table_rows_built = True

        # ── Quality Score ──
        print(f"\n  🏆 QUALITY SCORE: {quality}")

        # ── Key Takeaways ──
        print("\n  💡 KEY TAKEAWAYS")
        for i, takeaway in enumerate(takeaways, 1):
            print(f"  {i}. {takeaway}")

        # ── Footer ──
        print("\n" + "=" * 65)
        print(f"  ✅ Report complete | DCF scenarios rendered: {dcf_included}")
        print(f"  ✅ Table rows built: {table_rows_built}")
        print("  ✅ Report generation complete.\n")


# ---------------------------------------------------------------------------
# GLOBAL STATE TRACKING
# ---------------------------------------------------------------------------
LAST_STATE: Optional[ResearchState] = None
LAST_COMPARISON: Optional[tuple] = None


# ---------------------------------------------------------------------------
# 5. CONTEXTUAL METRICS ANALYSIS
# ---------------------------------------------------------------------------
def display_contextual_analysis() -> None:
    """
    Analyze metrics from the most recent run.
    If a comparison was done, shows both tickers. Otherwise just the last one.
    """
    global LAST_STATE, LAST_COMPARISON

    # ── Check if comparison was last ──
    if LAST_COMPARISON is not None:
        state1, ticker1, state2, ticker2 = LAST_COMPARISON
        _display_comparison_analysis(state1, ticker1, state2, ticker2)
        return

    # ── Single ticker analysis ──
    if LAST_STATE is None or not LAST_STATE.calculated_metrics:
        print(f"\n  {Colors.RED}❌ No analysis data found. Run option 1 first!{Colors.RESET}")
        print(f"  {Colors.DIM}  Analyze a ticker before checking metrics.{Colors.RESET}")
        return

    _display_single_analysis(LAST_STATE)

    raw = LAST_STATE.raw_financial_data
    calc = LAST_STATE.calculated_metrics
    dcf = LAST_STATE.calculated_metrics.get('dcf_scenarios', {})
    report = LAST_STATE.final_report
    ticker = LAST_STATE.ticker
    company = raw.get("company_name", ticker)
    current_price = report.get("current_price", 0)

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
    print("  ═════════════════════════════════════════════════════")
    print(f"  📖  CONTEXTUAL METRICS ANALYSIS — {ticker} ({company})")
    print("  ═════════════════════════════════════════════════════")
    print(f"{Colors.RESET}")

    takeaways = []

    # ── FCF Health ──
    fcf = calc.get("free_cash_flow")
    if fcf is not None:
        if fcf > 0:
            takeaways.append(f"  ✅ FCF: ${fcf/1e9:.2f}B — company generates real cash")
        else:
            takeaways.append(f"  ❌ FCF: ${fcf/1e9:.2f}B — company is burning cash")

        # FCF growth potential
        if "Conservative" in dcf:
            conservative_price = dcf["Conservative"]["estimated_stock_price"]
            if current_price > 0:
                gap = ((conservative_price - current_price) / current_price) * 100
                takeaways.append(f"  📊 Conservative DCF target: ${conservative_price:.2f} ({gap:+.1f}% vs current ${current_price:.2f})")

    # ── Earnings Quality ──
    net_income = raw.get("net_income")
    op_cash = raw.get("operating_cash_flow")
    if net_income is not None and op_cash is not None and net_income != 0:
        accrual = (op_cash - net_income) / abs(net_income)
        if accrual > 0.2:
            takeaways.append(f"  ✅ Strong earnings quality: OCF exceeds NI by {accrual*100:.1f}%")
        elif accrual < -0.1:
            takeaways.append(f"  ⚠️ Weak earnings quality: NI exceeds OCF by {abs(accrual)*100:.1f}%")
        else:
            takeaways.append(f"  ℹ️ Earnings quality: OCF/NI gap is {accrual*100:.1f}%")

    # ── Capital Efficiency ──
    capex = raw.get("capital_expenditures")
    if capex is not None and op_cash is not None and op_cash != 0:
        intensity = abs(capex) / op_cash
        if intensity < 0.3:
            takeaways.append(f"  ✅ Asset-light: CapEx intensity {intensity*100:.1f}% (below 30%)")
        elif intensity < 0.5:
            takeaways.append(f"  ℹ️ Moderate CapEx: {intensity*100:.1f}% intensity")
        else:
            takeaways.append(f"  ⚠️ Capital-heavy: {intensity*100:.1f}% CapEx intensity — cyclical sensitivity")

    # ── DCF Scenarios ──
    if dcf:
        print(f"\n  📊 DCF SCENARIO BREAKDOWN:")
        print(f"  ┌─────────────────┬──────────────┬──────────────────┐")
        print(f"  │ Scenario        │ Growth Rate  │ Target Price     │")
        print(f"  ├─────────────────┼──────────────┼──────────────────┤")
        for name, data in dcf.items():
            price_str = f"${data['estimated_stock_price']:.2f}"
            if current_price > 0:
                gap = ((data['estimated_stock_price'] - current_price) / current_price) * 100
                price_str += f" ({gap:+.0f}%)"
            print(f"  │ {name:<15} │ {data['growth_rate']*100:>8.0f}%   │ {price_str:<16} │")
        print(f"  └─────────────────┴──────────────┴──────────────────┘")

        # Fair value assessment
        base_price = dcf.get("Base Case", {}).get("estimated_stock_price", 0)
        if current_price > 0 and base_price > 0:
            if base_price > current_price * 1.2:
                takeaways.append(f"  🟢 DCF suggests UNDERVALUED — base target ${base_price:.2f} vs ${current_price:.2f}")
            elif base_price > current_price * 0.8:
                takeaways.append(f"  🟡 DCF suggests FAIRLY VALUED — base target ${base_price:.2f} vs ${current_price:.2f}")
            else:
                takeaways.append(f"  🔴 DCF suggests OVERVALUED — base target ${base_price:.2f} vs ${current_price:.2f}")

    # ── 4. Valuation Multiples ──
    print(f"\n  💎 VALUATION MULTIPLES")
    pe = calc.get("pe_trailing") or calc.get("pe_ratio")
    pe_fwd = calc.get("pe_forward")
    ps = calc.get("ps_ratio")
    ev_ebitda = calc.get("ev_ebitda")
    pb = calc.get("pb_ratio")
    if pe is not None or pe_fwd is not None or ps is not None or ev_ebitda is not None or pb is not None:
        pe_str = f"{pe:.2f}" if pe else "N/A"
        pe_fwd_str = f"{pe_fwd:.2f}" if pe_fwd else "N/A"
        ps_str = f"{ps:.2f}" if ps else "N/A"
        ev_str = f"{ev_ebitda:.2f}" if ev_ebitda else "N/A"
        pb_str = f"{pb:.2f}" if pb else "N/A"
        print(f"     P/E (trailing): {pe_str} | P/E (forward): {pe_fwd_str}")
        print(f"     P/S: {ps_str} | EV/EBITDA: {ev_str} | P/B: {pb_str}")

    # ── 5. Profitability ──
    print(f"\n  📈 PROFITABILITY")
    gross_margin = calc.get("gross_margin")
    op_margin = calc.get("operating_margin") or calc.get("operating_margin_yfinance")
    profit_margin = calc.get("profit_margin")
    roe = calc.get("roe")
    roa = calc.get("roa")
    profit_margins_val = raw.get("profit_margins")
    if gross_margin is not None:
        print(f"     Gross Margin: {gross_margin*100:.1f}%")
    if op_margin is not None:
        print(f"     Operating Margin: {op_margin*100:.1f}%")
    if profit_margin is not None:
        print(f"     Profit Margin: {profit_margin*100:.1f}%")
    elif profit_margins_val is not None:
        print(f"     Profit Margin: {float(profit_margins_val)*100:.1f}%")
    if roe is not None:
        print(f"     ROE: {roe*100:.1f}%")
    if roa is not None:
        print(f"     ROA: {roa*100:.1f}%")

    # ── 6. Balance Sheet & Risk ──
    print(f"\n  🏦 BALANCE SHEET")
    de_ratio = calc.get("debt_to_equity")
    net_debt = calc.get("net_debt")
    total_assets = raw.get("total_assets")
    total_liabilities = raw.get("total_liabilities_net_minority_interest")
    cash = raw.get("cash_and_cash_equivalents")
    if de_ratio is not None:
        debt_label = "LOW" if de_ratio < 0.5 else "MODERATE" if de_ratio < 1.0 else "HIGH"
        print(f"     Debt-to-Equity: {de_ratio:.2f} [{debt_label}]")
    if net_debt is not None:
        print(f"     Net Debt: ${net_debt/1e9:.2f}B")
    if cash is not None:
        print(f"     Cash: ${float(cash)/1e9:.2f}B")
    if total_assets is not None:
        print(f"     Total Assets: ${float(total_assets)/1e9:.2f}B")
    if total_liabilities is not None:
        print(f"     Total Liabilities: ${float(total_liabilities)/1e9:.2f}B")

    # ── 7. Growth ──
    print(f"\n  📊 GROWTH")
    rev_growth = calc.get("revenue_growth_yfinance")
    earn_growth = calc.get("earnings_growth_yfinance")
    fcf_growth = calc.get("fcf_growth_yoy")
    if rev_growth is not None:
        print(f"     Revenue Growth: {rev_growth:.1f}%")
    if earn_growth is not None:
        print(f"     Earnings Growth: {earn_growth:.1f}%")
    if fcf_growth is not None:
        print(f"     FCF Growth (YoY): {fcf_growth:.1f}%")

    # ── 8. FCF History ──
    fcf_history = raw.get("fcf_history")
    if fcf_history and len(fcf_history) >= 2:
        print(f"\n  📈 FCF TREND (Multi-Year)")
        years = sorted(fcf_history.keys())
        trend_str = " | ".join(f"{y}: ${fcf_history[y]/1e9:.2f}B" for y in years)
        print(f"     {trend_str}")

    # ── 9. Market Context ──
    print(f"\n  🌍 MARKET CONTEXT")
    beta = calc.get("beta")
    target_mean = raw.get("target_mean")
    div_yield = calc.get("dividend_yield")
    if beta is not None:
        beta_label = "LOW" if abs(beta) < 0.8 else "MODERATE" if abs(beta) < 1.2 else "HIGH"
        print(f"     Beta: {beta:.2f} [{beta_label} volatility]")
    if target_mean is not None and current_price > 0:
        upside = ((float(target_mean) - current_price) / current_price) * 100
        print(f"     Analyst Target: ${float(target_mean):.2f} ({upside:+.1f}%)")
    if div_yield is not None:
        print(f"     Dividend Yield: {div_yield:.2f}%")

    # ── COMPREHENSIVE NARRATIVE SYNTHESIS ──
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║              📝  FULL INVESTMENT SYNTHESIS           ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    print(f"  {Colors.BOLD}📌 {Colors.WHITE}{ticker} ({company}) — Current: ${current_price:.2f}{Colors.RESET}")

    # 1. Cash Flow Assessment
    if fcf is not None:
        if fcf > 0 and (not op_cash or fcf > float(op_cash) * 0.3):
            print(f"  {Colors.GREEN}{Colors.BOLD}💰 Cash Engine:{Colors.RESET} {Colors.GREEN}Strong — FCF ${fcf/1e9:.2f}B generates real cash. Healthy business model.{Colors.RESET}")
        elif fcf and fcf > 0:
            print(f"  {Colors.YELLOW}{Colors.BOLD}💰 Cash Engine:{Colors.RESET} {Colors.YELLOW}Moderate — FCF ${fcf/1e9:.2f}B thin vs OCF. Heavy reinvestment needs.{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}💰 Cash Engine:{Colors.RESET} {Colors.RED}Weak — Negative FCF (${fcf/1e9:.2f}B). Burning cash.{Colors.RESET}")

    # 2. Valuation
    if dcf and current_price > 0:
        base_price = dcf.get("Base Case", {}).get("estimated_stock_price", 0)
        conservative_price = dcf.get("Conservative", {}).get("estimated_stock_price", 0)
        aggressive_price = dcf.get("Aggressive", {}).get("estimated_stock_price", 0)
        if base_price > 0:
            gap = ((base_price - current_price) / current_price) * 100
            tag, tag_msg = _get_valuation_tag(current_price, base_price)
            vc = Colors.GREEN if gap > 25 else (Colors.YELLOW if gap > 0 else Colors.RED)
            print(f"  {Colors.BOLD}🎯 Valuation:{Colors.RESET} {vc}{tag} — Base: ${base_price:.2f} ({gap:+.1f}%). Range: ${conservative_price:.2f}-${aggressive_price:.2f}.{Colors.RESET}")

    # 3. Earnings Quality
    if net_income is not None and op_cash is not None and net_income != 0:
        accrual = (op_cash - net_income) / abs(net_income)
        if accrual > 0.2:
            print(f"  {Colors.GREEN}{Colors.BOLD}📊 Earnings Quality:{Colors.RESET} {Colors.GREEN}Strong — OCF exceeds NI by {accrual*100:.0f}%. Cash-backed.{Colors.RESET}")
        elif accrual < -0.2:
            print(f"  {Colors.RED}{Colors.BOLD}📊 Earnings Quality:{Colors.RESET} {Colors.RED}Caution — NI exceeds OCF by {abs(accrual)*100:.0f}%. Watch accounting.{Colors.RESET}")
        else:
            print(f"  {Colors.YELLOW}{Colors.BOLD}📊 Earnings Quality:{Colors.RESET} {Colors.YELLOW}Moderate — OCF/NI {op_cash/net_income:.2f}x. Reasonably backed.{Colors.RESET}")

    # 4. Capital Efficiency
    if capex is not None and op_cash and op_cash != 0:
        intensity = abs(capex) / op_cash
        if intensity < 0.3:
            print(f"  {Colors.GREEN}{Colors.BOLD}⚙️ CapEx Efficiency:{Colors.RESET} {Colors.GREEN}Asset-light — {intensity*100:.1f}% CapEx. High cash return.{Colors.RESET}")
        elif intensity < 0.5:
            print(f"  {Colors.YELLOW}{Colors.BOLD}⚙️ CapEx Efficiency:{Colors.RESET} {Colors.YELLOW}Moderate — {intensity*100:.1f}% CapEx. Needs reinvestment.{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}⚙️ CapEx Efficiency:{Colors.RESET} {Colors.RED}Heavy — {intensity*100:.1f}% CapEx. Cyclical risk high.{Colors.RESET}")

    # 5. Growth
    if rev_growth is not None and earn_growth is not None:
        if rev_growth > 20 and earn_growth > 20:
            print(f"  {Colors.GREEN}{Colors.BOLD}📈 Growth Profile:{Colors.RESET} {Colors.GREEN}Explosive — Rev +{rev_growth:.1f}%, EPS +{earn_growth:.1f}%. Expansion.{Colors.RESET}")
        elif rev_growth > 10 and earn_growth > 10:
            print(f"  {Colors.GREEN}{Colors.BOLD}📈 Growth Profile:{Colors.RESET} {Colors.GREEN}Solid — Rev +{rev_growth:.1f}%, EPS +{earn_growth:.1f}%. Above avg.{Colors.RESET}")
        elif rev_growth > 0 and earn_growth > 0:
            print(f"  {Colors.YELLOW}{Colors.BOLD}📈 Growth Profile:{Colors.RESET} {Colors.YELLOW}Modest — Rev +{rev_growth:.1f}%, EPS +{earn_growth:.1f}%. May be priced in.{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}📈 Growth Profile:{Colors.RESET} {Colors.RED}Mixed/Negative — Headwinds.{Colors.RESET}")

    # 6. Balance Sheet
    if de_ratio is not None:
        if de_ratio < 0.3:
            print(f"  {Colors.GREEN}{Colors.BOLD}🏦 Balance Sheet:{Colors.RESET} {Colors.GREEN}Robust — D/E {de_ratio:.2f}. Minimal leverage.{Colors.RESET}")
        elif de_ratio < 0.8:
            print(f"  {Colors.YELLOW}{Colors.BOLD}🏦 Balance Sheet:{Colors.RESET} {Colors.YELLOW}Healthy — D/E {de_ratio:.2f}. Manageable.{Colors.RESET}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}🏦 Balance Sheet:{Colors.RESET} {Colors.RED}Leveraged — D/E {de_ratio:.2f}. Higher risk.{Colors.RESET}")

    # 7. Market Risk
    if beta is not None:
        if beta > 1.5:
            print(f"  {Colors.RED}{Colors.BOLD}🌍 Market Risk:{Colors.RESET} {Colors.RED}HIGH — Beta {beta:.2f} ({abs(beta-1):.1f}x market vol). Aggressive only.{Colors.RESET}")
        elif beta > 1.0:
            print(f"  {Colors.YELLOW}{Colors.BOLD}🌍 Market Risk:{Colors.RESET} {Colors.YELLOW}MODERATE — Beta {beta:.2f}. Standard risk.{Colors.RESET}")
        else:
            print(f"  {Colors.GREEN}{Colors.BOLD}🌍 Market Risk:{Colors.RESET} {Colors.GREEN}LOW — Beta {beta:.2f}. Defensive.{Colors.RESET}")

    # 8. Income
    if div_yield is not None and float(div_yield) > 0:
        print(f"  {Colors.CYAN}{Colors.BOLD}💵 Income:{Colors.RESET} {Colors.CYAN}Dividend yield {float(div_yield):.2f}%. Income component.{Colors.RESET}")

    # 9. Overall Thesis
    score_components = []
    if fcf and fcf > 0:
        score_components.append(f"{Colors.GREEN}cash generation{Colors.RESET}")
    if de_ratio is not None and de_ratio < 0.5:
        score_components.append(f"{Colors.GREEN}strong balance sheet{Colors.RESET}")
    if accrual > 0:
        score_components.append(f"{Colors.GREEN}earnings quality{Colors.RESET}")
    if rev_growth is not None and rev_growth > 10:
        score_components.append(f"{Colors.GREEN}growth{Colors.RESET}")
    if base_price > 0 and base_price > current_price * 1.2:
        score_components.append(f"{Colors.GREEN}undervalued{Colors.RESET}")

    print(f"\n  {Colors.BOLD}{Colors.WHITE}🎯 INVESTMENT THESIS{Colors.RESET}")
    if tag.startswith("STRONG BUY") or tag.startswith("BUY"):
        print(f"  {Colors.GREEN}{Colors.BOLD}🟢 BULLISH{Colors.RESET}")
        print(f"     Multiple positive signals: {'; '.join(score_components[:3]) if score_components else 'strong fundamentals'}.")
        print(f"     DCF suggests significant upside. Favorable entry for long-term investors.")
    elif tag.startswith("ACCUMULATE") or tag == "HOLD":
        print(f"  {Colors.YELLOW}{Colors.BOLD}🟡 NEUTRAL{Colors.RESET}")
        print(f"     Mixed signals. Solid fundamentals ({'; '.join(score_components[:2]) if score_components else 'basic quality'}).")
        print(f"     Price reflects much of the value. Suitable for existing holders.")
    else:
        print(f"  {Colors.RED}{Colors.BOLD}🔴 CAUTION{Colors.RESET}")
        print(f"     DCF suggests overvalued ({tag_msg}). Strong qualities ({'; '.join(score_components[:2]) if score_components else 'some positives'}).")
        print(f"     Price embeds aggressive growth expectations. New buyers: wait for pullback.")
        print(f"     Holders: monitor closely for deteriorating fundamentals.")

    # 10. Key Risks
    risks = []
    if fcf and fcf < 0:
        risks.append(f"{Colors.RED}negative FCF{Colors.RESET}")
    if capex is not None and op_cash and op_cash != 0 and abs(capex) / op_cash > 0.5:
        risks.append(f"{Colors.RED}high CapEx dependency{Colors.RESET}")
    if accrual < -0.2:
        risks.append(f"{Colors.RED}earnings quality concerns{Colors.RESET}")
    if beta is not None and beta > 1.5:
        risks.append(f"{Colors.RED}high volatility{Colors.RESET}")
    if de_ratio is not None and de_ratio > 1.0:
        risks.append(f"{Colors.YELLOW}elevated leverage{Colors.RESET}")
    if not risks:
        risks.append(f"{Colors.GREEN}no critical risk factors{Colors.RESET}")

    print(f"\n  {Colors.BOLD}⚠️ KEY RISKS{Colors.RESET}")
    print(f"     {', '.join(risks)}. Inform position sizing.")

    # 11. Analyst Perspective
    target_mean = raw.get("target_mean")
    if target_mean is not None and current_price > 0:
        upside = ((float(target_mean) - current_price) / current_price) * 100
        tc = Colors.GREEN if upside > 20 else (Colors.YELLOW if upside > 0 else Colors.RED)
        print(f"\n  {Colors.BOLD}🌍 ANALYST PERSPECTIVE{Colors.RESET}")
        print(f"     Consensus target: ${float(target_mean):.2f} ({tc}{upside:+.1f}% vs current ${current_price:.2f}{Colors.RESET})")
        print(f"     {'Above current price — analysts see upside.' if upside > 0 else 'Below current price — analysts see overvaluation.' if upside < 0 else 'Near current price — fair value.'}")

    # 12. FCF Trend
    fcf_history = raw.get("fcf_history")
    if fcf_history and len(fcf_history) >= 2:
        print(f"\n  {Colors.BOLD}📈 FCF TREND{Colors.RESET}")
        years = sorted(fcf_history.keys())
        trend_parts = []
        for y in years:
            val = fcf_history[y]
            if val == val and val > 0:
                trend_parts.append(f"{Colors.GREEN}{y}: ${val/1e9:.2f}B{Colors.RESET}")
            elif val == val:
                trend_parts.append(f"{Colors.RED}{y}: ${val/1e9:.2f}B{Colors.RESET}")
            else:
                trend_parts.append(f"{y}: N/A")
        print(f"     {' | '.join(trend_parts)}")
        if len(years) >= 2:
            prev_y = years[-2]
            if prev_y in fcf_history and fcf_history[prev_y] == fcf_history[prev_y] and fcf_history[years[-1]] == fcf_history[years[-1]]:
                prev_val = fcf_history[prev_y]
                latest_val = fcf_history[years[-1]]
                if prev_val > 0 and latest_val > 0:
                    growth = ((latest_val - prev_val) / prev_val) * 100
                    if growth > 0:
                        print(f"     {Colors.GREEN}FCF growing {growth:.1f}% YoY{Colors.RESET}")
                    else:
                        print(f"     {Colors.RED}FCF declining {abs(growth):.1f}% YoY{Colors.RESET}")

    # ── LAYMAN SUMMARY ──
    print(f"\n  {Colors.BOLD}{Colors.CYAN}════════════════════════════════════════════════════{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.CYAN}  🧑 LAYMAN'S SUMMARY — Plain English{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.CYAN}════════════════════════════════════════════════════{Colors.RESET}")

    layman_parts = []

    if fcf is not None:
        if fcf > 0:
            layman_parts.append(f"  💵 💰 Is it making money? YES. Keeps ${fcf/1e9:.2f}B in cash after paying everything.")
        else:
            layman_parts.append(f"  💵 💰 Is it making money? NO. Spent more cash than it brought in.")

    if dcf and current_price > 0:
        base_price = dcf.get("Base Case", {}).get("estimated_stock_price", 0)
        if base_price > 0:
            gap = ((base_price - current_price) / current_price) * 100
            if gap > 25:
                layman_parts.append(f"  💵 💲 Is the stock price fair? CHEAP. Costs ${current_price:.2f} but worth about ${base_price:.2f}.")
            elif gap > 0:
                layman_parts.append(f"  💵 💲 Is the stock price fair? KIND OF. Price ${current_price:.2f}, worth about ${base_price:.2f} — close enough.")
            else:
                layman_parts.append(f"  💵 💲 Is the stock price fair? EXPENSIVE. Costs ${current_price:.2f} but only worth ${base_price:.2f}.")

    beta_val = calc.get("beta")
    raw_beta = raw.get("beta")
    beta_use = beta_val if beta_val is not None else raw_beta
    de_ratio_val = calc.get("debt_to_equity")
    risk_level = "LOW"
    if beta_use is not None and beta_use > 1.5:
        risk_level = "HIGH"
    elif beta_use is not None and beta_use > 1.0:
        risk_level = "MODERATE"
    if de_ratio_val is not None and float(de_ratio_val) > 1.0:
        risk_level = "HIGH"

    if risk_level == "LOW":
        layman_parts.append(f"  💵 ⚠️ How risky is it? LOW. Steady business, not much debt, stable.")
    elif risk_level == "MODERATE":
        layman_parts.append(f"  💵 ⚠️ How risky is it? MODERATE. Some ups and downs expected. Normal for stocks.")
    else:
        layman_parts.append(f"  💵 ⚠️ How risky is it? HIGH. The stock moves a lot and/or has a lot of debt.")

    rev_g_val = calc.get("revenue_growth_yfinance") or raw.get("revenue_growth")
    if rev_g_val is not None and float(rev_g_val) > 10:
        layman_parts.append(f"  💵 📈 Is it growing? YES. Revenue growing {float(rev_g_val):.1f}% — getting bigger.")
    elif rev_g_val is not None and float(rev_g_val) > 0:
        layman_parts.append(f"  💵 📈 Is it growing? A LITTLE. Revenue up {float(rev_g_val):.1f}% — slow but steady.")
    else:
        layman_parts.append(f"  💵 📈 Is it growing? NOT MUCH. Revenue flat or shrinking.")

    if tag.startswith("STRONG BUY") or tag.startswith("BUY"):
        layman_parts.append(f"  💵 🤔 Should I buy? YES, according to the numbers. Good value for money.")
    elif tag.startswith("ACCUMULATE") or tag == "HOLD":
        layman_parts.append(f"  💵 🤔 Should I buy? MAYBE. OK to own but not a screaming deal right now.")
    else:
        layman_parts.append(f"  💵 🤔 Should I buy? NOT RIGHT NOW. Wait for a lower price or better news.")

    score = 0
    if fcf and fcf > 0: score += 1
    if de_ratio_val is not None and float(de_ratio_val) < 0.5: score += 1
    if rev_g_val is not None and float(rev_g_val) > 10: score += 1
    if tag.startswith("BUY") or tag.startswith("STRONG BUY"): score += 1

    score_label = f"{score}/4"
    if score >= 4:
        score_color = Colors.GREEN
    elif score >= 3:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED

    layman_parts.append(f"  💵 📊 OVERALL SCORE: {score_color}{score_label}{Colors.RESET} (money-making + low debt + growing + good value)")

    for part in layman_parts:
        print(f"     {part}")

    print(f"\n  {Colors.DIM}{'═' * 50}{Colors.RESET}")
    print(f"  💻 Type the ticker for a new analysis, or ⌂ for menu")
    print(f"  {Colors.DIM}{'═' * 50}{Colors.RESET}")


def _get_valuation_tag(current_price: float, intrinsic: float) -> tuple:
    """
    Returns a precise valuation tag and recommendation based on the gap
    between current price and DCF intrinsic value.
    """
    if current_price <= 0 or intrinsic <= 0:
        return "N/A", "—"

    gap = ((intrinsic - current_price) / current_price) * 100

    if gap > 50:
        return "STRONG BUY", f"🟢 UNDERVALUED by {gap:.1f}% — significant margin of safety"
    elif gap > 25:
        return "BUY", f"🟢 UNDERVALUED by {gap:.1f}% — attractive entry point"
    elif gap > 10:
        return "ACCUMULATE", f"🟢 SLIGHTLY UNDERVALUED by {gap:.1f}%"
    elif gap > -10:
        return "HOLD", f"🟡 FAIRLY VALUED ({gap:+.1f}%) — near intrinsic value"
    elif gap > -25:
        return "REDUCE", f"🔴 OVERVALUED by {abs(gap):.1f}%"
    elif gap > -50:
        return "SELL", f"🔴 OVERVALUED by {abs(gap):.1f}%"
    else:
        return "STRONG SELL", f"🔴 OVERVALUED by {abs(gap):.1f}%"


def _get_valuation_tag(current_price: float, intrinsic: float) -> tuple:
    """
    Returns a precise valuation tag and recommendation based on the gap
    between current price and DCF intrinsic value.
    """
    if current_price <= 0 or intrinsic <= 0:
        return "N/A", "—"

    gap = ((intrinsic - current_price) / current_price) * 100

    if gap > 50:
        return "STRONG BUY", f"🟢 UNDERVALUED by {gap:.1f}% — significant margin of safety"
    elif gap > 25:
        return "BUY", f"🟢 UNDERVALUED by {gap:.1f}% — attractive entry point"
    elif gap > 10:
        return "ACCUMULATE", f"🟢 SLIGHTLY UNDERVALUED by {gap:.1f}%"
    elif gap > -10:
        return "HOLD", f"🟡 FAIRLY VALUED ({gap:+.1f}%) — near intrinsic value"
    elif gap > -25:
        return "REDUCE", f"🔴 OVERVALUED by {abs(gap):.1f}%"
    elif gap > -50:
        return "SELL", f"🔴 OVERVALUED by {abs(gap):.1f}%"
    else:
        return "STRONG SELL", f"🔴 OVERVALUED by {abs(gap):.1f}%"


def _assess_risk(raw: Dict, calc: Dict) -> str:
    """Assess overall risk profile based on financial metrics."""
    risk_factors = []
    fcf = calc.get("free_cash_flow")
    op_cash = raw.get("operating_cash_flow")
    capex = raw.get("capital_expenditures")
    net_income = raw.get("net_income")

    # FCF stability check
    if fcf is not None and op_cash and op_cash != 0:
        fcf_ratio = fcf / op_cash
        if fcf_ratio < 0.1:
            risk_factors.append("⚠️ FCF barely covers OpEx after CapEx — low financial cushion")
        elif fcf_ratio > 0.5:
            risk_factors.append("✅ Strong FCF conversion — healthy financial cushion")

    # CapEx risk
    if capex is not None and op_cash and op_cash != 0:
        intensity = abs(capex) / op_cash
        if intensity > 0.5:
            risk_factors.append("⚠️ High CapEx intensity — vulnerable to downturns")

    # Accrual risk
    if net_income is not None and op_cash is not None and net_income != 0:
        accrual = (op_cash - net_income) / abs(net_income)
        if accrual < -0.5:
            risk_factors.append("⚠️ Large gap between NI and OCF — potential earnings quality issue")

    if not risk_factors:
        risk_factors.append("✅ Low risk profile across key indicators")

    return "\n     ".join(risk_factors)


def _display_single_analysis(state: ResearchState) -> None:
    """Display detailed contextual analysis with valuation tags and risk assessment."""
    raw = state.raw_financial_data
    calc = state.calculated_metrics
    dcf = calc.get('dcf_scenarios', {})
    report = state.final_report
    ticker = state.ticker
    company = raw.get("company_name", ticker)
    current_price = report.get("current_price", 0)
    shares = raw.get("shares_outstanding", 1)

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
    print("  ═══════════════════════════════════════════════════════")
    print(f"  📖  DETAILED METRICS ANALYSIS — {ticker} ({company})")
    print("  ═══════════════════════════════════════════════════════")
    print(f"{Colors.RESET}")

    # ── Valuation Summary Banner ──
    if dcf and current_price > 0:
        base_price = dcf.get("Base Case", {}).get("estimated_stock_price", 0)
        tag, tag_msg = _get_valuation_tag(current_price, base_price)
        print(f"\n  ┌──────────────────────────────────────────────────┐")
        print(f"  │  VALUATION TAG: {tag:<18} │")
        print(f"  │  {tag_msg:<46} │")
        print(f"  └──────────────────────────────────────────────────┘")

    # ── 1. Cash Flow Health ──
    print(f"\n  💰 CASH FLOW ANALYSIS")
    fcf = calc.get("free_cash_flow")
    op_cash = raw.get("operating_cash_flow")
    if fcf is not None and op_cash is not None:
        fcf_margin = fcf / op_cash if op_cash != 0 else 0
        fcf_label = "HEALTHY" if fcf_margin > 0.3 else "THIN" if fcf_margin > 0.1 else "WEAK"
        print(f"     FCF: ${fcf/1e9:.2f}B | FCF/OCF: {fcf_margin*100:.1f}% [{fcf_label}]")
        if fcf > 0:
            fcf_per_share = fcf / shares if shares else 0
            print(f"     FCF Per Share: ${fcf_per_share:.2f}")
            if fcf_per_share > 0:
                fcf_yield_on_price = fcf_per_share / current_price * 100 if current_price > 0 else 0
                print(f"     FCF Yield (on price): {fcf_yield_on_price:.2f}%")

    # ── 2. Earnings Quality ──
    print(f"\n  📊 EARNINGS QUALITY")
    net_income = raw.get("net_income")
    op_cash = raw.get("operating_cash_flow")
    if net_income is not None and op_cash is not None and net_income != 0:
        accrual = (op_cash - net_income) / abs(net_income)
        quality = "HIGH" if accrual > 0 else "LOW" if accrual < -0.2 else "MODERATE"
        print(f"     Accrual Ratio: {accrual*100:+.1f}% [{quality} quality]")
        print(f"     OCF/NI Ratio: {op_cash/net_income:.2f}x")
        if accrual > 0.3:
            print(f"     ✅ OCF significantly exceeds NI — cash-backed earnings")
        elif accrual < -0.3:
            print(f"     ⚠️ NI exceeds OCF — watch for aggressive accounting")

    # ── 3. Capital Efficiency ──
    print(f"\n  ⚙️ CAPITAL EFFICIENCY")
    capex = raw.get("capital_expenditures")
    if capex is not None and op_cash is not None and op_cash != 0:
        intensity = abs(capex) / op_cash
        model = "ASSET-LIGHT" if intensity < 0.3 else "MODERATE" if intensity < 0.5 else "CAPITAL-HEAVY"
        print(f"     CapEx Intensity: {intensity*100:.1f}% [{model}]")
        if intensity < 0.2:
            print(f"     ✅ Minimal reinvestment needed — high cash return on capital")
        elif intensity > 0.7:
            print(f"     ⚠️ Heavy reinvestment — cyclical vulnerability increases")

    # ── 4. DCF Scenario Valuation ──
    if dcf and current_price > 0:
        print(f"\n  🎯 DCF VALUATION MATRIX")
        print(f"  ┌─────────────────┬────────────┬────────────┬────────────┬───────────┐")
        print(f"  │ Scenario        │ Growth     │ Target     │ Gap vs     │ Tag       │")
        print(f"  │                 │ Rate       │ Price      │ Current    │           │")
        print(f"  ├─────────────────┼────────────┼────────────┼────────────┼───────────┤")
        for name, data in dcf.items():
            target = data['estimated_stock_price']
            gap = ((target - current_price) / current_price) * 100
            tag, _ = _get_valuation_tag(current_price, target)
            # Truncate tag for table
            tag_short = tag[:10] if len(tag) > 10 else tag
            print(f"  │ {name:<15} │ {data['growth_rate']*100:>6.0f}%    │ ${target:<8.2f} │ {gap:>+7.1f}%   │ {tag_short:<9} │")
        print(f"  └─────────────────┴────────────┴────────────┴────────────┴───────────┘")

        # Overall verdict
        base_price = dcf.get("Base Case", {}).get("estimated_stock_price", 0)
        if base_price > 0:
            tag, tag_msg = _get_valuation_tag(current_price, base_price)
            print(f"\n  🏷️ VERDICT: {tag}")
            print(f"     {tag_msg}")

            # Price targets
            conservative = dcf.get("Conservative", {}).get("estimated_stock_price", 0)
            aggressive = dcf.get("Aggressive", {}).get("estimated_stock_price", 0)
            if conservative > 0 and aggressive > 0:
                print(f"\n  📌 Price Range:")
                print(f"     Floor (Conservative): ${conservative:.2f}")
                print(f"     Base (Base Case):     ${base_price:.2f}")
                print(f"     Ceiling (Aggressive): ${aggressive:.2f}")
                print(f"     Current Price:        ${current_price:.2f}")

    # ── 5. Risk Assessment ──
    print(f"\n  ⚠️ RISK ASSESSMENT")
    risk_text = _assess_risk(raw, calc)
    print(f"     {risk_text}")

    # ── Summary ──
    print(f"\n  📝 SUMMARY")
    takeaway_lines = []
    fcf = calc.get("free_cash_flow")
    if fcf is not None:
        takeaway_lines.append(f"FCF: ${fcf/1e9:.2f}B")
    if dcf:
        base_price = dcf.get("Base Case", {}).get("estimated_stock_price", 0)
        if base_price > 0 and current_price > 0:
            gap = ((base_price - current_price) / current_price) * 100
            takeaway_lines.append(f"DCF Gap: {gap:+.1f}%")
            tag, _ = _get_valuation_tag(current_price, base_price)
            takeaway_lines.append(f"Verdict: {tag}")
    eps = calc.get("eps")
    if eps is not None:
        takeaway_lines.append(f"EPS: ${eps:.2f}")

    if takeaway_lines:
        print(f"     {' | '.join(takeaway_lines)}")

    # ── LAYMAN SUMMARY ──
    print(f"\n  {Colors.BOLD}{Colors.CYAN}════════════════════════════════════════════════════{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.CYAN}  🧑 LAYMAN'S SUMMARY — Plain English{Colors.RESET}")
    print(f"  {Colors.BOLD}{Colors.CYAN}════════════════════════════════════════════════════{Colors.RESET}")

    # Simple questions and answers
    layman_parts = []

    # 1. Is the company making money?
    if fcf is not None:
        if fcf > 0:
            layman_parts.append(f"  💵 💰 Is it making money? YES. The company keeps ${fcf/1e9:.2f}B in cash after paying everything.")
        else:
            layman_parts.append(f"  💵 💰 Is it making money? NO. It spent more cash than it brought in.")

    # 2. Is the stock cheap or expensive?
    if dcf and current_price > 0:
        base_price = dcf.get("Base Case", {}).get("estimated_stock_price", 0)
        if base_price > 0:
            gap = ((base_price - current_price) / current_price) * 100
            if gap > 25:
                layman_parts.append(f"  💵 💲 Is the stock price fair? CHEAP. The stock costs ${current_price:.2f} but is actually worth about ${base_price:.2f}.")
            elif gap > 0:
                layman_parts.append(f"  💵 💲 Is the stock price fair? KIND OF. Price is ${current_price:.2f}, worth about ${base_price:.2f} — close enough.")
            else:
                layman_parts.append(f"  💵 💲 Is the stock price fair? EXPENSIVE. The stock costs ${current_price:.2f} but is only worth about ${base_price:.2f}.")

    # 3. Is it risky?
    beta_val = calc.get("beta")
    raw_beta = raw.get("beta")
    beta_use = beta_val if beta_val is not None else raw_beta
    risk_level = "LOW"
    if beta_use is not None and beta_use > 1.5:
        risk_level = "HIGH"
    elif beta_use is not None and beta_use > 1.0:
        risk_level = "MODERATE"
    de_ratio_val = calc.get("debt_to_equity")
    if de_ratio_val is not None and float(de_ratio_val) > 1.0:
        risk_level = "HIGH"
    
    if risk_level == "LOW":
        layman_parts.append(f"  💵 ⚠️ How risky is it? LOW. Steady business, not much debt, stable.")
    elif risk_level == "MODERATE":
        layman_parts.append(f"  💵 ⚠️ How risky is it? MODERATE. Some ups and downs expected. Normal for stocks.")
    else:
        layman_parts.append(f"  💵 ⚠️ How risky is it? HIGH. The stock moves a lot and/or has a lot of debt.")

    # 4. Is it growing?
    rev_g_val = calc.get("revenue_growth_yfinance") or raw.get("revenue_growth")
    if rev_g_val is not None and float(rev_g_val) > 10:
        layman_parts.append(f"  💵 📈 Is it growing? YES. Revenue is growing {float(rev_g_val):.1f}% — getting bigger.")
    elif rev_g_val is not None and float(rev_g_val) > 0:
        layman_parts.append(f"  💵 📈 Is it growing? A LITTLE. Revenue up {float(rev_g_val):.1f}% — slow but steady.")
    else:
        layman_parts.append(f"  💵 📈 Is it growing? NOT MUCH. Revenue is flat or shrinking.")

    # 5. Would I buy it?
    if tag.startswith("STRONG BUY") or tag.startswith("BUY"):
        layman_parts.append(f"  💵 🤔 Should I buy? YES, according to the numbers. Good value for money.")
    elif tag.startswith("ACCUMULATE") or tag == "HOLD":
        layman_parts.append(f"  💵 🤔 Should I buy? MAYBE. It's OK to own but not a screaming deal right now.")
    else:
        layman_parts.append(f"  💵 🤔 Should I buy? NOT RIGHT NOW. Wait for a lower price or better news.")

    # 6. Quick score
    score = 0
    if fcf and fcf > 0: score += 1
    if de_ratio_val is not None and float(de_ratio_val) < 0.5: score += 1
    if rev_g_val is not None and float(rev_g_val) > 10: score += 1
    if tag.startswith("BUY") or tag.startswith("STRONG BUY"): score += 1
    
    score_label = f"{score}/4"
    if score >= 4:
        score_color = Colors.GREEN
    elif score >= 3:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED

    layman_parts.append(f"  💵 📊 OVERALL SCORE: {score_color}{score_label}{Colors.RESET} (money-making + low debt + growing + good value)")

    # Print layman summary
    for part in layman_parts:
        print(f"     {part}")

    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    print(f"  💻 Type the ticker for a new analysis, or ⌂ for menu")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")


def _print_single_takeaways(raw, calc, dcf, current_price) -> None:
    """Print concise takeaways for comparison view with valuation tags."""
    fcf = calc.get("free_cash_flow")
    if fcf is not None:
        if fcf > 0:
            print(f"     ✅ FCF: ${fcf/1e9:.2f}B — generates real cash")
        else:
            print(f"     ❌ FCF: ${fcf/1e9:.2f}B — burning cash")

    if "Conservative" in dcf and current_price > 0:
        conservative_price = dcf["Conservative"]["estimated_stock_price"]
        gap = ((conservative_price - current_price) / current_price) * 100
        tag, tag_msg = _get_valuation_tag(current_price, conservative_price)
        print(f"     🏷️ Conservative: {tag} ({gap:+.1f}%)")

    net_income = raw.get("net_income")
    op_cash = raw.get("operating_cash_flow")
    if net_income is not None and op_cash is not None and net_income != 0:
        accrual = (op_cash - net_income) / abs(net_income)
        if accrual > 0.2:
            print(f"     ✅ Strong earnings quality (OCF/NI: {op_cash/net_income:.2f}x)")
        elif accrual < -0.1:
            print(f"     ⚠️ Weak earnings quality (OCF/NI: {op_cash/net_income:.2f}x)")
        else:
            print(f"     ℹ️ Earnings quality: OCF/NI: {op_cash/net_income:.2f}x")

    capex = raw.get("capital_expenditures")
    if capex is not None and op_cash is not None and op_cash != 0:
        intensity = abs(capex) / op_cash
        if intensity < 0.3:
            print(f"     ✅ Asset-light ({intensity*100:.1f}% CapEx)")
        elif intensity < 0.5:
            print(f"     ℹ️ Moderate ({intensity*100:.1f}% CapEx)")
        else:
            print(f"     ⚠️ Capital-heavy ({intensity*100:.1f}% CapEx)")

    # Add EPS
    eps = calc.get("eps")
    if eps is not None:
        print(f"     ℹ️ EPS: ${eps:.2f}")


def _display_comparison_analysis(state1, ticker1, state2, ticker2) -> None:
    """Display detailed contextual analysis for two tickers from a comparison."""
    raw1, raw2 = state1.raw_financial_data, state2.raw_financial_data
    calc1, calc2 = state1.calculated_metrics, state2.calculated_metrics
    dcf1 = calc1.get('dcf_scenarios', {})
    dcf2 = calc2.get('dcf_scenarios', {})
    report1 = state1.final_report
    report2 = state2.final_report
    price1 = report1.get("current_price", 0)
    price2 = report2.get("current_price", 0)
    company1 = raw1.get("company_name", ticker1)
    company2 = raw2.get("company_name", ticker2)

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
    print("  ═════════════════════════════════════════════════════")
    print(f"  📖  DETAILED ANALYSIS — {ticker1} vs {ticker2}")
    print("  ═════════════════════════════════════════════════════")
    print(f"{Colors.RESET}")

    # ── Side-by-side summary ──
    print(f"\n  📊 SIDE-BY-SIDE SUMMARY:")
    print(f"  ┌─────────────────┬────────────────┬────────────────┐")
    print(f"  │ Metric          │ {ticker1:<14} │ {ticker2:<14} │")
    print(f"  ├─────────────────┼────────────────┼────────────────┤")

    comparisons = [
        ("Price", f"${price1:.2f}", f"${price2:.2f}", "text"),
        ("Market Cap", f"${raw1.get('market_cap', 0)/1e9:.1f}B", f"${raw2.get('market_cap', 0)/1e9:.1f}B", "billions"),
        ("Net Income", f"${raw1.get('Net Income', 0)/1e9:.2f}B", f"${raw2.get('Net Income', 0)/1e9:.2f}B", "billions"),
        ("Operating CF", f"${raw1.get('Operating Cash Flow', 0)/1e9:.2f}B", f"${raw2.get('Operating Cash Flow', 0)/1e9:.2f}B", "billions"),
        ("FCF", f"${calc1.get('free_cash_flow', 0)/1e9:.2f}B", f"${calc2.get('free_cash_flow', 0)/1e9:.2f}B", "billions"),
        ("FCF Margin", f"{calc1.get('fcf_margin', 0)*100:.2f}%", f"{calc2.get('fcf_margin', 0)*100:.2f}%", "pct"),
        ("FCF Yield", f"{calc1.get('fcf_yield', 0)*100:.2f}%", f"{calc2.get('fcf_yield', 0)*100:.2f}%", "pct"),
        ("EPS", f"${calc1.get('eps', 0):.2f}", f"${calc2.get('eps', 0):.2f}", "usd"),
        ("CapEx Intensity", f"{abs(raw1.get('Capital Expenditures', 0))/raw1.get('Operating Cash Flow', 1)*100:.1f}%", f"{abs(raw2.get('Capital Expenditures', 0))/raw2.get('Operating Cash Flow', 1)*100:.1f}%", "pct"),
    ]

    for label, v1, v2, _fmt in comparisons:
        print(f"  │ {label:<15} │ {v1:<14} │ {v2:<14} │")
    print(f"  └─────────────────┴────────────────┴────────────────┘")

    # ── Valuation tags side-by-side ──
    print(f"\n  🏷️ VALUATION TAGS:")
    if dcf1 and price1 > 0:
        base1 = dcf1.get("Base Case", {}).get("estimated_stock_price", 0)
        tag1, msg1 = _get_valuation_tag(price1, base1)
    else:
        tag1, msg1 = "N/A", "—"
    if dcf2 and price2 > 0:
        base2 = dcf2.get("Base Case", {}).get("estimated_stock_price", 0)
        tag2, msg2 = _get_valuation_tag(price2, base2)
    else:
        tag2, msg2 = "N/A", "—"
    print(f"     {ticker1}: {tag1} — {msg1}")
    print(f"     {ticker2}: {tag2} — {msg2}")

    # ── Risk assessment ──
    print(f"\n  ⚠️ RISK PROFILE:")
    print(f"     {ticker1}:")
    print(f"     {_assess_risk(raw1, calc1)}")
    print(f"     {ticker2}:")
    print(f"     {_assess_risk(raw2, calc2)}")

    # ── Individual conclusions ──
    print(f"\n  💡 {ticker1} ({company1}) — DETAILED:")
    _print_single_takeaways(raw1, calc1, dcf1, price1)
    print(f"\n  💡 {ticker2} ({company2}) — DETAILED:")
    _print_single_takeaways(raw2, calc2, dcf2, price2)

    # ── Overall verdict ──
    fcf1 = calc1.get("free_cash_flow", 0)
    fcf2 = calc2.get("free_cash_flow", 0)
    print(f"\n  {'─' * 50}")
    if fcf1 > fcf2:
        print(f"  🏆 Stronger FCF Profile: {ticker1} (${fcf1/1e9:.2f}B vs ${fcf2/1e9:.2f}B)")
    else:
        print(f"  🏆 Stronger FCF Profile: {ticker2} (${fcf2/1e9:.2f}B vs ${fcf1/1e9:.2f}B)")

    # Add valuation comparison
    if dcf1 and dcf2 and price1 > 0 and price2 > 0:
        base1 = dcf1.get("Base Case", {}).get("estimated_stock_price", 0)
        base2 = dcf2.get("Base Case", {}).get("estimated_stock_price", 0)
        gap1 = ((base1 - price1) / price1) * 100 if base1 > 0 else 0
        gap2 = ((base2 - price2) / price2) * 100 if base2 > 0 else 0
        if gap1 > gap2:
            print(f"  📊 Better DCF Upside: {ticker1} ({gap1:+.1f}% vs {gap2:+.1f}%)")
        else:
            print(f"  📊 Better DCF Upside: {ticker2} ({gap2:+.1f}% vs {gap1:+.1f}%)")

    print(f"  {'─' * 50}")
    print(f"  💻 Type the ticker for a new analysis, or ⌂ for menu")
    print(f"  {'─' * 50}")


def _print_single_takeaways(raw, calc, dcf, current_price) -> None:
    """Print specific takeaways for a single ticker."""
    fcf = calc.get("free_cash_flow")
    if fcf is not None:
        if fcf > 0:
            print(f"     ✅ FCF: ${fcf/1e9:.2f}B — generates real cash")
        else:
            print(f"     ❌ FCF: ${fcf/1e9:.2f}B — burning cash")
        if "Conservative" in dcf and current_price > 0:
            conservative_price = dcf["Conservative"]["estimated_stock_price"]
            gap = ((conservative_price - current_price) / current_price) * 100
            print(f"     📊 Conservative DCF target: ${conservative_price:.2f} ({gap:+.1f}% vs ${current_price:.2f})")

    net_income = raw.get("net_income")
    op_cash = raw.get("operating_cash_flow")
    if net_income is not None and op_cash is not None and net_income != 0:
        accrual = (op_cash - net_income) / abs(net_income)
        if accrual > 0.2:
            print(f"     ✅ Strong earnings quality: OCF exceeds NI by {accrual*100:.1f}%")
        elif accrual < -0.1:
            print(f"     ⚠️ Weak earnings quality")
        else:
            print(f"     ℹ️ Earnings quality: OCF/NI gap {accrual*100:.1f}%")

    capex = raw.get("capital_expenditures")
    if capex is not None and op_cash is not None and op_cash != 0:
        intensity = abs(capex) / op_cash
        if intensity < 0.3:
            print(f"     ✅ Asset-light: CapEx {intensity*100:.1f}%")
        elif intensity < 0.5:
            print(f"     ℹ️ Moderate CapEx: {intensity*100:.1f}%")
        else:
            print(f"     ⚠️ Capital-heavy: {intensity*100:.1f}% — cyclical sensitivity")


# ---------------------------------------------------------------------------
# INTERACTIVE APP — Menu, Explanation, Comparison
# ---------------------------------------------------------------------------
def display_banner():
    """Print the app banner."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   🏦  FINANCE RESEARCH TERMINAL v2.0  🏦            ║")
    print("  ║   Multi-Agent DCF Analysis Engine                   ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")

def display_main_menu() -> str:
    """Display the main menu and return the user's choice."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}  ┌──────────────────────────────────────────┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  │                                        │{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  │   1  🔍  Analyze a Ticker               │{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  │   2  📊  Compare Two Tickers            │{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  │   3  📖  Metric Guide (General)         │{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  │   4  📊  Metric Analysis (Last Run)     │{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  │   5  ❌  Exit                           │{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  │                                        │{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  └──────────────────────────────────────────┘{Colors.RESET}")
    return input(f"\n  {Colors.GREEN}Select option{Colors.RESET} ({Colors.CYAN}1-5{Colors.RESET}): ").strip()


def run_analysis(ticker: str) -> Optional[ResearchState]:
    """
    Execute the full pipeline for a single ticker.
    Returns the populated ResearchState, or None on error.
    Stores result in global LAST_STATE for option 4.
    Resets LAST_COMPARISON since this is a single analysis.
    """
    global LAST_STATE, LAST_COMPARISON
    LAST_COMPARISON = None
    state = ResearchState()
    try:
        agent1 = DataIngestionAgent(state)
        agent1.ingest(ticker)
        agent2 = CalculationAgent(state)
        agent2.calculate()
        state = run_hardened_buyback_dcf(state)
        agent3 = SynthesisAgent(state)
        agent3.synthesize()
        agent3.print_report()
        LAST_STATE = state
        return state
    except Exception as e:
        print(f"\n{Colors.RED}  ❌ Error analyzing {ticker}: {e}{Colors.RESET}")
        return None


def run_comparison(ticker1: str, ticker2: str):
    """Run both tickers through the pipeline and display side-by-side results."""
    global LAST_COMPARISON
    print(f"\n{Colors.BOLD}  📊 Running side-by-side comparison: {ticker1} vs {ticker2}{Colors.RESET}")
    print(f"{Colors.DIM}  ⏳ Fetching data for both tickers…{Colors.RESET}")

    state1 = run_analysis(ticker1)
    state2 = run_analysis(ticker2)

    if state1 and state2:
        r1 = state1.final_report
        r2 = state2.final_report

        print(f"\n{Colors.BOLD}{Colors.CYAN}  ════════════════════════════════════════════════{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  📊 COMPARISON REPORT{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  ════════════════════════════════════════════════{Colors.RESET}")

        # Side-by-side table
        print(f"\n  {'Metric':<30} {'─'*5} {ticker1:<12} {'│':^3} {ticker2:<12}")
        print(f"  {'─'*30} {'─'*5} {'─'*12} {'─'*3} {'─'*12}")

        raw1, raw2 = r1.get('raw_data', {}), r2.get('raw_data', {})
        calc1, calc2 = r1.get('calculated_metrics', {}), r2.get('calculated_metrics', {})
        dcf1, dcf2 = r1.get('dcf_scenarios', {}), r2.get('dcf_scenarios', {})


        comparisons = [
            ("Company", r1.get('company_name', ''), r2.get('company_name', ''), 'text'),
            ("Market Price", r1.get('current_price', 0), r2.get('current_price', 0), 'usd'),
            ("Net Income", raw1.get('Net Income', 0), raw2.get('Net Income', 0), 'billions'),
            ("Operating Cash Flow", raw1.get('Operating Cash Flow', 0), raw2.get('Operating Cash Flow', 0), 'billions'),
            ("Free Cash Flow", calc1.get('free_cash_flow', 0), calc2.get('free_cash_flow', 0), 'billions'),
            ("FCF Margin", calc1.get('fcf_margin', 0), calc2.get('fcf_margin', 0), 'pct'),
            ("EPS", calc1.get('eps', 0), calc2.get('eps', 0), 'usd'),
            ("CapEx Intensity", abs(raw1.get('Capital Expenditures', 0)) / raw1.get('Operating Cash Flow', 1), abs(raw2.get('Capital Expenditures', 0)) / raw2.get('Operating Cash Flow', 1), 'pct'),
        ]

        for label, v1, v2, fmt_type in comparisons:
            if fmt_type == 'text':
                s1, s2 = str(v1), str(v2)
            elif fmt_type == 'usd':
                s1, s2 = f"${v1:.2f}", f"${v2:.2f}"
            elif fmt_type == 'billions':
                s1, s2 = f"${v1/1e9:.2f}B", f"${v2/1e9:.2f}B"
            elif fmt_type == 'pct':
                s1, s2 = f"{v1*100:.2f}%", f"{v2*100:.2f}%"
            else:
                s1, s2 = str(v1), str(v2)
            print(f"  {label:<30} {'─'*5} {s1:<12} {'│':^3} {s2:<12}")

        # DCF scenario comparison
        print(f"\n  {'DCF Base Case Target':<30} {'─'*5} {'─'*12} {'─'*3} {'─'*12}")
        price1 = dcf1.get('Base Case', {}).get('estimated_stock_price', 0) if isinstance(dcf1, dict) else 0
        price2 = dcf2.get('Base Case', {}).get('estimated_stock_price', 0) if isinstance(dcf2, dict) else 0
        print(f"  {'Base Case Target':<30} {'─'*5} {'$' + format(price1, '.2f'):<12} {'│':^3} {'$' + format(price2, '.2f'):<12}")

        # Winner
        fcf1 = calc1.get('free_cash_flow', 0)
        fcf2 = calc2.get('free_cash_flow', 0)
        if fcf1 > fcf2:
            winner = r1.get('company_name', '')
        else:
            winner = r2.get('company_name', '')
        print(f"\n  {Colors.GREEN}🏆 Stronger FCF Profile: {winner}{Colors.RESET}")

        print(f"\n{Colors.CYAN}  ════════════════════════════════════════════════{Colors.RESET}\n")

        # Save states for potential further analysis
        LAST_COMPARISON = (state1, ticker1, state2, ticker2)
        return state1, state2
    return None


def display_metric_explanations():
    """Display human-readable explanations of every metric in the report."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}  📖 METRIC EXPLANATION GUIDE{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}  ═══════════════════════════════════════════════{Colors.RESET}")

    explanations = [
        ("Net Income", "Total profit after all expenses, taxes, and costs. The bottom line."),
        ("Operating Cash Flow", "Actual cash generated by core business operations. More reliable than net income."),
        ("Capital Expenditures", "Money spent on physical assets (factories, equipment). Shown as negative."),
        ("Free Cash Flow (FCF)", "OCF minus CapEx. The cash truly available to shareholders."),
        ("FCF Margin", "FCF as a percentage of revenue. Higher = more efficient cash generation."),
        ("FCF Yield", "FCF divided by market cap. Shows how much cash you get per dollar invested."),
        ("Earnings Per Share (EPS)", "Net income divided by shares. What each share earns."),
        ("FCF Per Share", "FCF divided by shares. The 'real' earning power per share."),
        ("CapEx Intensity", "CapEx / OCF. Lower = asset-light (like software). Higher = capital-heavy (like manufacturing)."),
        ("Accrual Ratio", "OCF vs Net Income gap. Positive = strong cash backing for reported earnings."),
        ("DCF Intrinsic Value", "Present value of all future cash flows. The 'fair' price based on fundamentals."),
        ("Buyback Effect", "Share count reduction over time. Each year's buyback amplifies per-share value."),
    ]

    for metric, description in explanations:
        print(f"\n  {Colors.CYAN}▸ {metric}{Colors.RESET}")
        print(f"    {Colors.DIM}{description}{Colors.RESET}")

    print(f"\n{Colors.BOLD}{Colors.WHITE}  ═══════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.DIM}  Tip: A quality score of 3/3 means all health checks passed.{Colors.RESET}")
    print(f"{Colors.DIM}  Negative 'vs Current Price' means the DCF thinks the stock is priced above intrinsic value.{Colors.RESET}")
    print(f"{Colors.DIM}  CapEx intensity below 30% is generally preferred for stability.{Colors.RESET}")
    print()


def interactive_loop():
    """
    The main application loop. Presents a menu, handles user choices,
    and dispatches to the appropriate pipeline.
    """
    Colors.clear()
    display_banner()

    while True:
        choice = display_main_menu()

        if choice == "1":
            print(f"\n{Colors.DIM}  Type a ticker symbol (e.g., AAPL, MSFT, NVDA) or 'back' to return.{Colors.RESET}")
            ticker = input(f"  {Colors.GREEN}💻 Enter ticker:{Colors.RESET} ").strip().upper()
            if ticker.lower() == "back":
                continue
            if not ticker:
                print(f"  {Colors.RED}  ❌ No ticker provided.{Colors.RESET}")
                continue
            run_analysis(ticker)

        elif choice == "2":
            print(f"\n{Colors.DIM}  Enter two tickers to compare side-by-side.{Colors.RESET}")
            t1 = input(f"  {Colors.GREEN}💻 First ticker:{Colors.RESET} ").strip().upper()
            t2 = input(f"  {Colors.GREEN}💻 Second ticker:{Colors.RESET} ").strip().upper()
            if t1.lower() == "back" or t2.lower() == "back":
                continue
            if not t1 or not t2:
                print(f"  {Colors.RED}  ❌ Both tickers required.{Colors.RESET}")
                continue
            run_comparison(t1, t2)
            input(f"\n  {Colors.DIM}Press Enter to continue...{Colors.RESET}")

        elif choice == "3":
            display_metric_explanations()
            input(f"\n  {Colors.DIM}Press Enter to return to menu...{Colors.RESET}")

        elif choice == "4":
            display_contextual_analysis()
            input(f"\n  {Colors.DIM}Press Enter to return to menu...{Colors.RESET}")

        elif choice == "5" or choice.lower() == "quit":
            print(f"\n{Colors.CYAN}")
            print("  ┌──────────────────────────────────────────┐")
            print("  │  👋 Thanks for using Finance Terminal!   │")
            print("  │  Happy investing. 🚀                     │")
            print("  └──────────────────────────────────────────┘")
            print(f"{Colors.RESET}\n")
            break

        else:
            print(f"\n{Colors.RED}  ❌ Invalid option. Please choose 1-5.{Colors.RESET}")


def main() -> None:
    """Entry point — launches the interactive Finance Research Terminal."""
    try:
        interactive_loop()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}  ⌹ Interrupted. Goodbye!{Colors.RESET}\n")
    except EOFError:
        print(f"\n{Colors.YELLOW}  ⌹ No input detected. Goodbye!{Colors.RESET}\n")


if __name__ == "__main__":
    main()
