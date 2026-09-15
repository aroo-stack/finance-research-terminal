# 🏦 Finance Research Terminal

Multi-Agent DCF Analysis Engine built in Python.

## Features

- **3-Agent Pipeline**: Data Ingestion → Calculation → Synthesis
- **DCF Analysis**: Multi-scenario with 2.5% annual buyback factor
- **Valuation**: P/E, P/S, P/B, EV/EBITDA, FCF Yield
- **Ratios**: ROE, ROA, Gross/Operating/Profit Margins, Debt-to-Equity
- **Market Context**: Beta, Analyst Targets, Dividend Yield, 52-week range
- **Valuation Tags**: STRONG BUY → STRONG SELL
- **Layman Summary**: Plain English investment advice
- **.app Bundle**: Double-click to launch on macOS

## Usage

```bash
research_terminal
# or
python3 research_terminal.py
```

### What happens after launch:

1. **Type `1`** → Analyze a ticker → enter ticker → full report
2. **Type `2`** → Compare two tickers → side-by-side results
3. **Type `3`** → Metric guide → what every metric means
4. **Type `4`** → Metric analysis → detailed conclusions from last run
5. **Type `5`** → Exit

## App

```bash
open /Applications/FinanceTerminal.app
```

## How It Works

| Step | Agent | What it does |
|------|-------|-------------|
| 1 | Data Ingestion | Fetches real financial data from Yahoo Finance |
| 2 | Calculation | Computes ratios, margins, valuation multiples |
| 3 | DCF Model | Hardened buyback-adjusted discounted cash flow |
| 4 | Synthesis | Generates color-coded report with layman summary |

## Requirements

- Python 3.14+
- yfinance library
- macOS (for .app bundle)

## Example Output

```
💰 Cash Engine: Strong — FCF $98.77B generates real cash
🎯 Valuation: STRONG SELL — Base: $120.66 (-63.8%)
📊 Earnings Quality: Strong — OCF exceeds NI by 105%
⚙️ CapEx Efficiency: Asset-light — 11.4% CapEx
📈 Growth Profile: Solid — Rev +16.4%, EPS +28.7%
🏦 Balance Sheet: Leveraged — D/E 1.34
🌍 Market Risk: MODERATE — Beta 1.08
🎯 INVESTMENT THESIS: CAUTION
🧑 LAYMAN SUMMARY:
  💰 Is it making money? YES
  💲 Is the stock price fair? EXPENSIVE
  ⚠️ How risky is it? HIGH
  📈 Is it growing? YES
  🤔 Should I buy? NOT RIGHT NOW
  📊 OVERALL SCORE: 2/4
```
