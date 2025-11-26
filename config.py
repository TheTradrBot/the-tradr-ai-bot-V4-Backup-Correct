# config.py
"""
Configuration for Blueprint Trader AI.

You keep:
- secrets (tokens, API keys) in Replit Secrets (env vars)
- public config (channel IDs, instruments, intervals) here
"""

import os


# ==== 5%ers 100K High Stakes Risk Model ====

ACCOUNT_CURRENCY = "USD"
ACCOUNT_SIZE = 100_000
MAX_DAILY_LOSS_PCT = 0.05
MAX_TOTAL_LOSS_PCT = 0.10
RISK_PER_TRADE_PCT = 0.01
MAX_OPEN_RISK_PCT = 0.03
MIN_WITHDRAWAL_USD = 150

CONTRACT_SPECS = {
    "USD_JPY": {"pip_value": 0.01, "contract_size": 100000, "pip_location": 2},
    "GBP_USD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
    "EUR_USD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
    "NZD_USD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
    "AUD_USD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
    "USD_CHF": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
    "USD_CAD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
    "XAU_USD": {"pip_value": 0.01, "contract_size": 100, "pip_location": 2},
    "XAG_USD": {"pip_value": 0.001, "contract_size": 5000, "pip_location": 3},
    "NAS100_USD": {"pip_value": 1.0, "contract_size": 1, "pip_location": 0},
    "SPX500_USD": {"pip_value": 1.0, "contract_size": 1, "pip_location": 0},
    "WTICO_USD": {"pip_value": 0.01, "contract_size": 1000, "pip_location": 2},
    "BCO_USD": {"pip_value": 0.01, "contract_size": 1000, "pip_location": 2},
    "BTC_USD": {"pip_value": 1.0, "contract_size": 1, "pip_location": 0},
    "ETH_USD": {"pip_value": 0.01, "contract_size": 1, "pip_location": 2},
}


# How strict the confluence engine is.
# "standard"  = balanced trades and quality (recommended for live trading)
# "aggressive" = more trades, looser filters (for experimentation/backtesting)
# Set SIGNAL_MODE environment variable to override, e.g., SIGNAL_MODE=aggressive
SIGNAL_MODE = os.getenv("SIGNAL_MODE", "standard")


# ==== Discord ====

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Replace these with your actual channel IDs (integers)
SCAN_CHANNEL_ID = 1442194985142190230        # <- scan channel ID
TRADES_CHANNEL_ID = 1442195008525565962      # <- trades channel ID
TRADE_UPDATES_CHANNEL_ID = 1438452127767859254  # <- trade updates channel ID

# Autoscan interval (hours)
SCAN_INTERVAL_HOURS = 4  # every 4H as per your spec


# ==== Data source: OANDA (practice) ====

OANDA_API_KEY = os.getenv("OANDA_API_KEY")          # set in Replit secrets
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")    # set in Replit secrets
OANDA_API_URL = "https://api-fxpractice.oanda.com"  # practice endpoint

# Granularity mapping for OANDA
GRANULARITY_MAP = {
    "M": "M",      # Monthly
    "W": "W",      # Weekly
    "D": "D",      # Daily
    "H4": "H4",    # 4-hour
}


# ==== Instruments & groups ====
# All available OANDA instruments

# OANDA FX pairs - Major, Minor, and Exotic pairs
FOREX_PAIRS = [
    # Majors
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
    # Crosses
    "EUR_GBP",
    "EUR_JPY",
    "EUR_CHF",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_NZD",
    "GBP_JPY",
    "GBP_CHF",
    "GBP_AUD",
    "GBP_CAD",
    "GBP_NZD",
    "AUD_JPY",
    "AUD_CHF",
    "AUD_CAD",
    "AUD_NZD",
    "CAD_JPY",
    "CAD_CHF",
    "CHF_JPY",
    "NZD_JPY",
    "NZD_CHF",
    "NZD_CAD",
    # Exotics
    "USD_SGD",
    "USD_HKD",
    "USD_MXN",
    "USD_ZAR",
    "USD_TRY",
    "USD_SEK",
    "USD_NOK",
    "USD_DKK",
    "USD_PLN",
    "USD_HUF",
    "USD_CZK",
    "USD_THB",
    "EUR_SEK",
    "EUR_NOK",
    "EUR_DKK",
    "EUR_PLN",
    "EUR_HUF",
    "EUR_CZK",
    "EUR_TRY",
    "EUR_ZAR",
    "EUR_SGD",
    "GBP_SGD",
    "GBP_PLN",
    "GBP_ZAR",
]

# Metals
METALS = [
    "XAU_USD",  # Gold
    "XAG_USD",  # Silver
    "XPT_USD",  # Platinum
    "XPD_USD",  # Palladium
    "XAU_EUR",  # Gold/Euro
    "XAU_GBP",  # Gold/GBP
    "XAU_AUD",  # Gold/AUD
    "XAG_EUR",  # Silver/Euro
    "XCU_USD",  # Copper
]

# Indices
INDICES = [
    "US30_USD",    # Dow Jones
    "SPX500_USD",  # S&P 500
    "NAS100_USD",  # Nasdaq 100
    "US2000_USD",  # Russell 2000
    "UK100_GBP",   # FTSE 100
    "DE30_EUR",    # DAX
    "FR40_EUR",    # CAC 40
    "EU50_EUR",    # Euro Stoxx 50
    "JP225_USD",   # Nikkei 225
    "AU200_AUD",   # ASX 200
    "HK33_HKD",    # Hang Seng
    "CN50_USD",    # China A50
    "SG30_SGD",    # Singapore 30
    "TWIX_USD",    # Taiwan Index
]

# Energies
ENERGIES = [
    "WTICO_USD",   # WTI Crude Oil
    "BCO_USD",     # Brent Crude Oil
    "NATGAS_USD",  # Natural Gas
]

# Crypto
CRYPTO_ASSETS = [
    "BTC_USD",   # Bitcoin
    "ETH_USD",   # Ethereum
    "LTC_USD",   # Litecoin
    "BCH_USD",   # Bitcoin Cash
    "LINK_USD",  # Chainlink
]

# Bonds
BONDS = [
    "USB02Y_USD",  # US 2-Year T-Note
    "USB05Y_USD",  # US 5-Year T-Note
    "USB10Y_USD",  # US 10-Year T-Note
    "USB30Y_USD",  # US 30-Year T-Bond
    "UK10YB_GBP",  # UK 10-Year Gilt
    "DE10YB_EUR",  # German 10-Year Bund
]

# Convenience groups

def all_market_instruments() -> list[str]:
    """All instruments Blueprint can scan."""
    return sorted(set(
        FOREX_PAIRS + METALS + INDICES + ENERGIES + CRYPTO_ASSETS + BONDS
    ))
