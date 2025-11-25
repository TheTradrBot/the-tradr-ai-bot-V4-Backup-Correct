# Overview

Blueprint Trader AI is an automated trading signal bot that scans multiple markets (forex, metals, indices, energies, crypto) using a multi-timeframe confluence strategy. The bot identifies high-probability trading opportunities by analyzing 7 technical pillars across monthly, weekly, daily, and 4-hour timeframes. It integrates with Discord for signal delivery and uses OANDA's practice API for market data.

The system operates on a scan-and-signal model: it periodically scans instruments for confluence setups, sends signals to Discord channels, tracks active trades, and monitors for take-profit/stop-loss events.

# User Preferences

Preferred communication style: Simple, everyday language.

# Recent Changes

**November 25, 2024 - Strategy Optimization**
- Lowered confluence threshold from 6/7 to 5/7 for active trades (improves trade frequency while maintaining quality)
- Relaxed HTF bias to accept Weekly trend leadership even when Daily is "mixed" (allows pullback entries)
- Widened liquidity tolerance from 1.0% to 2.5% for proximity checks (more realistic detection range)
- Extended liquidity sweep detection from 1 to 5 candles with 100-candle lookback window (catches more sweep events)
- **Impact**: EUR_USD backtest improved from 0 to 2 trades, XAU_USD now achieves ACTIVE status (5/7 confluence)

# System Architecture

## Core Design Pattern

**Multi-Timeframe Confluence Engine**: The strategy evaluates 7 technical criteria (HTF bias, location, Fibonacci levels, liquidity, structure, 4H confirmation, risk/reward) across 4 timeframes. Trades require 5/7 criteria met for "active" status (configurable via SIGNAL_MODE), with an adapter pattern to handle strategy signature changes between versions.

**Event-Driven Discord Bot**: Uses discord.py with scheduled tasks for automated scanning at configurable intervals (default 4 hours). The bot maintains in-memory trade state and posts to dedicated channels (scans, trades, updates).

**Stateless Data Layer**: All market data fetched on-demand from OANDA REST API. No local caching or database persistence - the system queries fresh OHLCV data for each scan operation.

## Trading Logic Components

**Strategy Module (`strategy.py`)**: Implements the 7-pillar confluence logic with dataclass-based results. Scans can be run per-asset or grouped by market type. Includes trend inference using EMAs and structure detection for swing highs/lows.

**Backtesting Engine (`backtest.py`)**: Walk-forward simulation engine that replays historical candles to test strategy performance. Includes signature detection adapter to work with evolving strategy function parameters.

**Indicator Library (`indicators.py`)**: Provides EMA and RSI calculations. Designed for minimal dependencies - pure Python implementations without pandas/numpy requirements.

## Data Flow

1. **Scan Trigger**: Discord task scheduler or manual command initiates scan
2. **Data Fetch**: `data.py` queries OANDA for M/W/D/H4 candles per instrument
3. **Analysis**: `strategy.py` evaluates all 7 confluence criteria and assigns scores
4. **Formatting**: `formatting.py` converts ScanResult objects into readable Discord messages
5. **Delivery**: Bot posts formatted signals to appropriate Discord channels
6. **Trade Tracking**: Active trades (5/7+ confluence in standard mode) stored in `trade_state.py` in-memory registry
7. **Monitoring**: Periodic checks evaluate registered trades for TP/SL hits using latest H4 data

## Configuration Management

**Two-Tier Config**: Sensitive credentials (Discord token, OANDA keys) stored in Replit Secrets (environment variables). Public configuration (channel IDs, instrument lists, scan intervals, signal mode) defined in `config.py`.

**Signal Modes**: Strategy strictness controlled via `SIGNAL_MODE` setting - "standard" for fewer high-quality signals, "aggressive" for more permissive scanning (feature flag for future use).

## Architecture Decisions & Rationale

**In-Memory State vs Database**: 
- **Chosen**: In-memory dictionaries for active trades
- **Rationale**: Simpler deployment, faster reads, acceptable for signal bot use case where state reset on restart is tolerable
- **Tradeoff**: Lose trade history on crash/restart, can't analyze historical performance without separate logging

**OANDA Practice API**:
- **Chosen**: OANDA fxPractice endpoint for data
- **Rationale**: Free practice accounts, reliable forex/metals/indices data, straightforward REST API
- **Alternative Considered**: Crypto-native APIs (Coinbase, Binance) for crypto assets
- **Limitation**: May need multi-source architecture if expanding crypto coverage

**Discord as UI**:
- **Chosen**: Discord bot with channel-based routing
- **Rationale**: Low friction for traders already using Discord, rich formatting support, webhook simplicity
- **Tradeoff**: Limited interactivity compared to web dashboard, channel organization must be maintained manually

**No ML/AI Components**:
- **Chosen**: Pure technical analysis rules-based strategy
- **Rationale**: Deterministic behavior, easier to debug and backtest, transparent logic for users
- **Note**: "AI" in name refers to automated intelligence of multi-timeframe confluence, not machine learning

**Backtest Time Simulation**:
- **Chosen**: Walk-forward with exact timestamp matching for entry/exit simulation
- **Rationale**: More realistic than vectorized backtesting, captures actual candle-close timing
- **Limitation**: Slower than vectorized approaches, no tick-level precision (uses candle closes only)

# External Dependencies

## Third-Party Services

**Discord API**: 
- Purpose: Bot communication platform, signal delivery, user commands
- Library: `discord.py` (official Python wrapper)
- Credentials: `DISCORD_BOT_TOKEN` environment variable
- Channels: Three configured channels for scans, trades, and updates

**OANDA v20 REST API**:
- Purpose: OHLCV market data for forex, metals, indices, energies
- Endpoint: `https://api-fxpractice.oanda.com`
- Credentials: `OANDA_API_KEY` and `OANDA_ACCOUNT_ID` environment variables
- Data: Mid-price candles at M/W/D/H4 granularities

## Instrument Coverage

**Forex Pairs**: Major and cross pairs (defined in `FOREX_PAIRS` list in config.py)
**Metals**: Gold, Silver (XAU_USD, XAG_USD style instruments)
**Indices**: Stock market indices via CFDs
**Energies**: Oil and natural gas instruments
**Crypto**: Bitcoin and major altcoins (BTC_USD, ETH_USD format)

## Python Dependencies

- `requests`: HTTP client for OANDA API calls
- `discord.py`: Discord bot framework with task scheduling
- Standard library: `datetime`, `dataclasses`, `typing`, `inspect` for core functionality

## Infrastructure

**Hosting**: Designed for Replit deployment (references Replit Secrets for environment variables)
**Storage**: No database - all state ephemeral in process memory
**Scheduling**: Discord.py task loop for periodic scans, no external cron required