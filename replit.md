# Blueprint Trader AI

## Overview
Blueprint Trader AI is an automated trading signal bot designed to identify high-probability trading opportunities across multiple markets (forex, metals, indices, energies, crypto). It utilizes the HTF Confluence Strategy v4, analyzing 10 confluence pillars across various timeframes (monthly, weekly, daily, 4-hour). The bot integrates with Discord for signal delivery and uses OANDA's practice API for market data. It includes a confluence scoring system, a 5%ers challenge simulation, and historical data infrastructure for backtesting from 2003 with improved accuracy. The project aims to provide realistic trade simulations and signal generation, particularly for proprietary trading firm challenges like The 5%ers 10K High Stakes account.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Core Components
The system is built around a modular architecture:
- **Strategy Engine (`strategy.py`)**: Evaluates 7 confluence pillars across multiple timeframes.
- **Backtest Engine (`backtest.py`)**: Performs walk-forward simulations with conservative exit logic.
- **Data Layer (`data.py`)**: Handles OANDA v20 API integration and data retrieval.
- **Cache System (`cache.py`)**: Provides TTL-based, thread-safe in-memory caching for API responses.
- **Formatting (`formatting.py`)**: Manages Discord message formatting for scans, summaries, and backtest results.
- **Bot (`main.py`)**: Implements Discord slash commands, an autoscan loop, and trade tracking.
- **5%ers Risk Model**: Configured for The 5%ers 10K High Stakes account, including specific risk per trade, max daily loss, max total drawdown, and profit targets. Position sizing is dynamically calculated based on account size, risk percentage, stop loss distance, and pip value.
- **Trade State Tracking (`trade_state.py`)**: Persists trade notifications and entry times to prevent duplicate Discord messages and ensure accurate updates across restarts.

### UI/UX Decisions
- **Discord Integration**: Trade signals, updates, and backtest results are delivered through formatted Discord embeds.
- **Clean Output**: Autoscan output is streamlined, and detailed single asset scans provide specific setup information.
- **Emoji Usage**: Emojis are used for directional cues (🟢/🔴) to enhance readability.

### Technical Implementations
- **Confluence Strategy**: Based on 7 pillars: HTF Bias, Location, Fibonacci, Liquidity, Structure, Confirmation, and R:R.
- **Trade Status Levels**: Signals are categorized as `ACTIVE`, `WATCHING`, or `SCAN`.
- **Live Price Integration**: Trade entries and TP/SL monitoring use live OANDA prices, preventing reliance on stale data.
- **Strategy Optimization Framework**: Includes modules for CSV data loading (`data_loader.py`), a parameterized strategy engine (`strategy_core.py`), an enhanced backtest engine (`backtest_engine.py`), an optimizer (`optimizer.py`) for parameter tuning, and reporting (`report.py`).
- **Dependency Management**: Uses `pyproject.toml` and `uv.lock` for consistent dependency versions.

### Feature Specifications
- **Automated Scanning**: Scans multiple markets at configurable intervals.
- **Signal Generation**: Identifies high-probability trading setups.
- **Risk Management**: Implements 5%ers challenge-specific risk rules and dynamic position sizing.
- **Backtesting**: Comprehensive backtesting with multi-step challenge simulation, daily PnL tracking, and rule breach detection.
- **Discord Commands**: Provides various commands for scanning, trading analysis, and system management.
- **Anti-Spam Measures**: Delayed first autoscan and improved channel routing to minimize Discord message spam.

## External Dependencies

### Services
- **Discord API**: For bot communication and message delivery.
- **OANDA v20 API**: For real-time and historical market data (practice endpoint).

### Environment Variables
- `DISCORD_BOT_TOKEN`: Discord bot authentication token.
- `OANDA_API_KEY`: OANDA API key.
- `OANDA_ACCOUNT_ID`: OANDA account identifier.
- `USE_OPTIMIZED_STRATEGY`: Toggle for optimized strategy parameters.
- `DISCORD_SCAN_CHANNEL_ID`, `DISCORD_TRADES_CHANNEL_ID`, `DISCORD_TRADE_UPDATES_CHANNEL_ID`: Discord channel IDs for routing specific messages.

### Python Dependencies
- `discord-py`: Discord bot framework.
- `pandas`: Data processing and analysis.
- `requests`: HTTP client for API interactions.
- `uv`: Package installer and resolver.