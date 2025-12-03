
"""
MT5 Live Validation Module for Blueprint Trader AI.

Connects to MT5 demo account, executes trades using the same strategy logic
as backtests, and produces detailed comparison reports.
"""

import MetaTrader5 as mt5
import pandas as pd
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

from strategy_core import (
    StrategyParams,
    generate_signals,
    get_default_params,
    _infer_trend,
    _pick_direction_from_bias,
    compute_confluence,
)
from data import get_ohlcv
from backtest import run_backtest
from position_sizing import calculate_position_size_5ers
from config import ACCOUNT_SIZE, RISK_PER_TRADE_PCT, CONTRACT_SPECS


class MT5Validator:
    """Validates live MT5 execution against backtests."""
    
    def __init__(self, server: str, login: int, password: str):
        self.server = server
        self.login = login
        self.password = password
        self.connected = False
        self.symbol_map = {}  # Maps our symbols to broker symbols
        self.test_start_time = None
        self.live_trades: List[Dict] = []
        self.logs_dir = Path("/tmp/replit_test_logs")
        self.plots_dir = Path("/tmp/plots")
        
        # Create directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        print(f"\n{'='*70}")
        print("MT5 CONNECTION")
        print(f"{'='*70}")
        
        if not mt5.initialize():
            print(f"❌ MT5 initialize() failed: {mt5.last_error()}")
            return False
        
        print(f"✓ MT5 terminal initialized")
        
        authorized = mt5.login(self.login, password=self.password, server=self.server)
        
        if not authorized:
            print(f"❌ Login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        self.connected = True
        print(f"✓ Connected to {self.server}")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"\n📊 Account Information:")
            print(f"   Login: {account_info.login}")
            print(f"   Balance: ${account_info.balance:,.2f}")
            print(f"   Equity: ${account_info.equity:,.2f}")
            print(f"   Margin: ${account_info.margin:,.2f}")
            print(f"   Free Margin: ${account_info.margin_free:,.2f}")
            print(f"   Leverage: 1:{account_info.leverage}")
        
        # Get server time
        server_time = datetime.fromtimestamp(mt5.symbol_info_tick("EURUSD").time, tz=timezone.utc)
        print(f"   Server Time (UTC): {server_time}")
        
        return True
    
    def map_symbols(self, our_symbols: List[str]) -> Dict[str, str]:
        """Map our symbol format to broker's format."""
        print(f"\n{'='*70}")
        print("SYMBOL MAPPING")
        print(f"{'='*70}")
        
        all_symbols = mt5.symbols_get()
        broker_symbols = [s.name for s in all_symbols] if all_symbols else []
        
        symbol_map = {}
        
        for our_sym in our_symbols:
            # Try common variations
            candidates = [
                our_sym.replace("_", ""),      # EUR_USD -> EURUSD
                our_sym.replace("_", "."),     # EUR_USD -> EUR.USD
                our_sym.replace("_", ""),      # XAU_USD -> XAUUSD
                our_sym,                       # Keep as is
            ]
            
            for candidate in candidates:
                if candidate in broker_symbols:
                    symbol_map[our_sym] = candidate
                    
                    # Get symbol info
                    info = mt5.symbol_info(candidate)
                    if info:
                        print(f"✓ {our_sym} -> {candidate}")
                        print(f"   Digits: {info.digits}")
                        print(f"   Point: {info.point}")
                        print(f"   Min Lot: {info.volume_min}")
                        print(f"   Max Lot: {info.volume_max}")
                        print(f"   Lot Step: {info.volume_step}")
                        print(f"   Spread: {info.spread}")
                    break
            
            if our_sym not in symbol_map:
                print(f"⚠ {our_sym} -> NOT FOUND")
        
        self.symbol_map = symbol_map
        return symbol_map
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for symbol."""
        broker_symbol = self.symbol_map.get(symbol, symbol)
        
        tick = mt5.symbol_info_tick(broker_symbol)
        info = mt5.symbol_info(broker_symbol)
        
        if not tick or not info:
            return {}
        
        return {
            "broker_symbol": broker_symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid,
            "spread_points": info.spread,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
            "digits": info.digits,
            "point": info.point,
            "min_lot": info.volume_min,
            "max_lot": info.volume_max,
            "lot_step": info.volume_step,
        }
    
    def execute_trade(self, symbol: str, direction: str, entry: float, sl: float, 
                     tp1: float, volume: float) -> Optional[Dict]:
        """Execute a market order with SL and TP."""
        broker_symbol = self.symbol_map.get(symbol, symbol)
        
        # Get current price
        tick = mt5.symbol_info_tick(broker_symbol)
        if not tick:
            print(f"❌ Failed to get tick data for {broker_symbol}")
            return None
        
        # Determine order type and price
        if direction == "bullish":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            requested_price = price
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            requested_price = price
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp1,
            "deviation": 20,
            "magic": 234000,
            "comment": "Blueprint MT5 Validator",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        start_time = time.time()
        result = mt5.order_send(request)
        execution_time_ms = (time.time() - start_time) * 1000
        
        if result is None:
            print(f"❌ Order send failed: {mt5.last_error()}")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order failed: {result.retcode} - {result.comment}")
            return None
        
        # Get account state after trade
        account_info = mt5.account_info()
        
        execution_data = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "server_time": datetime.fromtimestamp(result.time, tz=timezone.utc).isoformat(),
            "symbol": symbol,
            "broker_symbol": broker_symbol,
            "direction": direction,
            "order_type": "BUY" if direction == "bullish" else "SELL",
            "requested_price": requested_price,
            "executed_price": result.price,
            "volume": result.volume,
            "spread": tick.ask - tick.bid,
            "slippage": result.price - requested_price,
            "execution_time_ms": execution_time_ms,
            "order_id": result.order,
            "deal_id": result.deal,
            "stop_loss": sl,
            "take_profit": tp1,
            "reason": "OPEN",
            "balance": account_info.balance if account_info else 0,
            "equity": account_info.equity if account_info else 0,
            "margin": account_info.margin if account_info else 0,
        }
        
        print(f"\n✓ Order executed: {result.order}")
        print(f"   Symbol: {broker_symbol}")
        print(f"   Direction: {direction.upper()}")
        print(f"   Volume: {result.volume} lots")
        print(f"   Price: {result.price:.5f} (requested: {requested_price:.5f})")
        print(f"   Slippage: {execution_data['slippage']:.5f}")
        print(f"   Execution: {execution_time_ms:.1f}ms")
        
        return execution_data
    
    def close_position(self, position: Dict) -> Optional[Dict]:
        """Close an open position."""
        broker_symbol = position["broker_symbol"]
        volume = position["volume"]
        
        # Get current tick
        tick = mt5.symbol_info_tick(broker_symbol)
        if not tick:
            return None
        
        # Determine close order type (opposite of open)
        if position["direction"] == "bullish":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Blueprint Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        start_time = time.time()
        result = mt5.order_send(request)
        execution_time_ms = (time.time() - start_time) * 1000
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            account_info = mt5.account_info()
            
            return {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": position["symbol"],
                "executed_price": result.price,
                "execution_time_ms": execution_time_ms,
                "order_id": result.order,
                "balance": account_info.balance if account_info else 0,
                "reason": "MANUAL_CLOSE",
            }
        
        return None
    
    def run_test(self, symbols: List[str], target_trades: int = 2) -> Dict:
        """Run live validation test until N trades execute."""
        print(f"\n{'='*70}")
        print(f"STARTING LIVE VALIDATION TEST")
        print(f"{'='*70}")
        print(f"Target: {target_trades} trades")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Strategy: Same as backtest (Blueprint)")
        print(f"Risk per trade: {RISK_PER_TRADE_PCT*100:.2f}%")
        
        self.test_start_time = datetime.now(timezone.utc)
        params = get_default_params()
        
        # Map symbols
        self.map_symbols(symbols)
        
        trades_executed = 0
        open_positions = []
        
        while trades_executed < target_trades:
            print(f"\n--- Scan {trades_executed + 1} ---")
            
            for symbol in symbols:
                if symbol not in self.symbol_map:
                    continue
                
                # Get market data
                print(f"\nScanning {symbol}...")
                
                # Load OHLCV data (same as backtest)
                daily = get_ohlcv(symbol, timeframe="D", count=100, use_cache=False)
                weekly = get_ohlcv(symbol, timeframe="W", count=50, use_cache=False) or []
                monthly = get_ohlcv(symbol, timeframe="M", count=24, use_cache=False) or []
                h4 = get_ohlcv(symbol, timeframe="H4", count=100, use_cache=False) or []
                
                if not daily or len(daily) < 50:
                    print(f"  ⚠ Insufficient data")
                    continue
                
                # Run strategy logic (same as backtest)
                mn_trend = _infer_trend(monthly) if monthly else "mixed"
                wk_trend = _infer_trend(weekly) if weekly else "mixed"
                d_trend = _infer_trend(daily) if daily else "mixed"
                
                direction, _, _ = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
                
                flags, notes, trade_levels = compute_confluence(
                    monthly, weekly, daily, h4, direction, params
                )
                
                entry, sl, tp1, tp2, tp3, tp4, tp5 = trade_levels
                
                confluence_score = sum(1 for v in flags.values() if v)
                has_rr = flags.get("rr", False)
                
                if confluence_score >= params.min_confluence and has_rr and entry and sl and tp1:
                    print(f"  ✓ SIGNAL FOUND: {direction} (confluence: {confluence_score}/7)")
                    
                    # Calculate position size (same as backtest)
                    sizing = calculate_position_size_5ers(symbol, entry, sl)
                    lot_size = sizing["lot_size"]
                    
                    print(f"  Entry: {entry:.5f}, SL: {sl:.5f}, TP1: {tp1:.5f}")
                    print(f"  Lot size: {lot_size:.2f} (risk: ${sizing['risk_usd']:.2f})")
                    
                    # Execute trade
                    execution = self.execute_trade(symbol, direction, entry, sl, tp1, lot_size)
                    
                    if execution:
                        self.live_trades.append(execution)
                        open_positions.append(execution)
                        trades_executed += 1
                        
                        # Save log
                        self.save_execution_log(execution)
                        
                        if trades_executed >= target_trades:
                            break
                else:
                    print(f"  - No signal (confluence: {confluence_score}/7)")
            
            if trades_executed < target_trades:
                print(f"\nWaiting 15 minutes before next scan...")
                time.sleep(900)  # 15 minutes
        
        print(f"\n{'='*70}")
        print(f"TEST COMPLETE - {trades_executed} trades executed")
        print(f"{'='*70}")
        
        # Close all positions
        for pos in open_positions:
            print(f"\nClosing position: {pos['symbol']}")
            close_data = self.close_position(pos)
            if close_data:
                print(f"  ✓ Closed at {close_data['executed_price']:.5f}")
        
        return {
            "trades_executed": trades_executed,
            "live_trades": self.live_trades,
        }
    
    def save_execution_log(self, execution: Dict):
        """Save execution log to CSV and JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV log
        csv_path = self.logs_dir / f"execution_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=execution.keys())
            writer.writeheader()
            writer.writerow(execution)
        
        # JSON log
        json_path = self.logs_dir / f"execution_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(execution, f, indent=2)
        
        print(f"  📝 Logs saved: {csv_path.name}")
    
    def compare_with_backtest(self, symbol: str, backtest_period: str) -> str:
        """Compare live trades with backtest results."""
        print(f"\n{'='*70}")
        print("BACKTEST COMPARISON")
        print(f"{'='*70}")
        
        # Run fresh backtest
        print(f"Running fresh backtest for {symbol} ({backtest_period})...")
        backtest_result = run_backtest(symbol, backtest_period)
        backtest_trades = backtest_result.get("trades", [])
        
        print(f"Backtest: {len(backtest_trades)} trades")
        print(f"Live: {len(self.live_trades)} trades")
        
        # Build comparison report
        report = f"""
# MT5 LIVE VALIDATION REPORT
Generated: {datetime.now(timezone.utc).isoformat()}

## Test Configuration
- Server: {self.server}
- Account: {self.login}
- Symbol: {symbol}
- Backtest Period: {backtest_period}
- Strategy: Blueprint (same as backtest)
- Risk per Trade: {RISK_PER_TRADE_PCT*100:.2f}%

## Results Summary
- Live Trades: {len(self.live_trades)}
- Backtest Trades: {len(backtest_trades)}
- Match Rate: {len(self.live_trades) / max(len(backtest_trades), 1) * 100:.1f}%

## Live Trade Details
"""
        
        for i, trade in enumerate(self.live_trades, 1):
            report += f"\n### Live Trade {i}\n"
            report += f"- Timestamp: {trade['timestamp_utc']}\n"
            report += f"- Symbol: {trade['symbol']} ({trade['broker_symbol']})\n"
            report += f"- Direction: {trade['direction']}\n"
            report += f"- Entry: {trade['executed_price']:.5f}\n"
            report += f"- SL: {trade['stop_loss']:.5f}\n"
            report += f"- TP: {trade['take_profit']:.5f}\n"
            report += f"- Volume: {trade['volume']:.2f} lots\n"
            report += f"- Spread: {trade['spread']:.5f}\n"
            report += f"- Slippage: {trade['slippage']:.5f}\n"
            report += f"- Execution Time: {trade['execution_time_ms']:.1f}ms\n"
        
        # Save report
        report_path = self.logs_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Report saved: {report_path}")
        
        return report
    
    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("\n✓ Disconnected from MT5")


def run_mt5_validation(server: str, login: int, password: str, 
                       symbols: List[str], target_trades: int = 2):
    """Main entry point for MT5 validation."""
    validator = MT5Validator(server, login, password)
    
    try:
        # Connect
        if not validator.connect():
            return None
        
        # Run test
        result = validator.run_test(symbols, target_trades)
        
        # Generate comparison report
        for symbol in symbols:
            report = validator.compare_with_backtest(symbol, "Jan 2024 - Now")
            print(report)
        
        return result
        
    finally:
        validator.disconnect()


if __name__ == "__main__":
    # FTMO Demo credentials
    SERVER = "FTMO-Demo"
    LOGIN = 1512190437
    PASSWORD = "$c@mAU7F"
    
    # Test all forex pairs except exotics
    from config import FOREX_PAIRS
    
    print("Starting MT5 Validation Test...")
    print(f"This will test the Blueprint strategy on FTMO demo account")
    print(f"Target: 2 trades across major forex pairs")
    
    result = run_mt5_validation(
        server=SERVER,
        login=LOGIN,
        password=PASSWORD,
        symbols=FOREX_PAIRS[:10],  # Test first 10 major pairs
        target_trades=2
    )
    
    if result:
        print(f"\n✅ Validation complete!")
        print(f"Trades executed: {result['trades_executed']}")
        print(f"Logs saved to: /tmp/replit_test_logs/")
    else:
        print(f"\n❌ Validation failed")
