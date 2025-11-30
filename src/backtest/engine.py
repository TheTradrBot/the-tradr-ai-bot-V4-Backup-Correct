"""
Backtest Engine for Blueprint Trader AI - v4 Spec.

Implements realistic walk-forward backtesting with:
- Accurate trade accounting (partial exits, R multiples)
- No look-ahead bias
- Conservative exit logic (SL checked first)
- 5ers challenge simulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class BacktestTrade:
    """Represents a single trade in backtest."""
    symbol: str
    direction: str
    entry_date: datetime
    entry_price: float
    stop_loss: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    
    partial_exits: List[Dict] = field(default_factory=list)
    
    risk: float = 0.0
    
    confluence_score: int = 0
    
    @property
    def total_r(self) -> float:
        """Calculate total R for this trade including partials."""
        if not self.partial_exits:
            if self.exit_price is None:
                return 0.0
            if self.direction == "bullish":
                return (self.exit_price - self.entry_price) / self.risk if self.risk > 0 else 0
            else:
                return (self.entry_price - self.exit_price) / self.risk if self.risk > 0 else 0
        
        total = 0.0
        for exit in self.partial_exits:
            total += exit.get("r_multiple", 0) * exit.get("portion", 0.333)
        return total
    
    @property
    def is_winner(self) -> bool:
        """Trade is a winner if total R > 0."""
        return self.total_r > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": str(self.entry_date),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "exit_date": str(self.exit_date) if self.exit_date else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "risk": self.risk,
            "total_r": self.total_r,
            "is_winner": self.is_winner,
            "confluence_score": self.confluence_score,
        }


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    symbol: str
    period_start: datetime
    period_end: datetime
    
    trades: List[BacktestTrade] = field(default_factory=list)
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.is_winner)
    
    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if not t.is_winner)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def total_r(self) -> float:
        return sum(t.total_r for t in self.trades)
    
    @property
    def avg_r_per_trade(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_r / self.total_trades
    
    @property
    def net_return_pct(self) -> float:
        """Net return assuming 1% risk per trade."""
        return self.total_r * 1.0
    
    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown in R."""
        if not self.trades:
            return 0.0
        
        equity = [0.0]
        for t in self.trades:
            equity.append(equity[-1] + t.total_r)
        
        peak = 0.0
        max_dd = 0.0
        
        for e in equity:
            if e > peak:
                peak = e
            dd = peak - e
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    @property
    def exit_breakdown(self) -> Dict[str, int]:
        """Count of trades by exit reason."""
        breakdown = {}
        for t in self.trades:
            reason = t.exit_reason or "Unknown"
            breakdown[reason] = breakdown.get(reason, 0) + 1
        return breakdown
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "period_start": str(self.period_start),
            "period_end": str(self.period_end),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 1),
            "total_r": round(self.total_r, 2),
            "avg_r_per_trade": round(self.avg_r_per_trade, 2),
            "net_return_pct": round(self.net_return_pct, 1),
            "max_drawdown_r": round(self.max_drawdown, 2),
            "exit_breakdown": self.exit_breakdown,
        }


@dataclass 
class FiversChallengeResult:
    """Results for a 5ers challenge simulation."""
    start_date: datetime
    end_date: Optional[datetime] = None
    
    starting_balance: float = 10000.0
    final_balance: float = 10000.0
    
    target_pct: float = 8.0
    max_daily_loss_pct: float = 5.0
    max_total_loss_pct: float = 10.0
    
    passed: bool = False
    failed: bool = False
    fail_reason: str = ""
    
    days_to_pass: int = 0
    profitable_days: int = 0
    
    trades: List[BacktestTrade] = field(default_factory=list)
    
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_return_pct(self) -> float:
        return ((self.final_balance - self.starting_balance) / self.starting_balance) * 100
    
    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.is_winner)
        return (winners / len(self.trades)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date) if self.end_date else None,
            "starting_balance": self.starting_balance,
            "final_balance": round(self.final_balance, 2),
            "target_pct": self.target_pct,
            "total_return_pct": round(self.total_return_pct, 2),
            "passed": self.passed,
            "failed": self.failed,
            "fail_reason": self.fail_reason,
            "days_to_pass": self.days_to_pass,
            "profitable_days": self.profitable_days,
            "total_trades": len(self.trades),
            "win_rate": round(self.win_rate, 1),
        }


def parse_period_string(period: str) -> Tuple[datetime, datetime]:
    """
    Parse human-friendly period string to datetime range.
    
    Examples:
        "Jan 2024 - Dec 2024"
        "Jan 24 - Dec 24"
        "2024"
        "Q1 2024"
    """
    period = period.strip()
    
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    
    if "-" in period:
        parts = [p.strip() for p in period.split("-")]
        if len(parts) == 2:
            start_str, end_str = parts
            
            start_parts = start_str.lower().split()
            end_parts = end_str.lower().split()
            
            if len(start_parts) >= 2:
                start_month = month_map.get(start_parts[0][:3], 1)
                start_year = int(start_parts[1])
                if start_year < 100:
                    start_year += 2000
            else:
                start_month = 1
                start_year = int(start_str)
            
            if len(end_parts) >= 2:
                end_month = month_map.get(end_parts[0][:3], 12)
                end_year = int(end_parts[1])
                if end_year < 100:
                    end_year += 2000
            else:
                end_month = 12
                end_year = int(end_str)
            
            start_date = datetime(start_year, start_month, 1)
            
            if end_month == 12:
                end_date = datetime(end_year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(end_year, end_month + 1, 1) - timedelta(days=1)
            
            return start_date, end_date
    
    if period.isdigit() and len(period) == 4:
        year = int(period)
        return datetime(year, 1, 1), datetime(year, 12, 31)
    
    if period.lower().startswith("q"):
        parts = period.split()
        quarter = int(period[1])
        year = int(parts[1]) if len(parts) > 1 else datetime.now().year
        
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        
        return (
            datetime(year, start_month, 1),
            datetime(year, end_month + 1, 1) - timedelta(days=1) if end_month < 12 
            else datetime(year, 12, 31)
        )
    
    return datetime(2024, 1, 1), datetime(2024, 12, 31)


def simulate_trade_exit(
    trade: BacktestTrade,
    candles: List[Dict],
    start_idx: int,
) -> BacktestTrade:
    """
    Simulate trade exit using walk-forward candle data.
    
    Exit logic:
    1. Check SL first (conservative)
    2. Check TP1, then TP2, then TP3
    3. Handle partial exits
    """
    if not candles or start_idx >= len(candles):
        return trade
    
    tp1_hit = False
    tp2_hit = False
    trailing_sl = trade.stop_loss
    
    for i in range(start_idx, len(candles)):
        candle = candles[i]
        high = candle["high"]
        low = candle["low"]
        time = candle.get("time", datetime.now())
        
        if trade.direction == "bullish":
            if low <= trailing_sl:
                trade.exit_date = time
                trade.exit_price = trailing_sl
                
                if tp1_hit:
                    trade.exit_reason = "TP1+Trail"
                else:
                    trade.exit_reason = "SL"
                    trade.partial_exits = [{
                        "level": "SL",
                        "price": trailing_sl,
                        "r_multiple": -1.0,
                        "portion": 1.0,
                    }]
                return trade
            
            if not tp1_hit and trade.tp1 and high >= trade.tp1:
                tp1_hit = True
                trailing_sl = trade.entry_price
                
                trade.partial_exits.append({
                    "level": "TP1",
                    "price": trade.tp1,
                    "r_multiple": (trade.tp1 - trade.entry_price) / trade.risk if trade.risk > 0 else 2.0,
                    "portion": 0.333,
                })
            
            if tp1_hit and not tp2_hit and trade.tp2 and high >= trade.tp2:
                tp2_hit = True
                
                trade.partial_exits.append({
                    "level": "TP2",
                    "price": trade.tp2,
                    "r_multiple": (trade.tp2 - trade.entry_price) / trade.risk if trade.risk > 0 else 3.0,
                    "portion": 0.333,
                })
            
            if tp2_hit and trade.tp3 and high >= trade.tp3:
                trade.exit_date = time
                trade.exit_price = trade.tp3
                trade.exit_reason = "TP3"
                
                trade.partial_exits.append({
                    "level": "TP3",
                    "price": trade.tp3,
                    "r_multiple": (trade.tp3 - trade.entry_price) / trade.risk if trade.risk > 0 else 4.5,
                    "portion": 0.334,
                })
                return trade
        
        else:
            if high >= trailing_sl:
                trade.exit_date = time
                trade.exit_price = trailing_sl
                
                if tp1_hit:
                    trade.exit_reason = "TP1+Trail"
                else:
                    trade.exit_reason = "SL"
                    trade.partial_exits = [{
                        "level": "SL",
                        "price": trailing_sl,
                        "r_multiple": -1.0,
                        "portion": 1.0,
                    }]
                return trade
            
            if not tp1_hit and trade.tp1 and low <= trade.tp1:
                tp1_hit = True
                trailing_sl = trade.entry_price
                
                trade.partial_exits.append({
                    "level": "TP1",
                    "price": trade.tp1,
                    "r_multiple": (trade.entry_price - trade.tp1) / trade.risk if trade.risk > 0 else 2.0,
                    "portion": 0.333,
                })
            
            if tp1_hit and not tp2_hit and trade.tp2 and low <= trade.tp2:
                tp2_hit = True
                
                trade.partial_exits.append({
                    "level": "TP2",
                    "price": trade.tp2,
                    "r_multiple": (trade.entry_price - trade.tp2) / trade.risk if trade.risk > 0 else 3.0,
                    "portion": 0.333,
                })
            
            if tp2_hit and trade.tp3 and low <= trade.tp3:
                trade.exit_date = time
                trade.exit_price = trade.tp3
                trade.exit_reason = "TP3"
                
                trade.partial_exits.append({
                    "level": "TP3",
                    "price": trade.tp3,
                    "r_multiple": (trade.entry_price - trade.tp3) / trade.risk if trade.risk > 0 else 4.5,
                    "portion": 0.334,
                })
                return trade
    
    if trade.exit_date is None:
        final_candle = candles[-1]
        trade.exit_date = final_candle.get("time", datetime.now())
        trade.exit_price = final_candle["close"]
        trade.exit_reason = "Period End"
    
    return trade


def simulate_5ers_challenge(
    trades: List[BacktestTrade],
    start_date: datetime,
    step: int = 1,
    starting_balance: float = 10000.0,
    risk_per_trade_pct: float = 0.75,
) -> FiversChallengeResult:
    """
    Simulate a 5ers challenge with the given trades.
    
    Step 1: 8% target
    Step 2: 5% target
    Funded: 5% monthly target
    """
    targets = {1: 8.0, 2: 5.0}
    target_pct = targets.get(step, 5.0)
    
    result = FiversChallengeResult(
        start_date=start_date,
        starting_balance=starting_balance,
        target_pct=target_pct,
    )
    
    sorted_trades = sorted(trades, key=lambda t: t.entry_date)
    
    balance = starting_balance
    daily_start_balance = starting_balance
    current_day = None
    max_balance = starting_balance
    
    for trade in sorted_trades:
        if trade.entry_date < start_date:
            continue
        
        trade_day = trade.entry_date.date()
        
        if current_day is None or trade_day != current_day:
            if current_day is not None:
                daily_pnl = balance - daily_start_balance
                result.daily_pnl[str(current_day)] = daily_pnl
                
                if daily_pnl > 0:
                    result.profitable_days += 1
            
            current_day = trade_day
            daily_start_balance = max(balance, max_balance)
        
        risk_amount = balance * (risk_per_trade_pct / 100)
        trade_pnl = trade.total_r * risk_amount
        balance += trade_pnl
        
        result.trades.append(trade)
        
        if balance > max_balance:
            max_balance = balance
        
        daily_loss = daily_start_balance - balance
        daily_loss_pct = (daily_loss / daily_start_balance) * 100
        
        if daily_loss_pct >= result.max_daily_loss_pct:
            result.failed = True
            result.fail_reason = f"Daily loss limit breached ({daily_loss_pct:.1f}%)"
            result.end_date = trade.exit_date or trade.entry_date
            result.final_balance = balance
            return result
        
        total_loss = starting_balance - balance
        total_loss_pct = (total_loss / starting_balance) * 100
        
        if total_loss_pct >= result.max_total_loss_pct:
            result.failed = True
            result.fail_reason = f"Max drawdown breached ({total_loss_pct:.1f}%)"
            result.end_date = trade.exit_date or trade.entry_date
            result.final_balance = balance
            return result
        
        return_pct = ((balance - starting_balance) / starting_balance) * 100
        if return_pct >= target_pct:
            if result.profitable_days >= 3:
                result.passed = True
                result.end_date = trade.exit_date or trade.entry_date
                result.final_balance = balance
                
                days_elapsed = (result.end_date - start_date).days
                result.days_to_pass = max(1, days_elapsed)
                return result
    
    result.final_balance = balance
    result.end_date = sorted_trades[-1].exit_date if sorted_trades else start_date
    
    return_pct = ((balance - starting_balance) / starting_balance) * 100
    if return_pct >= target_pct and result.profitable_days >= 3:
        result.passed = True
        result.days_to_pass = (result.end_date - start_date).days
    
    return result


def format_challenge_result(result: FiversChallengeResult) -> str:
    """Format 5ers challenge result for Discord output."""
    emoji = "✅" if result.passed else ("❌" if result.failed else "⏳")
    
    lines = [
        f"**5ers Challenge Simulation** {emoji}",
        f"",
        f"**Status:** {'PASSED' if result.passed else ('FAILED' if result.failed else 'In Progress')}",
    ]
    
    if result.passed:
        lines.append(f"**Days to Pass:** {result.days_to_pass}")
    elif result.failed:
        lines.append(f"**Fail Reason:** {result.fail_reason}")
    
    lines.extend([
        f"",
        f"**Starting Balance:** ${result.starting_balance:,.0f}",
        f"**Final Balance:** ${result.final_balance:,.2f}",
        f"**Return:** {result.total_return_pct:+.1f}%",
        f"**Target:** {result.target_pct}%",
        f"",
        f"**Trades:** {len(result.trades)}",
        f"**Win Rate:** {result.win_rate:.1f}%",
        f"**Profitable Days:** {result.profitable_days}",
    ])
    
    return "\n".join(lines)


def format_backtest_result(result: BacktestResult) -> str:
    """Format backtest result for Discord output."""
    emoji = "✅" if result.total_r > 0 else "❌"
    
    lines = [
        f"**Backtest Results** {emoji}",
        f"",
        f"**{result.symbol}** | {result.period_start.strftime('%b %Y')} - {result.period_end.strftime('%b %Y')}",
        f"",
        f"**Total Trades:** {result.total_trades}",
        f"**Win Rate:** {result.win_rate:.1f}%",
        f"**Winners:** {result.winning_trades} | **Losers:** {result.losing_trades}",
        f"",
        f"**Total R:** {result.total_r:+.2f}R",
        f"**Avg R/Trade:** {result.avg_r_per_trade:+.2f}R",
        f"**Net Return:** {result.net_return_pct:+.1f}% (at 1% risk)",
        f"**Max Drawdown:** {result.max_drawdown:.2f}R",
        f"",
        f"**Exit Breakdown:**",
    ]
    
    for reason, count in result.exit_breakdown.items():
        lines.append(f"  {reason}: {count}")
    
    return "\n".join(lines)
