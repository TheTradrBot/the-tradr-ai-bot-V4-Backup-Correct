"""
Strategy Core Module for Blueprint Trader AI.

This module provides the single source of truth for trading rules,
used by both backtests and live scanning/Discord outputs.

The strategy is parameterized to allow optimization while staying
faithful to the Blueprint confluence-based approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any


@dataclass
class StrategyParams:
    """
    Strategy parameters that can be optimized.
    
    These control confluence thresholds, SL/TP ratios, filters, etc.
    """
    min_confluence: int = 2
    min_quality_factors: int = 1
    
    atr_sl_multiplier: float = 1.5
    atr_tp1_multiplier: float = 0.6
    atr_tp2_multiplier: float = 1.1
    atr_tp3_multiplier: float = 1.8
    
    fib_low: float = 0.382
    fib_high: float = 0.886
    
    structure_sl_lookback: int = 35
    liquidity_sweep_lookback: int = 12
    
    use_htf_filter: bool = True
    use_structure_filter: bool = True
    use_liquidity_filter: bool = True
    use_fib_filter: bool = True
    use_confirmation_filter: bool = True
    
    require_htf_alignment: bool = False
    require_confirmation_for_active: bool = True
    require_rr_for_active: bool = True
    
    min_rr_ratio: float = 1.0
    risk_per_trade_pct: float = 1.0
    
    cooldown_bars: int = 0
    max_open_trades: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "min_confluence": self.min_confluence,
            "min_quality_factors": self.min_quality_factors,
            "atr_sl_multiplier": self.atr_sl_multiplier,
            "atr_tp1_multiplier": self.atr_tp1_multiplier,
            "atr_tp2_multiplier": self.atr_tp2_multiplier,
            "atr_tp3_multiplier": self.atr_tp3_multiplier,
            "fib_low": self.fib_low,
            "fib_high": self.fib_high,
            "structure_sl_lookback": self.structure_sl_lookback,
            "liquidity_sweep_lookback": self.liquidity_sweep_lookback,
            "use_htf_filter": self.use_htf_filter,
            "use_structure_filter": self.use_structure_filter,
            "use_liquidity_filter": self.use_liquidity_filter,
            "use_fib_filter": self.use_fib_filter,
            "use_confirmation_filter": self.use_confirmation_filter,
            "require_htf_alignment": self.require_htf_alignment,
            "require_confirmation_for_active": self.require_confirmation_for_active,
            "require_rr_for_active": self.require_rr_for_active,
            "min_rr_ratio": self.min_rr_ratio,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "cooldown_bars": self.cooldown_bars,
            "max_open_trades": self.max_open_trades,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


@dataclass
class Signal:
    """Represents a trading signal/setup."""
    symbol: str
    direction: str  # "bullish" or "bearish"
    bar_index: int
    timestamp: Any
    
    confluence_score: int = 0
    quality_factors: int = 0
    
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    
    is_active: bool = False
    is_watching: bool = False
    
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Trade:
    """Represents a completed trade for backtest analysis."""
    symbol: str
    direction: str
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    
    risk: float = 0.0
    reward: float = 0.0
    rr: float = 0.0
    
    is_winner: bool = False
    exit_reason: str = ""
    
    confluence_score: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "risk": self.risk,
            "reward": self.reward,
            "rr": self.rr,
            "is_winner": self.is_winner,
            "exit_reason": self.exit_reason,
            "confluence_score": self.confluence_score,
        }


def compute_confluence(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Tuple[Dict[str, bool], Dict[str, str], Tuple]:
    """
    Compute confluence flags for a given setup.
    
    Uses the same core logic as strategy.py but with parameterization.
    
    Args:
        monthly_candles: Monthly OHLCV data
        weekly_candles: Weekly OHLCV data
        daily_candles: Daily OHLCV data
        h4_candles: 4H OHLCV data
        direction: Trade direction ("bullish" or "bearish")
        params: Strategy parameters (uses defaults if None)
    
    Returns:
        Tuple of (flags dict, notes dict, trade_levels tuple)
    """
    if params is None:
        params = StrategyParams()
    
    price = daily_candles[-1]["close"] if daily_candles else float("nan")
    
    mn_trend = _infer_trend(monthly_candles) if monthly_candles else "mixed"
    wk_trend = _infer_trend(weekly_candles) if weekly_candles else "mixed"
    d_trend = _infer_trend(daily_candles) if daily_candles else "mixed"
    _, htf_note_text, htf_ok = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
    
    if params.use_htf_filter:
        loc_note, loc_ok = _location_context(
            monthly_candles, weekly_candles, daily_candles, price, direction
        )
    else:
        loc_note, loc_ok = "Location filter disabled", True
    
    if params.use_fib_filter:
        fib_note, fib_ok = _fib_context(weekly_candles, daily_candles, direction, price)
    else:
        fib_note, fib_ok = "Fib filter disabled", True
    
    if params.use_liquidity_filter:
        liq_note, liq_ok = _daily_liquidity_context(daily_candles, price)
    else:
        liq_note, liq_ok = "Liquidity filter disabled", True
    
    if params.use_structure_filter:
        struct_ok, struct_note = _structure_context(
            monthly_candles, weekly_candles, daily_candles, direction
        )
    else:
        struct_ok, struct_note = True, "Structure filter disabled"
    
    if params.use_confirmation_filter:
        conf_note, conf_ok = _h4_confirmation(h4_candles, direction, daily_candles)
    else:
        conf_note, conf_ok = "Confirmation filter disabled", True
    
    rr_note, rr_ok, entry, sl, tp1, tp2, tp3, tp4, tp5 = compute_trade_levels(
        daily_candles, direction, params
    )
    
    flags = {
        "htf_bias": htf_ok,
        "location": loc_ok,
        "fib": fib_ok,
        "liquidity": liq_ok,
        "structure": struct_ok,
        "confirmation": conf_ok,
        "rr": rr_ok,
    }
    
    notes = {
        "htf_bias": htf_note_text,
        "location": loc_note,
        "fib": fib_note,
        "liquidity": liq_note,
        "structure": struct_note,
        "confirmation": conf_note,
        "rr": rr_note,
    }
    
    trade_levels = (entry, sl, tp1, tp2, tp3, tp4, tp5)
    return flags, notes, trade_levels


def compute_trade_levels(
    daily_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Tuple[str, bool, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute entry, SL, and TP levels using parameterized logic.
    
    Args:
        daily_candles: Daily OHLCV data
        direction: Trade direction
        params: Strategy parameters
    
    Returns:
        Tuple of (note, is_valid, entry, sl, tp1, tp2, tp3, tp4, tp5)
    """
    if params is None:
        params = StrategyParams()
    
    if not daily_candles:
        return "R/R: no data.", False, None, None, None, None, None, None, None
    
    current = daily_candles[-1]["close"]
    atr = _atr(daily_candles, 14)
    
    if atr <= 0:
        return "R/R: ATR too small.", False, None, None, None, None, None, None, None
    
    leg = _find_last_swing_leg_for_fib(daily_candles, direction)
    structure_sl = _find_structure_sl(daily_candles, direction, lookback=params.structure_sl_lookback)
    
    if leg:
        lo, hi = leg
        span = hi - lo
        if span > 0:
            if direction == "bullish":
                gp_mid = hi - span * 0.618
                entry = current if abs(current - gp_mid) < atr * 0.3 else gp_mid
                
                base_sl = lo - atr * 0.5
                if structure_sl is not None:
                    sl = min(base_sl, structure_sl - atr * 0.4)
                else:
                    sl = base_sl
                
                risk = entry - sl
                
                if risk > 0:
                    tp1 = entry + risk * params.atr_tp1_multiplier
                    tp2 = entry + risk * params.atr_tp2_multiplier
                    tp3 = entry + risk * params.atr_tp3_multiplier
                    tp4 = entry + risk * 2.5
                    tp5 = entry + risk * 3.5
                    
                    note = f"R/R: Entry near {entry:.5f}, SL at {sl:.5f}"
                    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5
            else:
                gp_mid = lo + span * 0.618
                entry = current if abs(current - gp_mid) < atr * 0.3 else gp_mid
                
                base_sl = hi + atr * 0.5
                if structure_sl is not None:
                    sl = max(base_sl, structure_sl + atr * 0.4)
                else:
                    sl = base_sl
                
                risk = sl - entry
                
                if risk > 0:
                    tp1 = entry - risk * params.atr_tp1_multiplier
                    tp2 = entry - risk * params.atr_tp2_multiplier
                    tp3 = entry - risk * params.atr_tp3_multiplier
                    tp4 = entry - risk * 2.5
                    tp5 = entry - risk * 3.5
                    
                    note = f"R/R: Entry near {entry:.5f}, SL at {sl:.5f}"
                    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5
    
    entry = current
    sl_mult = params.atr_sl_multiplier
    
    if direction == "bullish":
        if structure_sl is not None:
            sl = min(entry - atr * sl_mult, structure_sl - atr * 0.4)
        else:
            sl = entry - atr * sl_mult
        risk = entry - sl
        tp1 = entry + risk * params.atr_tp1_multiplier
        tp2 = entry + risk * params.atr_tp2_multiplier
        tp3 = entry + risk * params.atr_tp3_multiplier
        tp4 = entry + risk * 2.5
        tp5 = entry + risk * 3.5
    else:
        if structure_sl is not None:
            sl = max(entry + atr * sl_mult, structure_sl + atr * 0.4)
        else:
            sl = entry + atr * sl_mult
        risk = sl - entry
        tp1 = entry - risk * params.atr_tp1_multiplier
        tp2 = entry - risk * params.atr_tp2_multiplier
        tp3 = entry - risk * params.atr_tp3_multiplier
        tp4 = entry - risk * 2.5
        tp5 = entry - risk * 3.5
    
    note = f"R/R: ATR+structure levels"
    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5


def generate_signals(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
) -> List[Signal]:
    """
    Generate trading signals from historical candles.
    
    This function walks through candles sequentially (no look-ahead bias)
    and generates signals based on the Blueprint strategy rules.
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data (derived from daily if not provided)
        weekly_candles: Optional weekly data (derived from daily if not provided)
        h4_candles: Optional 4H data (uses daily for confirmation if not provided)
    
    Returns:
        List of Signal objects
    """
    if params is None:
        params = StrategyParams()
    
    if len(candles) < 50:
        return []
    
    signals = []
    
    for i in range(50, len(candles)):
        daily_slice = candles[:i+1]
        
        weekly_slice = weekly_candles[:i//5+1] if weekly_candles else None
        monthly_slice = monthly_candles[:i//20+1] if monthly_candles else None
        h4_slice = h4_candles[:i*6+1] if h4_candles else None
        
        mn_trend = _infer_trend(monthly_slice) if monthly_slice else _infer_trend(daily_slice[-60:])
        wk_trend = _infer_trend(weekly_slice) if weekly_slice else _infer_trend(daily_slice[-20:])
        d_trend = _infer_trend(daily_slice[-10:])
        
        direction, _, _ = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
        
        flags, notes, trade_levels = compute_confluence(
            monthly_slice or [],
            weekly_slice or [],
            daily_slice,
            h4_slice or daily_slice[-20:],
            direction,
            params,
        )
        
        entry, sl, tp1, tp2, tp3, tp4, tp5 = trade_levels
        
        confluence_score = sum(1 for v in flags.values() if v)
        
        quality_factors = sum([
            flags.get("location", False),
            flags.get("fib", False),
            flags.get("liquidity", False),
            flags.get("structure", False),
            flags.get("htf_bias", False),
        ])
        
        has_rr = flags.get("rr", False)
        has_confirmation = flags.get("confirmation", False)
        
        is_active = False
        is_watching = False
        
        if confluence_score >= params.min_confluence and quality_factors >= params.min_quality_factors:
            if params.require_rr_for_active and not has_rr:
                is_watching = True
            elif params.require_confirmation_for_active and not has_confirmation:
                is_watching = True
            else:
                is_active = True
        elif confluence_score >= params.min_confluence - 1:
            is_watching = True
        
        if is_active or is_watching:
            candle = candles[i]
            timestamp = candle.get("time") or candle.get("timestamp") or candle.get("date")
            
            signal = Signal(
                symbol=symbol,
                direction=direction,
                bar_index=i,
                timestamp=timestamp,
                confluence_score=confluence_score,
                quality_factors=quality_factors,
                entry=entry,
                stop_loss=sl,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                is_active=is_active,
                is_watching=is_watching,
                flags=flags,
                notes=notes,
            )
            signals.append(signal)
    
    return signals


def simulate_trades(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
) -> List[Trade]:
    """
    Simulate trades through historical candles using the Blueprint strategy.
    
    This is a walk-forward simulation with no look-ahead bias.
    Uses the same logic as live trading but runs through historical data.
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data
        weekly_candles: Optional weekly data
        h4_candles: Optional 4H data
    
    Returns:
        List of completed Trade objects
    """
    if params is None:
        params = StrategyParams()
    
    signals = generate_signals(
        candles, symbol, params,
        monthly_candles, weekly_candles, h4_candles
    )
    
    active_signals = [s for s in signals if s.is_active]
    
    trades = []
    open_trade = None
    last_trade_bar = -params.cooldown_bars - 1
    
    for signal in active_signals:
        if signal.bar_index <= last_trade_bar + params.cooldown_bars:
            continue
        
        if signal.entry is None or signal.stop_loss is None or signal.tp1 is None:
            continue
        
        if open_trade is not None:
            continue
        
        entry_bar = signal.bar_index
        entry_price = signal.entry
        sl = signal.stop_loss
        tp1 = signal.tp1
        tp2 = signal.tp2
        tp3 = signal.tp3
        direction = signal.direction
        
        risk = abs(entry_price - sl)
        if risk <= 0:
            continue
        
        trade_closed = False
        tp1_hit = False
        trailing_sl = sl
        
        for exit_bar in range(entry_bar + 1, len(candles)):
            c = candles[exit_bar]
            high = c["high"]
            low = c["low"]
            exit_timestamp = c.get("time") or c.get("timestamp") or c.get("date")
            
            if direction == "bullish":
                if low <= trailing_sl:
                    if tp1_hit:
                        reward = trailing_sl - entry_price
                        rr = reward / risk
                        exit_reason = "TP1+Trail"
                        is_winner = rr >= 0
                    else:
                        reward = trailing_sl - entry_price
                        rr = -1.0
                        exit_reason = "SL"
                        is_winner = False
                    
                    trade_closed = True
                elif tp1 is not None and high >= tp1 and not tp1_hit:
                    tp1_hit = True
                    trailing_sl = entry_price
                    
                    if low <= trailing_sl:
                        reward = 0
                        rr = 0.0
                        exit_reason = "TP1+Trail"
                        is_winner = True
                        trade_closed = True
                
                if not trade_closed and tp1_hit:
                    if tp3 is not None and high >= tp3:
                        reward = tp3 - entry_price
                        rr = reward / risk
                        exit_reason = "TP3"
                        is_winner = True
                        trade_closed = True
                    elif tp2 is not None and high >= tp2:
                        reward = tp2 - entry_price
                        rr = reward / risk
                        exit_reason = "TP2"
                        is_winner = True
                        trade_closed = True
            else:
                if high >= trailing_sl:
                    if tp1_hit:
                        reward = entry_price - trailing_sl
                        rr = reward / risk
                        exit_reason = "TP1+Trail"
                        is_winner = rr >= 0
                    else:
                        reward = entry_price - trailing_sl
                        rr = -1.0
                        exit_reason = "SL"
                        is_winner = False
                    
                    trade_closed = True
                elif tp1 is not None and low <= tp1 and not tp1_hit:
                    tp1_hit = True
                    trailing_sl = entry_price
                    
                    if high >= trailing_sl:
                        reward = 0
                        rr = 0.0
                        exit_reason = "TP1+Trail"
                        is_winner = True
                        trade_closed = True
                
                if not trade_closed and tp1_hit:
                    if tp3 is not None and low <= tp3:
                        reward = entry_price - tp3
                        rr = reward / risk
                        exit_reason = "TP3"
                        is_winner = True
                        trade_closed = True
                    elif tp2 is not None and low <= tp2:
                        reward = entry_price - tp2
                        rr = reward / risk
                        exit_reason = "TP2"
                        is_winner = True
                        trade_closed = True
            
            if trade_closed:
                entry_timestamp = signal.timestamp
                
                trade = Trade(
                    symbol=symbol,
                    direction=direction,
                    entry_date=entry_timestamp,
                    exit_date=exit_timestamp,
                    entry_price=entry_price,
                    exit_price=entry_price + reward if direction == "bullish" else entry_price - reward,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    risk=risk,
                    reward=reward,
                    rr=rr,
                    is_winner=is_winner,
                    exit_reason=exit_reason,
                    confluence_score=signal.confluence_score,
                )
                trades.append(trade)
                last_trade_bar = exit_bar
                break
        
        if not trade_closed:
            pass
    
    return trades


def get_default_params() -> StrategyParams:
    """Get default strategy parameters."""
    return StrategyParams()


def get_aggressive_params() -> StrategyParams:
    """Get aggressive (more trades) strategy parameters."""
    return StrategyParams(
        min_confluence=1,
        min_quality_factors=0,
        require_confirmation_for_active=False,
        require_htf_alignment=False,
        cooldown_bars=0,
    )


def get_conservative_params() -> StrategyParams:
    """Get conservative (higher quality) strategy parameters."""
    return StrategyParams(
        min_confluence=4,
        min_quality_factors=2,
        require_confirmation_for_active=True,
        require_htf_alignment=True,
        cooldown_bars=5,
        atr_tp1_multiplier=0.8,
        atr_tp2_multiplier=1.5,
        atr_tp3_multiplier=2.5,
    )
