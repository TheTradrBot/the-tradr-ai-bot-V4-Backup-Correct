#!/usr/bin/env python3
"""Test strategy improvements"""

from backtest import run_backtest
from strategy import scan_single_asset

print("=" * 80)
print("TESTING STRATEGY IMPROVEMENTS")
print("=" * 80)

# Test 1: Current scan to see improved confluence
print("\n" + "="*80)
print("TEST 1: Current Market Scans (Should see more HTF bias passes)")
print("="*80)

test_assets = ["EUR_USD", "XAU_USD", "BTC_USD", "GBP_USD"]
for asset in test_assets:
    result = scan_single_asset(asset)
    if result:
        print(f"\n{asset}:")
        print(f"  Direction: {result.direction}, Status: {result.status}")
        print(f"  Confluence: {result.confluence_score}/7")
        flags_str = result.summary_reason.split('confluence: ')[1].split('.')[0]
        parts = flags_str.split(', ')
        for part in parts:
            key, val = part.split('=')
            symbol = '✓' if val == 'yes' else '✗'
            print(f"    {symbol} {key}")

# Test 2: Backtest EUR_USD over multiple periods
print("\n" + "="*80)
print("TEST 2: Backtest EUR_USD (Should see trades now!)")
print("="*80)

periods = [
    ("Last 12 months", "12m"),
    ("Last 6 months", "6m"),
    ("2024 Jan-Sep", "2024-01-01:2024-09-30")
]

for label, period in periods:
    result = run_backtest("EUR_USD", period)
    print(f"\n{label}:")
    print(f"  Period: {result['period']}")
    print(f"  Total Trades: {result['total_trades']}")
    if result['total_trades'] > 0:
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Net Return: {result['net_return_pct']:.2f}%")
        
        trades = result.get('trades', [])
        tp_counts = {}
        for t in trades:
            reason = t.get('exit_reason', 'Unknown')
            tp_counts[reason] = tp_counts.get(reason, 0) + 1
        
        print(f"  Exit breakdown: {tp_counts}")

# Test 3: Backtest BTC
print("\n" + "="*80)
print("TEST 3: Backtest BTC_USD")
print("="*80)

result = run_backtest("BTC_USD", "12m")
print(f"Period: {result['period']}")
print(f"Total Trades: {result['total_trades']}")
if result['total_trades'] > 0:
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Net Return: {result['net_return_pct']:.2f}%")

print("\n" + "="*80)
print("✅ TESTING COMPLETE")
print("="*80)
