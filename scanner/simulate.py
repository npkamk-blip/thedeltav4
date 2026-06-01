"""
THE DELTA v2 — Simulation
Tests TGHL and HKIT as if it were Sunday 11PM and Monday morning
"""
import sys, json, requests
sys.path.insert(0, '/opt/render/project/src')
from scanner.main import *

load_models()
load_si_lookup()

tickers = ['TGHL', 'HKIT']

print("=" * 60)
print("SIMULATION — SUNDAY 11PM MIDNIGHT SCAN")
print("=" * 60)

for ticker in tickers:
    pc, pv = get_prev_close(ticker)
    if pc <= 0:
        print(f"{ticker}: no data")
        continue

    # Get AH bars (Friday after hours)
    ah = get_ah_bars(ticker)
    ah_move = 0
    ah_vol  = 0
    ah_sus  = 0
    if ah:
        prices = [float(b.get("c",0) or 0) for b in ah if b.get("c",0)]
        vols   = [float(b.get("v",0) or 0) for b in ah]
        if prices and pc > 0:
            ah_move = (prices[-1] - pc) / pc
        ah_vol = sum(vols)
        if len(prices) >= 2:
            ah_sus = 1 if prices[-1] >= prices[0] * 0.90 else 0

    # Get EDGAR features
    edgar = get_edgar_features(ticker)
    has_8k      = edgar.get("has_8k", 0)
    has_merger  = edgar.get("has_merger", 0)
    has_rsplit  = edgar.get("has_reverse_split", 0)
    has_dilute  = edgar.get("has_dilution", 0)

    # Get news
    news      = get_polygon_news(ticker, hours_back=72)
    news_info = analyze_news(news)
    has_pr    = news_info.get("has_bullish_pr", 0)
    has_bad   = news_info.get("has_bearish_pr", 0)
    news_title = news_info.get("news_title","")

    # Get SI
    si_pct, si_tier = get_si(ticker)

    # Get float
    float_M, _, _ = get_float(ticker)

    # Score midnight
    scores = score_midnight(ticker, pc, pv)
    mid_seed  = scores.get("midnight_seed", 0)
    mid_super = scores.get("midnight_super", 0)

    # Catalyst check
    ah_vol_ratio = ah_vol / pv if pv > 0 else 0
    has_catalyst = (
        has_8k == 1 or
        has_merger == 1 or
        has_pr == 1 or
        abs(ah_move) > 0.05 or
        ah_vol_ratio > 2.0 or
        ah_sus == 1
    )

    would_alert = mid_seed >= 0.55 and has_catalyst and not has_bad

    print(f"\n{'🌱 MIDNIGHT ALERT' if would_alert else '📋 WATCHLIST ONLY'} — {ticker}")
    print(f"  Prev close:      ${pc:.3f}")
    print(f"  Float:           {float_M:.1f}M shares")
    print(f"  SI:              {si_pct:.1f}%")
    print(f"  midnight_seed:   {mid_seed:.3f} {'✅' if mid_seed>=0.55 else '❌'}")
    print(f"  midnight_super:  {mid_super:.3f}")
    print(f"  AH move:         {ah_move*100:+.1f}%")
    print(f"  AH vol ratio:    {ah_vol_ratio:.1f}x")
    print(f"  AH sustained:    {ah_sus}")
    print(f"  has_8k:          {has_8k}")
    print(f"  has_merger:      {has_merger}")
    print(f"  has_rev_split:   {has_rsplit}")
    print(f"  has_dilution:    {has_dilute}")
    print(f"  has_bullish_pr:  {has_pr}")
    print(f"  has_bad_news:    {has_bad}")
    if news_title:
        print(f"  News:            {news_title[:60]}")
    print(f"  Has catalyst:    {'YES ✅' if has_catalyst else 'NO ❌'}")

print()
print("=" * 60)
print("SIMULATION — MONDAY MORNING RESCANS")
print("=" * 60)

# Simulate morning scans at key times using actual bars
scan_times = [
    (5, 30,  "5:30AM — initial scan"),
    (6, 0,   "6:00AM — rescan"),
    (7, 0,   "7:00AM — rescan"),
    (8, 0,   "8:00AM — rescan (TGHL building)"),
    (8, 30,  "8:30AM — rescan (TGHL explosion)"),
    (9, 15,  "9:15AM — final premarket scan"),
]

for ticker in tickers:
    print(f"\n{'='*55}")
    print(f"  {ticker} — Morning Timeline")
    print(f"{'='*55}")

    pc, pv = get_prev_close(ticker)
    pm_all = get_pm_bars(ticker)

    if not pm_all:
        print(f"  No PM bars available")
        continue

    for scan_hour, scan_min, label in scan_times:
        # Get bars up to scan time
        window_end   = scan_hour * 60 + scan_min
        window_start = window_end - 90

        window_bars = [b for b in pm_all
                       if (b.get("hour",0)*60 + b.get("minute",0)) <= window_end
                       and (b.get("hour",0)*60 + b.get("minute",0)) >= window_start]

        all_bars_to_now = [b for b in pm_all
                           if (b.get("hour",0)*60 + b.get("minute",0)) <= window_end]

        if not window_bars and not all_bars_to_now:
            print(f"  {label}: no bars yet")
            continue

        score_bars = window_bars if window_bars else all_bars_to_now

        # Current price at scan time
        price_now = float(score_bars[-1].get("c", pc) or pc)
        pm_high   = max(float(b.get("h",0) or 0) for b in score_bars)
        gap       = (price_now - pc) / pc if pc > 0 else 0
        fade      = (pm_high - price_now) / pm_high if pm_high > 0 else 0
        active    = sum(1 for b in score_bars if float(b.get("v",0) or 0) > 100)

        # Filters
        gap_ok  = 0.02 <= gap <= 0.25
        fade_ok = fade <= 0.25

        # Score
        wl = {"prev_close": pc, "prev_vol": pv}
        scores = score_morning(ticker, pc, pv, wl)
        ms = scores.get("morning_seed", 0)
        score_ok = ms >= 0.50

        would_alert = gap_ok and fade_ok and score_ok
        status = "🌱 ALERT" if would_alert else "⛔ skip"

        print(f"  {label}: {status}")
        print(f"    Price=${price_now:.3f} Gap={gap*100:+.1f}% {'✅' if gap_ok else '❌'} "
              f"Fade={fade*100:.1f}% {'✅' if fade_ok else '❌'} "
              f"Active={active} Seed={ms:.3f} {'✅' if score_ok else '❌'}")

print("\n" + "=" * 60)
print("SIMULATION COMPLETE")
print("=" * 60)
