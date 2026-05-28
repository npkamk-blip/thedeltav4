"""
THE DELTA v2 — phase1_5_seed_discovery.py
==========================================
Runs AFTER phase 1 collection finishes.
Runs BEFORE phase 1B (1-min PM collection).

What this does:
1. Loops every trading day 2025-2026
2. Finds every stock that seeded (2x+), super (6x+), mega (11x+)
3. For each seed finds same-day similar controls
4. Outputs:
   - seed_registry.json      {date: [tickers that seeded]}
   - control_registry.json   {date: {seed_ticker: [control_tickers]}}
   - pm_1min_candidates.json [all tickers needing 1-min PM bars]
   - seed_stats.json         summary stats

No new API calls needed — uses cached daily bars only.
"""

import os, json, gc, logging
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT      = Path(os.environ.get("DATA_DIR", "/app/data"))
TICKER_RAW_DIR = DATA_ROOT / "raw" / "tickers"
OUTPUT_DIR     = DATA_ROOT / "raw"
LOG_DIR        = DATA_ROOT / "logs"

START_DATE = date(2025, 1, 1)
END_DATE   = date(2026, 5, 23)

# Seed thresholds
SEED_MULT  = 2.00   # 100% gain (2x prev_close)
SUPER_MULT = 3.50   # 250% gain (3.5x prev_close)

# Universe filters — same as collector
MIN_PRICE        = 0.10
MAX_PRICE        = 10.00   # slightly wider to catch pre-seed prices
MIN_DOLLAR_VOL   = 10_000  # looser than main scanner to catch quiet seeds
MIN_LABEL_VOLUME = 25_000

# Control selection
SEED_CONTROLS        = 5
SUPER_CONTROLS       = 4
FLOAT_RANGE_FACTOR   = 3.0   # control float within 3x of seed float
PRICE_RANGE_FACTOR   = 3.0   # control price within 3x of seed price

ET = ZoneInfo("America/New_York")

KNOWN_HOLIDAYS = {
    date(2025,1,1), date(2025,1,9), date(2025,1,20), date(2025,2,17),
    date(2025,4,18), date(2025,5,26), date(2025,6,19), date(2025,7,4),
    date(2025,9,1), date(2025,11,27), date(2025,12,25),
    date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,4,3),
    date(2026,5,25),
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "seed_discovery.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("seed_discovery")


# ─────────────────────────────────────────────
# TRADING CALENDAR
# ─────────────────────────────────────────────
def get_trading_days(start: date, end: date) -> list[date]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5 and cur not in KNOWN_HOLIDAYS:
            days.append(cur)
        cur += timedelta(days=1)
    return days

def get_next_trading_day(d: date) -> date:
    next_d = d + timedelta(days=1)
    while next_d.weekday() >= 5 or next_d in KNOWN_HOLIDAYS:
        next_d += timedelta(days=1)
    return next_d


# ─────────────────────────────────────────────
# LOAD TICKER CACHE — lightweight, daily only
# ─────────────────────────────────────────────
def load_daily_bars(ticker: str) -> pd.DataFrame | None:
    """Load only daily bars from cache — fast and memory efficient."""
    cache_path = TICKER_RAW_DIR / f"{ticker}.parquet"
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path, columns=["daily", "float_data", "fetch_ok"])
        row = df.iloc[0].to_dict()
        del df

        daily_raw = row.get("daily")
        if daily_raw is None:
            return None

        if isinstance(daily_raw, pd.DataFrame):
            daily_df = daily_raw
        elif hasattr(daily_raw, "tolist"):
            daily_df = pd.DataFrame(daily_raw.tolist())
        else:
            daily_df = pd.DataFrame(list(daily_raw)) if daily_raw else pd.DataFrame()

        if daily_df.empty:
            return None

        if "date" not in daily_df.columns:
            if "t" in daily_df.columns:
                daily_df["date"] = pd.to_datetime(
                    daily_df["t"], unit="ms", utc=True
                ).dt.tz_convert(ET).dt.date
            else:
                return None

        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

        # Rename columns if needed
        col_map = {"o":"open","h":"high","l":"low","c":"close","v":"volume"}
        daily_df = daily_df.rename(columns={k:v for k,v in col_map.items() if k in daily_df.columns})

        # Get float data
        float_data = row.get("float_data", {})
        if isinstance(float_data, str):
            float_data = json.loads(float_data)
        elif hasattr(float_data, "tolist"):
            float_data = float_data.tolist()

        shares = float_data.get("shares_outstanding", 0) if isinstance(float_data, dict) else 0
        float_M = float(shares) / 1_000_000 if shares and float(shares) > 0 else -1

        daily_df["float_M"] = float_M
        return daily_df

    except Exception as e:
        log.warning(f"WARN load_daily | {ticker} | {e}")
        return None


# ─────────────────────────────────────────────
# LOAD ALL TICKERS
# ─────────────────────────────────────────────
def get_cached_tickers() -> list[str]:
    """Get all tickers that have been collected."""
    tickers = [f.stem for f in TICKER_RAW_DIR.glob("*.parquet")]
    log.info(f"Found {len(tickers)} cached tickers")
    return sorted(tickers)


# ─────────────────────────────────────────────
# BUILD DAILY UNIVERSE — CHUNKED FOR LOW MEMORY
# ─────────────────────────────────────────────
CHUNK_SIZE = 300  # process 300 tickers at a time to stay under 512MB

def process_ticker_chunk(
    tickers: list[str],
    trading_days: list[date],
    chunk_idx: int,
) -> dict:
    """
    Process one chunk of tickers.
    Returns partial daily_universe for this chunk only.
    Frees memory immediately after.
    """
    partial: dict[date, list[dict]] = {d: [] for d in trading_days}

    for i, ticker in enumerate(tickers):
        daily_df = load_daily_bars(ticker)
        if daily_df is None or daily_df.empty:
            continue

        daily_df = daily_df.sort_values("date").reset_index(drop=True)
        float_M  = float(daily_df["float_M"].iloc[0]) if "float_M" in daily_df.columns else -1

        dates_in_df = set(daily_df["date"].tolist())

        for trade_date in trading_days:
            if trade_date not in dates_in_df:
                continue

            today_rows = daily_df[daily_df["date"] == trade_date]
            if today_rows.empty:
                continue

            past_rows = daily_df[daily_df["date"] < trade_date]
            if past_rows.empty:
                continue

            today_row  = today_rows.iloc[-1]
            prev_row   = past_rows.iloc[-1]

            prev_close = float(prev_row.get("close", 0) or 0)
            prev_vol   = float(prev_row.get("volume", 0) or 0)
            day_high   = float(today_row.get("high", 0) or 0)
            day_volume = float(today_row.get("volume", 0) or 0)
            dollar_vol = prev_close * prev_vol

            if prev_close < MIN_PRICE or prev_close > MAX_PRICE:
                continue
            if dollar_vol < MIN_DOLLAR_VOL:
                continue
            if day_volume < MIN_LABEL_VOLUME:
                continue

            partial[trade_date].append({
                "ticker":     ticker,
                "prev_close": prev_close,
                "prev_vol":   prev_vol,
                "day_high":   day_high,
                "day_volume": day_volume,
                "float_M":    float_M,
                "dollar_vol": dollar_vol,
            })

    return partial


def build_daily_snapshots_chunked(
    tickers: list[str],
    trading_days: list[date],
) -> tuple[dict, dict, dict]:
    """
    Process tickers in chunks of CHUNK_SIZE to stay under 512MB RAM.
    Finds seeds per chunk, accumulates results, frees memory between chunks.
    
    Returns seed_registry, seed_details, nonseed_by_date directly
    instead of building the full universe in memory first.
    """
    log.info(f"Building snapshots in chunks of {CHUNK_SIZE} (memory efficient)")
    log.info(f"Total: {len(tickers)} tickers × {len(trading_days)} days")

    # Accumulators
    seed_registry:   dict[str, list[str]]       = {}
    seed_details:    dict[str, dict]             = {}
    nonseed_by_date: dict[str, list[dict]]       = {}

    total_seeds  = 0
    total_supers = 0
    n_chunks     = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end   = min(chunk_start + CHUNK_SIZE, len(tickers))
        chunk       = tickers[chunk_start:chunk_end]

        log.info(f"Chunk {chunk_idx+1}/{n_chunks} | tickers {chunk_start}-{chunk_end}")

        # Process this chunk
        partial = process_ticker_chunk(chunk, trading_days, chunk_idx)

        # Find seeds in this chunk's data
        for trade_date, stocks in partial.items():
            if not stocks:
                continue

            date_str = trade_date.isoformat()

            for stock in stocks:
                ticker     = stock["ticker"]
                prev_close = stock["prev_close"]
                day_high   = stock["day_high"]

                if prev_close <= 0:
                    continue

                ratio = day_high / prev_close

                if ratio >= SEED_MULT:
                    seed_type = "super" if ratio >= SUPER_MULT else "seed"
                    if seed_type == "super":
                        total_supers += 1
                    else:
                        total_seeds += 1

                    if date_str not in seed_registry:
                        seed_registry[date_str]   = []
                        seed_details[date_str]    = {}

                    seed_registry[date_str].append(ticker)
                    seed_details[date_str][ticker] = {
                        "ticker":     ticker,
                        "type":       seed_type,
                        "prev_close": prev_close,
                        "day_high":   day_high,
                        "pct":        round((ratio - 1) * 100, 1),
                        "float_M":    stock["float_M"],
                        "dollar_vol": stock["dollar_vol"],
                    }
                    log.debug(f"SEED {date_str}: {ticker} {seed_type} +{(ratio-1)*100:.0f}%")
                else:
                    # Non-seed — keep for control selection
                    if date_str not in nonseed_by_date:
                        nonseed_by_date[date_str] = []
                    nonseed_by_date[date_str].append(stock)

        # Free chunk memory immediately
        del partial
        gc.collect()

    # Log seed days found
    for date_str, tickers_list in sorted(seed_registry.items()):
        log.info(f"{date_str}: {len(tickers_list)} seeds — {tickers_list}")

    log.info("=" * 50)
    log.info("SEED DISCOVERY COMPLETE")
    log.info(f"Total seed days:  {len(seed_registry)}")
    log.info(f"Total seeds:      {total_seeds}")
    log.info(f"Total supers:     {total_supers}")
    log.info(f"Total events:     {total_seeds + total_supers}")
    log.info(f"Avg seeds/day:    {(total_seeds+total_supers)/max(len(seed_registry),1):.1f}")
    log.info("=" * 50)

    return seed_registry, seed_details, nonseed_by_date


# ─────────────────────────────────────────────
# SEED DISCOVERY
# ─────────────────────────────────────────────
def discover_seeds(daily_universe: dict) -> tuple[dict, dict, dict]:
    """
    Find all seeds, supers, megas in the daily universe.
    
    Returns:
        seed_registry:   {date_str: [seed_tickers]}
        seed_details:    {date_str: {ticker: {type, prev_close, day_high, pct, float_M}}}
        nonseed_by_date: {date_str: [non_seed_ticker_dicts]} for control selection
    """
    seed_registry   = {}
    seed_details    = {}
    nonseed_by_date = {}

    total_seeds  = 0
    total_supers = 0

    for trade_date, stocks in daily_universe.items():
        if not stocks:
            continue

        date_str = trade_date.isoformat()
        seeds    = []
        nonseeds = []

        for stock in stocks:
            ticker     = stock["ticker"]
            prev_close = stock["prev_close"]
            day_high   = stock["day_high"]

            if prev_close <= 0:
                continue

            ratio = day_high / prev_close

            if ratio >= SEED_MULT:
                seed_type = "seed"
                if ratio >= SUPER_MULT:
                    seed_type = "super"
                    total_supers += 1
                else:
                    total_seeds += 1

                seeds.append({
                    "ticker":     ticker,
                    "type":       seed_type,
                    "prev_close": prev_close,
                    "day_high":   day_high,
                    "pct":        round((ratio - 1) * 100, 1),
                    "float_M":    stock["float_M"],
                    "dollar_vol": stock["dollar_vol"],
                })
                log.debug(f"SEED {date_str}: {ticker} {seed_type} +{(ratio-1)*100:.0f}% float={stock['float_M']:.1f}M")
            else:
                nonseeds.append(stock)

        if seeds:
            seed_registry[date_str]   = [s["ticker"] for s in seeds]
            seed_details[date_str]    = {s["ticker"]: s for s in seeds}
            nonseed_by_date[date_str] = nonseeds
            log.info(f"{date_str}: {len(seeds)} seeds — {[s['ticker'] for s in seeds]}")

    log.info(f"=" * 50)
    log.info(f"SEED DISCOVERY COMPLETE")
    log.info(f"Total seed days:  {len(seed_registry)}")
    log.info(f"Total seeds:      {total_seeds}")
    log.info(f"Total supers:     {total_supers}")
    log.info(f"Total events:     {total_seeds + total_supers + total_megas}")
    log.info(f"Avg seeds/day:    {(total_seeds+total_supers+total_megas)/max(len(seed_registry),1):.1f}")
    log.info(f"=" * 50)

    return seed_registry, seed_details, nonseed_by_date


# ─────────────────────────────────────────────
# CONTROL SELECTION
# ─────────────────────────────────────────────
def select_controls(
    seed_details: dict,
    nonseed_by_date: dict,
) -> dict:
    """
    For each seed on each day, find same-day control tickers
    with similar float and price profile that did NOT seed.
    
    Returns:
        control_registry: {date_str: {seed_ticker: [control_tickers]}}
    """
    log.info("Selecting same-day controls for each seed...")
    control_registry = {}
    
    total_controls_found = 0
    total_controls_needed = 0

    for date_str, seed_tickers in seed_details.items():
        nonseeds = nonseed_by_date.get(date_str, [])
        if not nonseeds:
            log.warning(f"WARN controls | {date_str} | no non-seed stocks available")
            continue

        control_registry[date_str] = {}

        for seed_ticker, seed_info in seed_tickers.items():
            seed_float = seed_info["float_M"]
            seed_price = seed_info["prev_close"]
            n_ctrl = SUPER_CONTROLS if seed_info.get("type") == "super" else SEED_CONTROLS
            total_controls_needed += n_ctrl

            # Find similar non-seeds
            candidates = []
            for ns in nonseeds:
                if ns["ticker"] == seed_ticker:
                    continue

                ns_float = ns["float_M"]
                ns_price = ns["prev_close"]

                # Float similarity check
                if seed_float > 0 and ns_float > 0:
                    float_ratio = max(seed_float, ns_float) / min(seed_float, ns_float)
                    if float_ratio > FLOAT_RANGE_FACTOR:
                        continue
                elif seed_float > 0 and ns_float <= 0:
                    # Seed has float but control doesn't — skip
                    continue

                # Price similarity check
                if seed_price > 0 and ns_price > 0:
                    price_ratio = max(seed_price, ns_price) / min(seed_price, ns_price)
                    if price_ratio > PRICE_RANGE_FACTOR:
                        continue

                # Calculate similarity score
                float_score = 0
                price_score = 0

                if seed_float > 0 and ns_float > 0:
                    float_score = 1 - abs(seed_float - ns_float) / max(seed_float, ns_float)
                if seed_price > 0 and ns_price > 0:
                    price_score = 1 - abs(seed_price - ns_price) / max(seed_price, ns_price)

                similarity = (float_score + price_score) / 2
                candidates.append((ns["ticker"], similarity))

            # Sort by similarity, take top N
            n_controls = SUPER_CONTROLS if seed_info["type"] == "super" else SEED_CONTROLS
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = [t for t, _ in candidates[:n_controls]]

            if not selected:
                log.warning(
                    f"WARN controls | {date_str} | {seed_ticker} | "
                    f"no similar controls found (float={seed_float:.1f}M price=${seed_price:.2f})"
                )
            else:
                log.debug(
                    f"Controls | {date_str} | {seed_ticker} → {selected}"
                )

            control_registry[date_str][seed_ticker] = selected
            total_controls_found += len(selected)

    log.info(f"Controls: {total_controls_found}/{total_controls_needed} found")
    return control_registry


# ─────────────────────────────────────────────
# BUILD 1-MIN CANDIDATES LIST
# ─────────────────────────────────────────────
def build_pm_candidates(
    seed_registry: dict,
    control_registry: dict,
) -> list[str]:
    """
    Build deduplicated list of all tickers that need 1-min PM bars.
    = all seeds + all controls
    """
    candidates = set()

    for date_str, seed_tickers in seed_registry.items():
        for ticker in seed_tickers:
            candidates.add(ticker)

    for date_str, seed_controls in control_registry.items():
        for seed_ticker, controls in seed_controls.items():
            for ticker in controls:
                candidates.add(ticker)

    candidates_list = sorted(candidates)
    log.info(f"PM 1-min candidates: {len(candidates_list)} tickers")
    return candidates_list


# ─────────────────────────────────────────────
# STATS SUMMARY
# ─────────────────────────────────────────────
def build_stats(
    seed_registry: dict,
    seed_details: dict,
    control_registry: dict,
    pm_candidates: list[str],
) -> dict:
    """Build summary statistics for validation."""

    # Seed type breakdown
    type_counts = {"seed": 0, "super": 0}
    float_buckets = {"nano(<5M)": 0, "micro(5-15M)": 0, "small(15-50M)": 0, "mid(50M+)": 0, "unknown": 0}
    
    all_seeds = []
    for date_str, seeds in seed_details.items():
        for ticker, info in seeds.items():
            type_counts[info["type"]] += 1
            all_seeds.append(info)
            
            fm = info["float_M"]
            if fm <= 0:       float_buckets["unknown"] += 1
            elif fm < 5:      float_buckets["nano(<5M)"] += 1
            elif fm < 15:     float_buckets["micro(5-15M)"] += 1
            elif fm < 50:     float_buckets["small(15-50M)"] += 1
            else:             float_buckets["mid(50M+)"] += 1

    # Top seeds by pct gain
    top_seeds = sorted(all_seeds, key=lambda x: x["pct"], reverse=True)[:20]

    # Days with most seeds
    busiest_days = sorted(
        [(d, len(t)) for d, t in seed_registry.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Control coverage
    total_seeds_needing_controls = sum(len(v) for v in seed_registry.values())
    total_controls_assigned = sum(
        len(controls)
        for date_controls in control_registry.values()
        for controls in date_controls.values()
    )

    stats = {
        "total_seed_days":     len(seed_registry),
        "total_events":        sum(type_counts.values()),
        "type_breakdown":      type_counts,
        "float_breakdown":     float_buckets,
        "avg_seeds_per_day":   round(sum(type_counts.values()) / max(len(seed_registry), 1), 2),
        "busiest_days":        busiest_days,
        "top_20_seeds":        [
            f"{s['ticker']} +{s['pct']}% ({s['type']}) float={s['float_M']:.1f}M {s.get('date','')}"
            for s in top_seeds
        ],
        "controls_needed":     total_controls_needed,
        "controls_assigned":   total_controls_assigned,
        "pm_1min_candidates":  len(pm_candidates),
        "date_range":          f"{START_DATE} → {END_DATE}",
    }
    return stats


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("THE DELTA v2 — Phase 1.5 Seed Discovery")
    log.info(f"Window: {START_DATE} → {END_DATE}")
    log.info("=" * 60)

    # Check output already exists
    seed_reg_path = OUTPUT_DIR / "seed_registry.json"
    if seed_reg_path.exists():
        log.info("Seed registry already exists — loading cached results")
        with open(seed_reg_path) as f:
            seed_registry = json.load(f)
        with open(OUTPUT_DIR / "seed_details.json") as f:
            seed_details = json.load(f)
        with open(OUTPUT_DIR / "control_registry.json") as f:
            control_registry = json.load(f)
        with open(OUTPUT_DIR / "pm_1min_candidates.json") as f:
            pm_candidates = json.load(f)
        log.info(f"Loaded: {len(seed_registry)} seed days, {len(pm_candidates)} PM candidates")
        return

    # Step 1 — Get all cached tickers
    tickers = get_cached_tickers()
    if not tickers:
        log.error("FAIL | no cached tickers found — run phase 1 first")
        return

    # Step 2 — Get trading calendar
    trading_days = get_trading_days(START_DATE, END_DATE)
    log.info(f"Trading days: {len(trading_days)}")

    # Step 3 — Build snapshots + discover seeds in chunks (memory efficient)
    seed_registry, seed_details, nonseed_by_date = build_daily_snapshots_chunked(
        tickers, trading_days
    )

    if not seed_registry:
        log.error("FAIL | no seeds found — check data collection")
        return

    # Step 5 — Select same-day controls
    control_registry = select_controls(seed_details, nonseed_by_date)

    # Free memory
    del nonseed_by_date
    gc.collect()

    # Step 6 — Build PM 1-min candidates list
    pm_candidates = build_pm_candidates(seed_registry, control_registry)

    # Step 7 — Build stats
    stats = build_stats(seed_registry, seed_details, control_registry, pm_candidates)

    # Step 8 — Save everything
    log.info("Saving outputs...")

    with open(OUTPUT_DIR / "seed_registry.json", "w") as f:
        json.dump(seed_registry, f, indent=2)
    log.info(f"Saved seed_registry.json ({len(seed_registry)} days)")

    with open(OUTPUT_DIR / "seed_details.json", "w") as f:
        json.dump(seed_details, f, indent=2)
    log.info(f"Saved seed_details.json")

    with open(OUTPUT_DIR / "control_registry.json", "w") as f:
        json.dump(control_registry, f, indent=2)
    log.info(f"Saved control_registry.json")

    with open(OUTPUT_DIR / "pm_1min_candidates.json", "w") as f:
        json.dump(pm_candidates, f, indent=2)
    log.info(f"Saved pm_1min_candidates.json ({len(pm_candidates)} tickers)")

    with open(OUTPUT_DIR / "seed_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Step 9 — Print summary
    log.info("=" * 60)
    log.info("PHASE 1.5 COMPLETE")
    log.info(f"Seed days:        {stats['total_seed_days']}")
    log.info(f"Total events:     {stats['total_events']}")
    log.info(f"  Seeds:          {stats['type_breakdown']['seed']}")
    log.info(f"  Supers:         {stats['type_breakdown']['super']}")
    log.info(f"  Megas:          {stats['type_breakdown']['mega']}")
    log.info(f"Avg per day:      {stats['avg_seeds_per_day']}")
    log.info(f"Float breakdown:  {stats['float_breakdown']}")
    log.info(f"PM candidates:    {stats['pm_1min_candidates']} tickers need 1-min bars")
    log.info(f"Controls:         {stats['controls_assigned']}/{stats['controls_needed']}")
    log.info("")
    log.info("Top 10 busiest seed days:")
    for d, count in stats["busiest_days"][:10]:
        log.info(f"  {d}: {count} seeds")
    log.info("")
    log.info("Top 10 biggest moves:")
    for s in stats["top_20_seeds"][:10]:
        log.info(f"  {s}")
    log.info("=" * 60)
    log.info("Next step: run collect_pm_1min.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
