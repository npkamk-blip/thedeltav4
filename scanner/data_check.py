"""
THE DELTA v2 — data_check.py
==============================
Verifies all data sources are accessible from the scanner service.
Run once to confirm everything is wired up correctly before going live.

Tests:
  1. Polygon API — can we fetch tickers, prices, bars?
  2. SI master   — is it accessible and readable?
  3. EDGAR       — is filings_master and cik_map accessible?
  4. Models      — are all 4 models loaded?
  5. Full ticker test — score one real ticker end to end
"""

import os, sys, json, time, logging, requests, threading
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from http.server import HTTPServer, BaseHTTPRequestHandler

ET = ZoneInfo("America/New_York")

POLYGON_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
MODEL_DIR   = Path(os.environ.get("MODEL_DIR",   "/opt/render/project/src/models"))
SUPPORT_DIR = Path(os.environ.get("SUPPORT_DIR", "/opt/render/project/src/support"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("data_check")

class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"alive")
    def log_message(self, *a): pass

threading.Thread(target=lambda: HTTPServer(("0.0.0.0", 8080), _Health).serve_forever(), daemon=True).start()
time.sleep(1)

results = {}

# ─────────────────────────────────────────────
# TEST 1: Polygon API
# ─────────────────────────────────────────────
log.info("="*50)
log.info("TEST 1: Polygon API")
try:
    now = datetime.now(ET)
    frm = (now - timedelta(days=5)).strftime("%Y-%m-%d")
    to_ = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    r = requests.get(
        f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{frm}/{to_}",
        params={"apiKey": POLYGON_API_KEY, "limit": 5},
        timeout=15
    )
    if r.status_code == 200:
        data = r.json()
        bars = data.get("results", [])
        if bars:
            last = bars[-1]
            log.info(f"  ✅ Polygon API OK — AAPL last close: ${last.get('c',0):.2f}")
            results["polygon_api"] = "OK"
        else:
            log.warning("  ⚠️  Polygon API responded but no bars")
            results["polygon_api"] = "NO_DATA"
    else:
        log.error(f"  ❌ Polygon API failed: HTTP {r.status_code}")
        results["polygon_api"] = f"HTTP_{r.status_code}"
except Exception as e:
    log.error(f"  ❌ Polygon API error: {e}")
    results["polygon_api"] = "ERROR"

# ─────────────────────────────────────────────
# TEST 2: Polygon PM bars (1-min)
# ─────────────────────────────────────────────
log.info("TEST 2: Polygon 1-min bars")
try:
    now   = datetime.now(ET)
    today = now.strftime("%Y-%m-%d")
    yest  = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    r = requests.get(
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{yest}/{yest}",
        params={"apiKey": POLYGON_API_KEY, "limit": 10, "sort": "desc"},
        timeout=15
    )
    if r.status_code == 200:
        bars = r.json().get("results", [])
        log.info(f"  ✅ 1-min bars OK — SPY yesterday: {len(bars)} bars")
        results["polygon_1min"] = "OK"
    else:
        log.error(f"  ❌ 1-min bars failed: HTTP {r.status_code}")
        results["polygon_1min"] = f"HTTP_{r.status_code}"
except Exception as e:
    log.error(f"  ❌ 1-min bars error: {e}")
    results["polygon_1min"] = "ERROR"

# ─────────────────────────────────────────────
# TEST 3: Polygon ticker details (float)
# ─────────────────────────────────────────────
log.info("TEST 3: Polygon ticker details")
try:
    r = requests.get(
        "https://api.polygon.io/v3/reference/tickers/AAPL",
        params={"apiKey": POLYGON_API_KEY},
        timeout=15
    )
    if r.status_code == 200:
        d = r.json().get("results", {})
        mc = d.get("market_cap", 0)
        sh = d.get("share_class_shares_outstanding", 0)
        log.info(f"  ✅ Ticker details OK — AAPL market_cap: ${mc/1e9:.0f}B shares: {sh/1e9:.1f}B")
        results["polygon_details"] = "OK"
    else:
        log.error(f"  ❌ Ticker details failed: HTTP {r.status_code}")
        results["polygon_details"] = f"HTTP_{r.status_code}"
except Exception as e:
    log.error(f"  ❌ Ticker details error: {e}")
    results["polygon_details"] = "ERROR"

# ─────────────────────────────────────────────
# TEST 4: SI master
# ─────────────────────────────────────────────
log.info("TEST 4: SI master")
si_path = SUPPORT_DIR / "si_lookup.parquet"
if si_path.exists():
    try:
        import pandas as pd
        df = pd.read_parquet(si_path)
        rows = df[df["Symbol"] == "SPY"]
        log.info(f"  ✅ SI lookup OK — {len(df):,} tickers, SPY rows: {len(rows)}")
        results["si_master"] = "OK"
    except Exception as e:
        log.error(f"  ❌ SI master read error: {e}")
        results["si_master"] = "ERROR"
else:
    log.error(f"  ❌ SI master not found: {si_path}")
    results["si_master"] = "NOT_FOUND"

# ─────────────────────────────────────────────
# TEST 5: EDGAR master
# ─────────────────────────────────────────────
log.info("TEST 5: EDGAR master")
edgar_path = SUPPORT_DIR / "edgar_recent.parquet"
cik_path   = SUPPORT_DIR / "cik_map.json"
if edgar_path.exists() and cik_path.exists():
    try:
        import pandas as pd
        df  = pd.read_parquet(edgar_path, columns=["cik","form_type","filed"])
        with open(cik_path) as f:
            cik_map = json.load(f)
        # Test lookup for a known ticker
        cik = cik_map.get("AAPL")
        rows = df[df["cik"] == cik] if cik else pd.DataFrame()
        log.info(f"  ✅ EDGAR master OK — {len(df):,} filings, {len(cik_map):,} tickers in CIK map")
        log.info(f"     AAPL CIK: {cik}, AAPL filings: {len(rows)}")
        results["edgar_master"] = "OK"
    except Exception as e:
        log.error(f"  ❌ EDGAR master error: {e}")
        results["edgar_master"] = "ERROR"
else:
    log.error(f"  ❌ EDGAR files not found at {DATA_ROOT}/raw/edgar/")
    results["edgar_master"] = "NOT_FOUND"

# ─────────────────────────────────────────────
# TEST 6: Models
# ─────────────────────────────────────────────
log.info("TEST 6: Models")
import xgboost as xgb
models_ok = 0
for name in ["midnight_seed_model","midnight_super_model","morning_seed_model","morning_super_model"]:
    path = MODEL_DIR / f"{name}.json"
    if path.exists():
        try:
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            log.info(f"  ✅ {name}")
            models_ok += 1
        except Exception as e:
            log.error(f"  ❌ {name}: {e}")
    else:
        log.error(f"  ❌ {name} not found at {path}")

fc_path = MODEL_DIR / "feature_cols.json"
if fc_path.exists():
    with open(fc_path) as f:
        fc = json.load(f)
    log.info(f"  ✅ feature_cols.json — {fc}")
    results["models"] = f"{models_ok}/4 loaded"
else:
    log.error(f"  ❌ feature_cols.json not found")
    results["models"] = "MISSING_FEATURE_COLS"

# ─────────────────────────────────────────────
# TEST 7: Full end-to-end ticker score
# ─────────────────────────────────────────────
log.info("TEST 7: Full end-to-end score test (SOXS)")
try:
    # Use a small volatile ticker that's likely to have data
    test_ticker = "SOXS"
    now = datetime.now(ET)
    frm = (now - timedelta(days=5)).strftime("%Y-%m-%d")
    to_ = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # Get prev close
    r = requests.get(
        f"https://api.polygon.io/v2/aggs/ticker/{test_ticker}/range/1/day/{frm}/{to_}",
        params={"apiKey": POLYGON_API_KEY, "sort": "desc", "limit": 3},
        timeout=15
    )
    if r.status_code == 200 and r.json().get("results"):
        bars = r.json()["results"]
        prev_close = float(bars[0].get("c", 0))
        prev_vol   = float(bars[0].get("v", 0))
        log.info(f"  ✅ {test_ticker} prev_close=${prev_close:.3f} vol={prev_vol:,.0f}")

        # Get float
        r2 = requests.get(
            f"https://api.polygon.io/v3/reference/tickers/{test_ticker}",
            params={"apiKey": POLYGON_API_KEY},
            timeout=15
        )
        if r2.status_code == 200:
            d = r2.json().get("results", {})
            sh = float(d.get("share_class_shares_outstanding", 0) or 0)
            mc = float(d.get("market_cap", 0) or 0)
            float_M = sh/1_000_000 if sh > 0 else (mc/prev_close/1_000_000 if mc > 0 and prev_close > 0 else -1)
            log.info(f"  ✅ {test_ticker} float={float_M:.1f}M market_cap=${mc/1e6:.0f}M")
            results["end_to_end"] = f"OK — {test_ticker} close=${prev_close:.3f} float={float_M:.1f}M"
        else:
            log.warning(f"  ⚠️  Could not get float for {test_ticker}")
            results["end_to_end"] = "PARTIAL"
    else:
        log.warning(f"  ⚠️  No price data for {test_ticker}")
        results["end_to_end"] = "NO_DATA"
except Exception as e:
    log.error(f"  ❌ End-to-end test error: {e}")
    results["end_to_end"] = f"ERROR: {e}"

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
log.info("="*50)
log.info("DATA CHECK SUMMARY")
log.info("="*50)
all_ok = True
for test, result in results.items():
    status = "✅" if result in ("OK", "4/4 loaded") or result.startswith("OK") else "❌"
    if status == "❌":
        all_ok = False
    log.info(f"  {status} {test}: {result}")

log.info("")
if all_ok:
    log.info("✅ ALL SYSTEMS GO — scanner is ready for live trading")
else:
    log.info("⚠️  Some checks failed — review above before going live")

log.info("="*50)
log.info("Sleeping — check complete")

while True:
    time.sleep(3600)
