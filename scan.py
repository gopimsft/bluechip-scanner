"""
Blue-chip scanner (Yahoo Finance MVP) — NO pandas_ta / NO numba

Adds:
- Two candidate modes:
  1) Dip-reversion: down on the day + oversold-ish + not catastrophically below MAs + target within PROFIT_TARGET_PCT
  2) Trend-bounce: strong trend (price above SMA20/50) + red/flat day + target = recent swing high (20D) capped at PROFIT_TARGET_PCT

- Robust time-window gating (handles windows that wrap midnight)
- Optional market-calendar gating (disabled by default)
- Built-in universe: **always uses the ticker list inside this script** (no env vars needed)

Install:
  pip install yfinance pandas lxml beautifulsoup4 pandas_market_calendars apscheduler

SMTP (optional):
  SMTP_HOST, SMTP_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_TO

Key env vars (optional):
  SCAN_DROP_PCT         default -1.5   (names down more than this are included in scan table)
  MIN_DIP_PCT           default -1.0   (minimum dip for Dip-reversion candidates)
  PROFIT_TARGET_PCT     default 5.0    (your profit cap)
  MAX_NAMES             default 10
  SESSION_START_ET      default 04:00
  SESSION_END_ET        default 20:00
  TIMEZONE              default America/New_York
  SKIP_MARKET_CALENDAR  default 1
  MODE                  once | daemon  (default once)

Weekend/off-hours testing (no env vars needed):
  Set FORCE_RUN = True and (optionally) FORCE_ALL_DAY_WINDOW = True in the Config section.
"""

from __future__ import annotations

import os
import re
import math
import smtplib
import csv
from io import StringIO
from datetime import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from apscheduler.schedulers.blocking import BlockingScheduler
import pandas_market_calendars as mcal


# ---------------------------
# Config
# ---------------------------
FORCE_RUN = True  # True = allow running on weekends (useful for testing). Set to False for normal behavior.
FORCE_ALL_DAY_WINDOW = True  # True = ignore SESSION_START/END and run 24h (useful for testing)

# Output formatting
# - pretty: grouped multi-line blocks (original)
# - csv:    one row per ticker, comma-separated (easy copy/paste into a .csv)
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "csv").strip().lower()  # pretty | csv
CSV_INCLUDE_HEADER = os.getenv("CSV_INCLUDE_HEADER", "1") == "1"

# Printing toggle
# - False: print all rows (grouped by TrendSignal)
# - True:  print only rows that have a non-empty EntrySignal (e.g., CLEAR_BUY / WAIT_PULLBACK / DIP_SETUP)
PRINT_ONLY_ENTRY_SIGNAL = False

# Trend filter strictness for "buy high, sell higher" style scanning.
# Options: "LOOSE" | "MEDIUM" | "STRICT"
# - LOOSE:   Up-ish (Last >= SMA20) + mild red/flat day. (More signals, more false positives)
# - MEDIUM:  Uptrend (Last >= SMA50 and SMA50 >= SMA200) + mild pullback. (Good default)
# - STRICT:  Strong uptrend (SMA20 > SMA50 > SMA200, Last >= SMA50) + near recent highs. (Fewest signals)
TREND_MODE = "MEDIUM"

# For TREND candidates, we only want small pullbacks (not big down days).
TREND_PULLBACK_MIN_PCT = -3.0   # don't buy if it's down more than this in a day
TREND_PULLBACK_MAX_PCT = 0.25   # allow slightly green/flat days

# STRICT mode: require price to be within this % of the 20-day swing high (proxy for "near highs").
NEAR_HIGH_WITHIN_PCT = 10.0

DROP_THRESHOLD_PCT = float(os.getenv("SCAN_DROP_PCT", "-1.5"))  # scan table includes <= this (negative)
MIN_DIP_PCT = float(os.getenv("MIN_DIP_PCT", "-1.0"))          # Dip-reversion candidates require <= this
PROFIT_TARGET_PCT = float(os.getenv("PROFIT_TARGET_PCT", "5.0"))

# EntrySignal tagging (adds extra labels; does NOT change candidate logic)
ENTRY_SMA20_MAX_ABS_PCT = 2.0   # |%BelowSMA20| <= this => "near SMA20"
ENTRY_MIN_TARGET_PCT = 2.0      # require at least this much room to target for CLEAR_BUY
ENTRY_RSI_MIN = 40.0
ENTRY_RSI_MAX = 70.0

MAX_NAMES = int(os.getenv("MAX_NAMES", "500"))
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
SKIP_MARKET_CALENDAR = os.getenv("SKIP_MARKET_CALENDAR", "1") == "1"

SESSION_START_ET = os.getenv("SESSION_START_ET", "04:00")
SESSION_END_ET = os.getenv("SESSION_END_ET", "20:00")

SUBJECT_PREFIX = os.getenv("SUBJECT_PREFIX", "Blue-chip scan")

# Universe selection
# Provide tickers via env var (ideal for GitHub Actions) and fall back to the script list.
# Supported env vars (first non-empty wins):
#   TICKERS, SCAN_TICKERS
TICKERS_ENV_RAW = (os.getenv("TICKERS", "") or os.getenv("SCAN_TICKERS", "")).strip()
UNIVERSE_NAME = "ENV_TICKERS" if TICKERS_ENV_RAW else "SCRIPT_LIST"

# Performance toggle: fetching Yahoo news for every ticker can be slow and sometimes rate-limited.
# For "print all tickers" mode, default to fetching news only for meaningful movers/candidates.
FETCH_NEWS_FOR_ALL = False

# Yahoo throttling: keep requests modest
RUN_PREPOST = True
YF_THREADS = True
INTRADAY_INTERVAL = "2m"   # friendlier than 1m
INTRADAY_PERIOD = "2d"
DAILY_LOOKBACK_DAYS = 340  # enough for SMA200 buffer


# ---------------------------
# Indicators (pure pandas)
# ---------------------------
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------
# Time/calendar gating
# ---------------------------
def parse_hhmm(s: str) -> time:
    m = re.match(r"^(\d{1,2}):(\d{2})$", s.strip())
    if not m:
        raise ValueError(f"Bad time format '{s}', expected HH:MM")
    return time(hour=int(m.group(1)), minute=int(m.group(2)))

def get_local_now(tz: str) -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_convert(tz)

def in_run_window(now_ts: pd.Timestamp) -> bool:
    """
    Handles normal windows (04:00–20:00) and windows that wrap midnight (20:00–03:59).
    """
    if FORCE_ALL_DAY_WINDOW:
        return True

    start_t = parse_hhmm(SESSION_START_ET)
    end_t = parse_hhmm(SESSION_END_ET)
    t = now_ts.time()

    if start_t <= end_t:
        return start_t <= t <= end_t
    # wraps midnight
    return (t >= start_t) or (t <= end_t)

def is_market_day(now_ts: pd.Timestamp) -> bool:
    """
    Best-effort NYSE holiday gate. Disabled by default because calendars can vary by environment.
    """
    nyse = mcal.get_calendar("NYSE")
    day_et = now_ts.tz_convert(TIMEZONE).normalize()
    start = (day_et - pd.Timedelta(days=7)).date()
    end = (day_et + pd.Timedelta(days=7)).date()

    sched = nyse.schedule(start_date=str(start), end_date=str(end))
    idx = sched.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    sched_days_et = idx.tz_convert(TIMEZONE).normalize()
    return day_et in sched_days_et


# ---------------------------
# Email
# ---------------------------
def send_email(subject: str, body: str, attachments: Optional[List[Tuple[str, bytes, str]]] = None) -> None:
    """
    Send an email. If attachments are provided, each is (filename, bytes_content, mime_type).
    Example mime_type: "text/csv"
    """
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    email_user = os.environ["EMAIL_USER"]
    email_pass = os.environ["EMAIL_PASS"]
    email_to = os.environ["EMAIL_TO"]

    if attachments:
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "plain"))
        for fname, content, mime_type in attachments:
            maintype, subtype = (mime_type.split("/", 1) + ["octet-stream"])[:2]
            part = MIMEApplication(content, _subtype=subtype)
            part.add_header("Content-Disposition", "attachment", filename=fname)
            msg.attach(part)
    else:
        msg = MIMEText(body)

    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = email_to

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as s:
        s.login(email_user, email_pass)
        s.send_message(msg)

# ---------------------------
# Universe
# ---------------------------
def _normalize_tickers(tickers: List[str]) -> List[str]:
    """Normalize to Yahoo-friendly symbols and dedupe while preserving order."""
    seen = set()
    out: List[str] = []
    for t in tickers:
        if not t:
            continue
        sym = str(t).strip().upper().replace(".", "-")
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out

def load_universe() -> pd.DataFrame:
    """
    Uses env var tickers if provided (ideal for GitHub Actions), otherwise falls back to the
    ticker list defined inside this script.

    Supported env vars (first non-empty wins): TICKERS, SCAN_TICKERS
    Example: TICKERS="AAPL,MSFT,NVDA"
    """
    raw = (os.getenv("TICKERS", "") or os.getenv("SCAN_TICKERS", "")).strip()
    if raw:
        tickers = re.split(r"[\s,;]+", raw)
        tickers = _normalize_tickers(tickers)
        return pd.DataFrame({"Symbol": tickers})

    tickers = [
        # Monday
        "NTSK", "KLAR", "DPZ", "AXSM", "D", "STEP", "FRGT", "EMA", "FRPT", "LINC",
        # Tuesday
        "HIMS", "BWXT", "KTOS", "KEYS", "PRIM", "OVV", "OKE", "OPAD", "PLAB",
        # Wednesday
        "CIFR", "HD", "DOCN", "NRG", "FIS", "XMTR", "AMR", "LTH", "PLNT", "HPQ",
        # Thursday
        "AMC", "MELI", "CAVA", "AXON", "ZETA", "NVTS", "WDAY", "TEM", "HUT", "TJX", "TRIB",
        # Friday
        "NVDA", "TTD", "CRM", "SNOW", "IONQ", "SNPS", "ARRY", "P", "VICI", "NTNX",
        # Extended List from Search Results
        "WAVE", "CELH", "VST", "ACMR", "EOS", "BIDU", "Q", "RKLB", "WBD", "GCT", "CRWV", 
        "S", "MARA", "INOD", "DELL", "OPK", "SOUN", "ZS", "DUOL", "COMP", "UUUU",  
        "GSAT", "VIA", "TCPC", "SUI", "NWN",
        "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
        "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
        "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WMT", 
        "docn", "cava", "figr", "gral", "frog", "afrm", "snap", "team", "arm", "flnc", 
        "qcom", "nvo", "jci", "bsx", "adnt", "ctsh", "amcr", "cmg", "nxpi", "pypl", 
        "rblx", "iqv", "lin", "gold", "el", "uaa", "cme", "tsn", "nssc", "hesm", 
        "bsrr", "idxx", "etn", "dva", "wwd", "spg", "fn", "rmbs", "it", "adm", 
        "glxy", "enph", "elf", "sym", "cci", "ubs", "cmi", "clne", "mrvl", "mrna", 
        "deck", "cava", "etsy", "bmnr", "path", "tlry", "crwv", "apld", "cpng", "orcl", 
        "corz", "s", "pd", "fivn", "rzlv", "fubo", "ddog", "fig", "pl", "abvx", 
        "snow", "nvda", "crwd", "ntnx", "veev", "urbn", "iren", "dell", "baba", 
        "ibm", "hpe", "dlo", "axsm", "hims", "hood", "cvna", "gev", "googl", "rklb", 
        "pins", "smr", "rivn", "vst", "u", "grab", "pltr", "vrt", "ttd", "adbe", 
        "lulu", "algn", "ttan", "iot", "snps", "rbrk", "avo", "sail", "fcel", 
        "gtlb", "docu", "ai", "crm", "okta", "mdb", "box", "ncno", "kss", "anf", 
        "hpq", "ntap", "bbwi", "bby", "dks", "gap", "amba", "estc", "woof", "gen", 
        "mndy", "nu", "ce", "wix", "wday", "akam", "abnb", "fisv", "regn", 
        "four", "twlo", "crcl", "fitb", "unh", "amd", "intc", "v", "now", "isrg", 
        "xyz", "gddy", "on", "crox", "docn", "kvyo", "bill", "pi", "swks", "asan", 
        "glob", "farm", "rfil", "kmts", "hoft", "vra", "kalv", "culp", "vnce", "love", 
        "poet", "lsak", "oxm", "le", "dakt", "biox", "lmnr", "cvgw", "casy", 
        "mama", "cnm", "dbi", "kfy", "cgnt", "pvh", "joyy", "ggal", "ooma", "tuya", 
        "sjm", "five", "schw", "pld", "mrtn", "hwc", "rexr", "mtb", "nvs", 
        "tsla", "mp", "gild", "frog", "wynn", "dbx", "serv", "chym", "mchp", "msi", 
        "trip", "arry", "bird", "aeye", "expe", "real", "eton", "kins", "main", "oust", 
        "iova", "mnst", "tem", "lamr", "bbai", "oklo", "amc", "rgti", "asts", "achr", 
        "amat", "csco", "mvst", "b", "ag", "pnnt", "plug", "gral", "eat", "zvra", 
        "eqx", "fnv", "kopn", "kulr", "gpro", "pony", "pflt", "linc", "jd", "wprt", 
        "tmc", "rekr", "sndk", "ww", "de", "ddd", "ee", "gamb", "lqda", "hrb", 
        "onon", "nvgs", "ibta", "zm", "avgo", "ba", "celh", "cpri", "dxcm", "fslr", 
        "lly", "lyft", "mbly", "msft", "mstr", "mu", "nke", "qrvo", "roku", "sedg", 
        "smci", "sofi", "tmdx", "uber", "zeta", "zs", "crdo", "cohr", "shop", "meta", 
        "amzn", "a", "^ndx", "^gspc", "oscr", "vsco", "aaon", "upst", "pdd", 
        "w", "frpt", "bntx", "meli", "axon", "nvts", "vrtx", "lscc", "tdup", "lmnd", 
        "evtl", "open", "alab", "fet", "trgp", "ktos", "wulf", "zts", "docs", "entg", 
        "frmi", "aal", "aptv", "bax", "bbio", "bldr", "blsh", "burl", "dow", "futu", 
        "gfs", "gh", "dhi", "ir", "podd", "ip", "len", "mga", "mgm", "odfl", 
        "oc", "bros", "phm", "qxo", "rrx", "rkt", "rost", "stla", "swk", "luv", 
        "xpo", "zbra", "pgy", "fsly", "mara", "cifr", "sg", "cvlt", "fisv", "net", "dt",
        "PTRN", "AVAV", "BBY", "BOX", "OKTA", "VEEV", "OUST", "WIX", "EVGO", "GTLB", "AEYE", "IOT", "ACHR", "ONON", "ANF", "AEO", "TGT", "AAON", "CIEN", "U", "BULL",
    ]
    
    tickers = _normalize_tickers(tickers)
    return pd.DataFrame({"Symbol": tickers})


# ---------------------------
# Data helpers
# ---------------------------
def pct_change(last_price: float, ref_price: float) -> float:
    if ref_price and ref_price > 0:
        return (last_price / ref_price - 1.0) * 100.0
    return float("nan")
def _nyse_session_times_et(now_et: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return (market_open_et, market_close_et) for the NYSE trading day containing now_et.
    Uses pandas_market_calendars when possible (handles early closes), otherwise falls back
    to 09:30–16:00 ET.
    """
    day = now_et.tz_convert(TIMEZONE).normalize()
    default_open = day + pd.Timedelta(hours=9, minutes=30)
    default_close = day + pd.Timedelta(hours=16)

    try:
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=str(day.date()), end_date=str(day.date()))
        if sched is None or sched.empty:
            return default_open, default_close

        row = sched.iloc[0]
        mkt_open = row.get("market_open", None)
        mkt_close = row.get("market_close", None)
        if mkt_open is None or mkt_close is None:
            return default_open, default_close

        # Ensure tz-aware then convert to TIMEZONE
        if getattr(mkt_open, "tz", None) is None:
            mkt_open = pd.Timestamp(mkt_open).tz_localize("UTC")
        if getattr(mkt_close, "tz", None) is None:
            mkt_close = pd.Timestamp(mkt_close).tz_localize("UTC")

        return mkt_open.tz_convert(TIMEZONE), mkt_close.tz_convert(TIMEZONE)
    except Exception:
        return default_open, default_close


def _is_regular_session_et(now_et: pd.Timestamp) -> bool:
    open_et, close_et = _nyse_session_times_et(now_et)
    return open_et <= now_et.tz_convert(TIMEZONE) <= close_et


def chunked(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]

def fetch_quotes(symbols: List[str]) -> pd.DataFrame:
    """
    "Last" comes from the most recent intraday bar (with pre/post when RUN_PREPOST=True).

    "PrevClose" is chosen to keep % change intuitive:
      - During regular session: yesterday's *regular-session* close.
      - After regular close: today's *regular-session* close (so after-hours move is vs the close).
      - Premarket: yesterday's close (normal).

    Why: Yahoo's daily "Close" for *today* can update intraday, which makes % change look wrong
    during the regular session if you treat today's evolving value as "PrevClose".
    """
    now_et = get_local_now(TIMEZONE)
    now_local = now_et.tz_convert(TIMEZONE)
    in_regular = _is_regular_session_et(now_et)
    open_et, close_et = _nyse_session_times_et(now_et)
    today = now_local.date()

    def _to_tz_series(s: pd.Series) -> pd.Series:
        idx = s.index
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        s2 = s.copy()
        s2.index = idx.tz_convert(TIMEZONE)
        return s2

    rows = []
    for batch in chunked(symbols, 50):
        tickers = " ".join(batch)

        intraday = yf.download(
            tickers=tickers,
            period=INTRADAY_PERIOD,
            interval=INTRADAY_INTERVAL,
            prepost=RUN_PREPOST,
            group_by="ticker",
            auto_adjust=False,
            threads=YF_THREADS,
            progress=False,
        )

        daily = yf.download(
            tickers=tickers,
            period="10d",
            interval="1d",
            prepost=False,
            group_by="ticker",
            auto_adjust=False,
            threads=YF_THREADS,
            progress=False,
        )

        for sym in batch:
            try:
                if isinstance(intraday.columns, pd.MultiIndex):
                    last_series = intraday[sym]["Close"].dropna()
                else:
                    last_series = intraday["Close"].dropna()

                if last_series.empty:
                    continue
                last_price = float(last_series.iloc[-1])

                if isinstance(daily.columns, pd.MultiIndex):
                    dclose = daily[sym]["Close"].dropna()
                else:
                    dclose = daily["Close"].dropna()

                if dclose.empty:
                    continue

                # Default: use last available daily close
                prev_close = float(dclose.iloc[-1])

                # If today's daily bar exists and we're *in the regular session*,
                # use yesterday's close as the reference (avoid intraday-updating daily close).
                last_daily_date = pd.Timestamp(dclose.index[-1]).date()
                if in_regular and last_daily_date == today and len(dclose) >= 2:
                    prev_close = float(dclose.iloc[-2])

                # After-hours robustness: if daily hasn't updated to include today's close yet,
                # approximate today's regular close from intraday bars at/just before market close.
                if (not in_regular) and (now_local > close_et) and (last_daily_date < today):
                    s_et = _to_tz_series(last_series)
                    mask = (
                        (s_et.index.date == today)
                        & (s_et.index >= open_et)
                        & (s_et.index <= close_et)
                    )
                    if mask.any():
                        prev_close = float(s_et.loc[mask].iloc[-1])

                chg = pct_change(last_price, prev_close)
                rows.append((sym, last_price, prev_close, chg))
            except Exception:
                continue

    return pd.DataFrame(rows, columns=["Symbol", "Last", "PrevClose", "ChgPct"])


def fetch_daily_bars(symbol: str, lookback_days: int = DAILY_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Robust daily bars fetch for a single symbol.
    Returns standardized columns: open, high, low, close, volume
    """
    try:
        df = yf.download(
            tickers=symbol,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=False,
            threads=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        elif symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, axis=1, level=0)

    df.columns = [str(c).strip() for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    out = df[required].copy().rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })

    for c in ["open","high","low","close","volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.dropna()


def compute_technicals(daily_df: pd.DataFrame) -> dict:
    if daily_df is None or daily_df.empty or "close" not in daily_df.columns:
        return {}

    close = daily_df["close"].astype(float)
    if len(close) < 50:
        return {}

    rsi14 = rsi_wilder(close, 14).iloc[-1]
    sma20v = sma(close, 20).iloc[-1]
    sma50v = sma(close, 50).iloc[-1]
    sma200v = sma(close, 200).iloc[-1] if len(close) >= 200 else float("nan")

    # 20-day swing high (use "high" if present, otherwise close)
    if "high" in daily_df.columns and len(daily_df["high"].dropna()) >= 20:
        swing_high_20 = float(daily_df["high"].rolling(20, min_periods=20).max().iloc[-1])
    else:
        swing_high_20 = float(close.rolling(20, min_periods=20).max().iloc[-1])

    return {
        "rsi14": float(rsi14) if not pd.isna(rsi14) else float("nan"),
        "sma20": float(sma20v) if not pd.isna(sma20v) else float("nan"),
        "sma50": float(sma50v) if not pd.isna(sma50v) else float("nan"),
        "sma200": float(sma200v) if not pd.isna(sma200v) else float("nan"),
        "swing_high_20": swing_high_20,
        "last_daily_close": float(close.iloc[-1]),
    }


def fetch_yahoo_news(symbol: str, limit: int = 6) -> List[str]:
    try:
        t = yf.Ticker(symbol)
        news = t.news or []
        heads: List[str] = []
        for item in news[:limit]:
            title = (item or {}).get("title")
            if title:
                heads.append(str(title).strip())
        return heads
    except Exception:
        return []


# ---------------------------
# "Why down" classifier (simple rules)
# ---------------------------
def classify_drop(headlines: List[str]) -> str:
    text = " ".join(headlines).lower()
    if any(k in text for k in ["earnings", "eps", "guidance", "revenue", "quarter", "q1", "q2", "q3", "q4"]):
        return "Earnings / guidance related"
    if any(k in text for k in ["downgrade", "upgrade", "price target", "initiated", "initiates"]):
        return "Analyst action"
    if any(k in text for k in ["sec", "doj", "ftc", "lawsuit", "investigation", "settlement", "regulator", "recall"]):
        return "Regulatory / legal"
    if any(k in text for k in ["outage", "breach", "hack", "incident", "cyber"]):
        return "Operational / security incident"
    if headlines:
        return "News catalyst (uncategorized)"
    return "No obvious Yahoo headline catalyst"


# ---------------------------
# Candidate logic
# ---------------------------
def pct_diff(a: float, b: float) -> float:
    """(a/b - 1) * 100"""
    if b and b > 0:
        return (a / b - 1.0) * 100.0
    return float("nan")

def pick_nearest_resistance(last_price: float, tech: dict) -> Tuple[Optional[str], float, float]:
    """
    Return (level_name, level_price, upside_pct) for the nearest MA above price.
    """
    candidates = []
    for k in ("sma20", "sma50", "sma200"):
        v = tech.get(k, float("nan"))
        if isinstance(v, float) and not math.isnan(v) and v > last_price:
            candidates.append((k, v))
    if not candidates:
        return None, float("nan"), float("nan")
    k, v = min(candidates, key=lambda kv: kv[1])
    up = pct_diff(v, last_price)
    return k, float(v), float(up)

def decide_candidate(last_price: float, chg_pct: float, tech: dict) -> dict:
    """
    Outputs:
      CandidateType: "DIP" | "TREND" | "" (blank means not a candidate)
      TargetPrice, TargetUpPct, TargetReason
    """
    rsi = tech.get("rsi14", float("nan"))
    sma20v = tech.get("sma20", float("nan"))
    sma50v = tech.get("sma50", float("nan"))
    sma200v = tech.get("sma200", float("nan"))
    swing_high_20 = tech.get("swing_high_20", float("nan"))

    # Features
    pct_below_sma20 = pct_diff(last_price, sma20v)  # negative if below
    pct_below_sma50 = pct_diff(last_price, sma50v)

    # Trend signal tag (independent of TREND_MODE and candidate selection)
    loose_trend = (not math.isnan(sma20v)) and (last_price >= sma20v)
    medium_trend = (not math.isnan(sma50v)) and (last_price >= sma50v) and (not math.isnan(sma200v)) and (sma50v >= sma200v)
    strict_trend = medium_trend and (not math.isnan(sma20v)) and (sma20v > sma50v)

    near_high = False
    if not math.isnan(swing_high_20) and swing_high_20 > 0:
        near_high = pct_diff(last_price, swing_high_20) >= -NEAR_HIGH_WITHIN_PCT

    trend_signal = ""
    if strict_trend and near_high:
        trend_signal = "STRICT"
    elif medium_trend:
        trend_signal = "MEDIUM"
    elif loose_trend:
        trend_signal = "LOOSE"

    # --- Dip-reversion candidate ---
    # Conditions: meaningful dip, oversold-ish, and not totally broken vs SMA20 (avoid 30%+ below)
    if (chg_pct <= MIN_DIP_PCT) and (not math.isnan(rsi) and rsi <= 35.0) and (not math.isnan(pct_below_sma20) and pct_below_sma20 >= -20.0):
        lvl_name, lvl_price, up = pick_nearest_resistance(last_price, tech)

        # target is min(nearest MA above, capped 5% up)
        target_cap = last_price * (1.0 + PROFIT_TARGET_PCT / 100.0)

        if lvl_name and not math.isnan(lvl_price):
            target = min(lvl_price, target_cap)
            up_target = pct_diff(target, last_price)
            if up_target <= PROFIT_TARGET_PCT + 1e-9:
                return {
                    "CandidateType": "DIP",
                    "TrendSignal": trend_signal,
                    "TargetPrice": round(target, 2),
                    "TargetUpPct": round(up_target, 2),
                    "TargetReason": f"Nearest {lvl_name} (capped at {PROFIT_TARGET_PCT:.1f}%)",
                    "PctBelowSMA20": round(pct_below_sma20, 2),
                    "PctBelowSMA50": round(pct_below_sma50, 2),
                }

        # If no MA above, still allow target = cap if swing_high_20 within cap
        if not math.isnan(swing_high_20):
            up_swing = pct_diff(swing_high_20, last_price)
            if 0 < up_swing <= PROFIT_TARGET_PCT:
                return {
                    "CandidateType": "DIP",
                    "TrendSignal": trend_signal,
                    "TargetPrice": round(swing_high_20, 2),
                    "TargetUpPct": round(up_swing, 2),
                    "TargetReason": "20D swing high",
                    "PctBelowSMA20": round(pct_below_sma20, 2),
                    "PctBelowSMA50": round(pct_below_sma50, 2),
                }

    # --- Trend-bounce signals ("buy high, sell higher") ---
    # We want mild pullbacks inside an uptrend.
    in_pullback_band = (TREND_PULLBACK_MIN_PCT <= chg_pct <= TREND_PULLBACK_MAX_PCT)

    # Trend definitions already computed above; we now apply TREND_MODE gating.

    mode = str(TREND_MODE).strip().upper()
    if mode == "STRICT":
        trend_ok = strict_trend and near_high
    elif mode == "LOOSE":
        trend_ok = loose_trend
    else:
        # default MEDIUM
        trend_ok = medium_trend

    # Final TREND gate
    if trend_ok and in_pullback_band:
        # target = min(20D swing high, cap)
        target_cap = last_price * (1.0 + PROFIT_TARGET_PCT / 100.0)
        if not math.isnan(swing_high_20):
            target = min(swing_high_20, target_cap)
            up_target = pct_diff(target, last_price)
            if 0 < up_target <= PROFIT_TARGET_PCT + 1e-9:
                return {
                    "CandidateType": "TREND",
                    "TrendSignal": trend_signal,
                    "TargetPrice": round(target, 2),
                    "TargetUpPct": round(up_target, 2),
                    "TargetReason": "20D swing high (capped)",
                    "PctBelowSMA20": round(pct_below_sma20, 2),
                    "PctBelowSMA50": round(pct_below_sma50, 2),
                }

        # fallback: just use cap
        return {
            "CandidateType": "TREND",
            "TrendSignal": trend_signal,
            "TargetPrice": round(target_cap, 2),
            "TargetUpPct": round(PROFIT_TARGET_PCT, 2),
            "TargetReason": f"{PROFIT_TARGET_PCT:.1f}% cap (no swing high)",
            "PctBelowSMA20": round(pct_below_sma20, 2),
            "PctBelowSMA50": round(pct_below_sma50, 2),
        }

    return {
        "CandidateType": "",
        "TrendSignal": trend_signal,
        "TargetPrice": None,
        "TargetUpPct": None,
        "TargetReason": "",
        "PctBelowSMA20": round(pct_below_sma20, 2) if not math.isnan(pct_below_sma20) else None,
        "PctBelowSMA50": round(pct_below_sma50, 2) if not math.isnan(pct_below_sma50) else None,
    }


def compute_entry_signal(last_price: float, chg_pct: float, tech: dict, cand: dict) -> str:
    # Extra tag only (does not affect CandidateType):
    #   - CLEAR_BUY: TREND + (TrendSignal STRICT/MEDIUM) + near SMA20 + enough room to target + RSI in band
    #   - WAIT_PULLBACK: TREND + (TrendSignal STRICT/MEDIUM/LOOSE) but not CLEAR_BUY
    #   - DIP_SETUP: DIP candidate
    #   - "" otherwise
    ctype = str(cand.get("CandidateType", "") or "")
    if ctype == "DIP":
        return "DIP_SETUP"

    if ctype != "TREND":
        return ""

    trend_sig = str(cand.get("TrendSignal", "") or "")
    # only consider strict/medium as "buy-high" quality
    if trend_sig not in ("STRICT", "MEDIUM"):
        return "WAIT_PULLBACK"

    rsi = tech.get("rsi14", float("nan")) if tech else float("nan")
    pct_below_sma20 = cand.get("PctBelowSMA20", None)
    target_up = cand.get("TargetUpPct", None)

    near_sma20 = False
    if isinstance(pct_below_sma20, (int, float)) and (pct_below_sma20 == pct_below_sma20):
        near_sma20 = abs(float(pct_below_sma20)) <= ENTRY_SMA20_MAX_ABS_PCT

    enough_room = False
    if isinstance(target_up, (int, float)) and (target_up == target_up):
        enough_room = float(target_up) >= ENTRY_MIN_TARGET_PCT

    rsi_ok = (isinstance(rsi, (int, float)) and (rsi == rsi) and (ENTRY_RSI_MIN <= float(rsi) <= ENTRY_RSI_MAX))

    if near_sma20 and enough_room and rsi_ok:
        return "CLEAR_BUY"

    return "WAIT_PULLBACK"


# ---------------------------
# Core scan
# ---------------------------
def run_scan() -> pd.DataFrame:
    now_et = get_local_now(TIMEZONE)
    """
    if (not FORCE_RUN) and (now_et.weekday() >= 5):
        print(f"Skipping scan: weekend. ET now={now_et}")
        return pd.DataFrame()

    if not in_run_window(now_et):
        print(f"Skipping scan: outside scan window ({SESSION_START_ET}–{SESSION_END_ET} ET). ET now={now_et}")
        return pd.DataFrame()

    if (not SKIP_MARKET_CALENDAR) and (not is_market_day(now_et)):
        print(f"Skipping scan: NYSE holiday. ET now={now_et}")
        return pd.DataFrame()

    """

    uni = load_universe()
    symbols = uni["Symbol"].tolist()

    quotes = fetch_quotes(symbols)
    if quotes.empty:
        print("No quotes returned (Yahoo throttled or symbols invalid).")
        return pd.DataFrame()

    # Always process every ticker that returned quotes (no filtering).
    # Note: Yahoo may not return quotes for every requested symbol.
    quotes = quotes.sort_values("ChgPct").reset_index(drop=True)
    movers = quotes  # name retained for minimal changes downstream

    report_rows = []
    for _, r in movers.iterrows():
        sym = str(r["Symbol"])
        last_price = float(r["Last"])
        prev_close = float(r["PrevClose"])
        chg = float(r["ChgPct"])

        daily_bars = fetch_daily_bars(sym)
        tech = compute_technicals(daily_bars)

        # News can be slow/rate-limited; default to fetching only for meaningful movers/candidates.
        headlines: List[str] = []
        reason = "(news skipped)"

        lvl_name, lvl_price, up_ma = pick_nearest_resistance(last_price, tech)
        within_5_ma = (not math.isnan(up_ma)) and (up_ma <= PROFIT_TARGET_PCT)

        cand = decide_candidate(last_price, chg, tech)

        entry_signal = compute_entry_signal(last_price, chg, tech, cand)

        if FETCH_NEWS_FOR_ALL or (chg <= DROP_THRESHOLD_PCT) or (cand.get("CandidateType") != ""):
            headlines = fetch_yahoo_news(sym, limit=6)
            reason = classify_drop(headlines)
        else:
            # Keep the default label; if you prefer a cleaner output, change to "".
            reason = "(news skipped)"

        report_rows.append({
            "Symbol": sym,
            "ChgPct_vsPrevClose": round(chg, 2),
            "Last": round(last_price, 2),
            "PrevClose": round(prev_close, 2),
            "RSI14": round(float(tech.get("rsi14", float("nan"))), 2) if tech else None,
            "SMA20": round(float(tech.get("sma20", float("nan"))), 2) if tech else None,
            "SMA50": round(float(tech.get("sma50", float("nan"))), 2) if tech else None,
            "SMA200": round(float(tech.get("sma200", float("nan"))), 2) if tech else None,
            "SwingHigh20": round(float(tech.get("swing_high_20", float("nan"))), 2) if tech else None,
            "NearestMA": lvl_name,
            "NearestMAUp%": round(up_ma, 2) if not math.isnan(up_ma) else None,
            "Within5%MA": bool(within_5_ma),
            "CandidateType": cand["CandidateType"],
            "TrendSignal": cand.get("TrendSignal", ""),
            "EntrySignal": entry_signal,
            "TargetPrice": cand["TargetPrice"],
            "TargetUp%": cand["TargetUpPct"],
            "TargetReason": cand["TargetReason"],
            "%BelowSMA20": cand["PctBelowSMA20"],
            "%BelowSMA50": cand["PctBelowSMA50"],
            "Why": reason,
            "Headlines": " | ".join(headlines[:3]),
        })

    report = pd.DataFrame(report_rows)
    if not report.empty:
        report = report.sort_values(["TrendSignal", "ChgPct_vsPrevClose"], ascending=[True, True])
    return report



def format_report_pretty(df: pd.DataFrame) -> str:
    now_et = get_local_now(TIMEZONE)
    header = [
        f"Scan time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Universe: {UNIVERSE_NAME}",
        f"Scan threshold (table): ChgPct <= {DROP_THRESHOLD_PCT:.2f}%",
        f"Candidate rules: MIN_DIP_PCT={MIN_DIP_PCT:.2f}%, PROFIT_TARGET_PCT={PROFIT_TARGET_PCT:.1f}%",
        f"Trend mode: {str(TREND_MODE).strip().upper()} | Trend pullback band: {TREND_PULLBACK_MIN_PCT:.2f}% to {TREND_PULLBACK_MAX_PCT:.2f}% | Near-high (STRICT): {NEAR_HIGH_WITHIN_PCT:.1f}%",
        "",
    ]
    if df.empty:
        return "\n".join(header + ["No rows (skipped or no quotes returned)."])

    # Group by TrendSignal, then sort within each group by ChgPct (most fallen -> most increase).
    # Empty TrendSignal means: it doesn't meet the LOOSE condition (Last < SMA20) OR SMA20 is missing.
    group_order = ["STRICT", "MEDIUM", "LOOSE", ""]
    group_titles = {
        "STRICT": "STRICT (strong uptrend leaders)",
        "MEDIUM": "MEDIUM (uptrend)",
        "LOOSE": "LOOSE (above SMA20)",
        "": "NO TREND SIGNAL (below SMA20 or missing MAs)",
    }

    lines = header
    for g in group_order:
        gdf = df[df["TrendSignal"] == g].copy()
        if PRINT_ONLY_ENTRY_SIGNAL:
            gdf = gdf[gdf["EntrySignal"].astype(str).str.strip() != ""]
        if gdf.empty:
            continue
        gdf = gdf.sort_values("ChgPct_vsPrevClose", ascending=True)
        lines.append("=" * 72)
        lines.append(group_titles[g])
        lines.append("=" * 72)
        lines.append("")

        for _, r in gdf.iterrows():
            lines += [
                f"{r['Symbol']}: {r['ChgPct_vsPrevClose']:.2f}%  Last={r['Last']:.2f}  PrevClose={r['PrevClose']:.2f}",
                f"  RSI14={r['RSI14']}  SMA20={r['SMA20']}  SMA50={r['SMA50']}  SMA200={r['SMA200']}  SwingHigh20={r['SwingHigh20']}",
                f"  NearestMA={r['NearestMA']}  NearestMAUp%={r['NearestMAUp%']}  Within5%MA={r['Within5%MA']}",
                f"  TrendSignal={r.get('TrendSignal','')}  EntrySignal={r.get('EntrySignal','')}  CandidateType={r['CandidateType']}  Target={r['TargetPrice']} ({r['TargetUp%']}%)  TargetReason={r['TargetReason']}",
                f"  %BelowSMA20={r['%BelowSMA20']}  %BelowSMA50={r['%BelowSMA50']}",
                f"  Why: {r['Why']}",
            ]
            if r.get("Headlines"):
                lines.append(f"  Headlines: {r['Headlines']}")
            lines.append("")

    return "\n".join(lines)


def format_report_csv(df: pd.DataFrame) -> str:
    """Return a pure CSV string (no extra preamble lines)."""
    if df is None or df.empty:
        # still return header if requested (so you can paste into a CSV and keep columns consistent)
        cols = [
            "ticker","change_pct","last","prev_close","rsi14","sma20","sma50","sma200","swing_high_20",
            "nearest_ma","nearest_ma_up_pct","within5pct_ma",
            "trend_signal","entry_signal","candidate_type",
            "target_price","target_up_pct","target_reason",
            "pct_below_sma20","pct_below_sma50",
            "why","headlines",
        ]
        if CSV_INCLUDE_HEADER:
            return ",".join(cols)
        return ""

    # Keep same filtering behavior as pretty output when PRINT_ONLY_ENTRY_SIGNAL=True
    out_df = df.copy()
    if PRINT_ONLY_ENTRY_SIGNAL:
        out_df = out_df[out_df["EntrySignal"].astype(str).str.strip() != ""]

    # Order + rename columns to match "ticker, change, last, prev close, RSI14, SMA20..." (no name=value pairs)
    col_map = [
        ("Symbol", "ticker"),
        ("ChgPct_vsPrevClose", "change_pct"),
        ("Last", "last"),
        ("PrevClose", "prev_close"),
        ("RSI14", "rsi14"),
        ("SMA20", "sma20"),
        ("SMA50", "sma50"),
        ("SMA200", "sma200"),
        ("SwingHigh20", "swing_high_20"),
        ("NearestMA", "nearest_ma"),
        ("NearestMAUp%", "nearest_ma_up_pct"),
        ("Within5%MA", "within5pct_ma"),
        ("TrendSignal", "trend_signal"),
        ("EntrySignal", "entry_signal"),
        ("CandidateType", "candidate_type"),
        ("TargetPrice", "target_price"),
        ("TargetUp%", "target_up_pct"),
        ("TargetReason", "target_reason"),
        ("%BelowSMA20", "pct_below_sma20"),
        ("%BelowSMA50", "pct_below_sma50"),
        ("Why", "why"),
        ("Headlines", "headlines"),
    ]

    cols_in = [c for c, _ in col_map if c in out_df.columns]
    out_df = out_df[cols_in].rename(columns={c: new for c, new in col_map if c in out_df.columns})

    # Stable ordering
    sort_cols = []
    if "trend_signal" in out_df.columns:
        sort_cols.append("trend_signal")
    if "change_pct" in out_df.columns:
        sort_cols.append("change_pct")
    if sort_cols:
        out_df = out_df.sort_values(sort_cols, ascending=[True] * len(sort_cols))

    # Convert NaN/None to empty strings so you don't paste "nan" into spreadsheets
    out_df = out_df.where(out_df.notna(), "")

    # Use csv.writer to safely quote commas/newlines in text fields (Why/Headlines/TargetReason)
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    if CSV_INCLUDE_HEADER:
        writer.writerow(list(out_df.columns))

    for row in out_df.itertuples(index=False, name=None):
        writer.writerow(row)

    return buf.getvalue().rstrip("\n")


def format_report(df: pd.DataFrame) -> str:
    """Backwards-compatible formatter selector."""
    if str(OUTPUT_FORMAT).strip().lower() == "csv":
        return format_report_csv(df)
    return format_report_pretty(df)


def run_once_and_email() -> None:
    df = run_scan()

    # Console output (respects OUTPUT_FORMAT)
    body = format_report(df)
    print(body)

    # Always attach a CSV, regardless of the console format.
    csv_str = format_report_csv(df)
    now_et = get_local_now(TIMEZONE)
    subject = f"{SUBJECT_PREFIX} — {now_et.strftime('%Y-%m-%d %H:%M ET')}"
    fname = f"scan_{now_et.strftime('%Y%m%d_%H%M')}.csv"

    try:
        send_email(subject, body, attachments=[(fname, csv_str.encode("utf-8"), "text/csv")])
        print("Email sent (CSV attached).")
    except Exception as e:
        print("EMAIL FAILED:", repr(e))
        raise

if __name__ == "__main__":
    mode = os.getenv("MODE", "once").lower()

    if mode == "once":
        run_once_and_email()

    elif mode == "daemon":
        scheduler = BlockingScheduler(timezone=TIMEZONE)
        scheduler.add_job(run_once_and_email, "cron", day_of_week="mon-sun", minute="0,30")
        print("Scheduler started: running at :00 and :30 ET (cron: mon-sun).")
        scheduler.start()
    else:
        raise ValueError("MODE must be 'once' or 'daemon'")
