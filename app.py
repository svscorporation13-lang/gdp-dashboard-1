# app.py — Authoritarian Monitor (LITE) — copy/replace entire file with this

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import requests

# lazy imports for optional libs
def import_yfinance():
    try:
        import yfinance as yf
        return yf
    except Exception:
        return None

def import_pytrends():
    try:
        from pytrends.request import TrendReq
        return TrendReq
    except Exception:
        return None

# ---------------- Config ----------------
st.set_page_config(page_title="Authoritarian Monitor — Lite", layout="wide")
NEUTRAL = 0.5

COUNTRIES = {
    "Russia": {"wb": "RUS", "fx": "RUBUSD=X", "iso": "RU"},
    "China": {"wb": "CHN", "fx": "CNYUSD=X", "iso": "CN"},
    "Iran": {"wb": "IRN", "fx": "IRRUSD=X", "iso": "IR"},
    "Turkey": {"wb": "TUR", "fx": "TRYUSD=X", "iso": "TR"},
}

# ---------------- Helpers / Fetchers ----------------
@st.cache_data(ttl=60 * 60 * 6)
def fetch_fx(pair: str, period: str = "3y"):
    yf = import_yfinance()
    if yf is None:
        return None
    try:
        df = yf.download(pair, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        s = df["Close"].resample("M").last()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return None

@st.cache_data(ttl=60 * 60 * 24)
def fetch_gdp_per_capita(wb_code: str, start_year: int = 2015):
    try:
        end_year = dt.datetime.now().year
        url = f"https://api.worldbank.org/v2/country/{wb_code}/indicator/NY.GDP.PCAP.CD?date={start_year}:{end_year}&format=json&per_page=2000"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        df = pd.DataFrame(data[1])[["date", "value"]].dropna()
        df["date"] = pd.to_datetime(df["date"].astype(str) + "-01-01")
        s = df.set_index("date")["value"].resample("A").last()
        return s
    except Exception:
        return None

@st.cache_data(ttl=60 * 60 * 6)
def fetch_trends_protest(iso: str, timeframe: str = "today 5-y"):
    TrendReq = import_pytrends()
    if TrendReq is None:
        return None
    try:
        py = TrendReq(hl="en-US", tz=0)
        kw = ["protest"]
        py.build_payload(kw, timeframe=timeframe, geo=iso if iso else "")
        df = py.interest_over_time()
        if df is None or df.empty:
            return None
        s = df["protest"].resample("M").mean()
        return s
    except Exception:
        return None

# ---------------- Normalization (robust) ----------------
def safe_scale(series, inverse: bool = False):
    """Return pd.Series scaled to 0..1 or None if not enough data."""
    try:
        if series is None:
            return None
        if not isinstance(series, pd.Series):
            # try convertable sequence -> series
            try:
                series = pd.Series(series)
            except Exception:
                return None
        s = series.dropna().astype(float)
        if s.empty or s.shape[0] < 2:
            return None
        low = float(s.min())
        high = float(s.max())
        if abs(high - low) < 1e-12:
            # constant series -> neutral vector of same index
            return pd.Series(np.repeat(NEUTRAL, len(s)), index=s.index)
        scaled = (s - low) / (high - low)
        if inverse:
            scaled = 1.0 - scaled
        return scaled
    except Exception:
        return None

def last_or_neutral(s):
    if s is None:
        return NEUTRAL
    try:
        v = float(s.iloc[-1])
        if np.isnan(v):
            return NEUTRAL
        return float(v)
    except Exception:
        return NEUTRAL

# ---------------- UI ----------------
st.title("Authoritarian Regime Effectiveness — Lite Dashboard")
st.markdown("Simple, robust prototype. Scores 0..1 (1 = stronger regime).")

selected = st.sidebar.multiselect("Countries", list(COUNTRIES.keys()), default=["Russia", "China"])
start = st.sidebar.date_input("Start date", value=dt.date.today() - dt.timedelta(days=365 * 3))
if st.sidebar.button("Fetch / Update now"):
    fetch_flag = True
else:
    fetch_flag = False

# ---------------- Processing & Output ----------------
if not fetch_flag:
    st.warning("No data fetched. Click 'Fetch / Update now' in the sidebar to pull fresh data.")
    st.stop()

results = {}
errors = []

for country in selected:
    meta = COUNTRIES[country]
    # fetch raw data
    fx = fetch_fx(meta["fx"])
    gdp = fetch_gdp_per_capita(meta["wb"])
    trends = fetch_trends_protest(meta["iso"])

    # restrict to start date if applicable
    try:
        if fx is not None:
            fx = fx[fx.index >= pd.to_datetime(start)]
        if gdp is not None:
            gdp = gdp[gdp.index >= pd.to_datetime(start)]
        if trends is not None:
            trends = trends[trends.index >= pd.to_datetime(start)]
    except Exception:
        pass

    # build indicators (safe)
    econ_fx = safe_scale(-fx) if fx is not None else None   # negative because weaker currency => worse
    econ_gdp = safe_scale(gdp) if gdp is not None else None
    legitimacy = safe_scale(-trends) if trends is not None else None   # more protest interest => lower legitimacy
    stability = safe_scale(gdp) if gdp is not None else None

    # combine into monthly frame
    pieces = []
    if econ_fx is not None:
        pieces.append(econ_fx.rename("econ_fx"))
    if econ_gdp is not None:
        pieces.append(econ_gdp.rename("econ_gdp"))
    if legitimacy is not None:
        pieces.append(legitimacy.rename("legitimacy"))
    if stability is not None:
        pieces.append(stability.rename("stability"))

    if pieces:
        df = pd.concat(pieces, axis=1)
        # economic composite: mean of available econ submetrics
        econ_cols = [c for c in df.columns if c.startswith("econ")]
        if econ_cols:
            df["economic"] = df[econ_cols].mean(axis=1)
        else:
            df["economic"] = np.nan
        # if legitimacy/stability already in df keep, otherwise add NaN
        if "legitimacy" not in df.columns:
            df["legitimacy"] = np.nan
        if "stability" not in df.columns:
            df["stability"] = np.nan
    else:
        # create small monthly index from start to today and fill NaN
        idx = pd.date_range(start=pd.to_datetime(start), end=pd.Timestamp.today(), freq="M")
        df = pd.DataFrame(index=idx, data={"economic": np.nan, "legitimacy": np.nan, "stability": np.nan})

    # fill simple timeline for plotting
    df = df.sort_index().resample("M").last().ffill().bfill()

    # composite: mean of available indicators, fallback to neutral if row empty
    df["composite"] = df[["economic", "legitimacy", "stability"]].mean(axis=1)
    df["composite_filled"] = df["composite"].fillna(NEUTRAL)

    results[country] = df

# if nothing fetched -> show message
if not results:
    st.error("No usable data fetched for chosen countries. Try again later.")
    st.stop()

# comparison chart
st.header("Composite comparison (filled)")
comp = pd.concat({k: v["composite_filled"] for k, v in results.items()}, axis=1)
comp.columns = list(results.keys())
st.line_chart(comp.fillna(method="ffill"))

# per-country panels
for country, df in results.items():
    st.subheader(country)
    st.dataframe(df.tail(12).round(3))
    st.line_chart(df[["economic", "legitimacy", "stability", "composite_filled"]].fillna(NEUTRAL))

st.caption("Prototype — to increase rigor add ACLED/GDELT or official economic series for inflation/corruption.")
