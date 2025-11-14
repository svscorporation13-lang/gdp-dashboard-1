import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from pytrends.request import TrendReq
import requests

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

st.set_page_config(page_title="Authoritarian Monitor — Lite", layout="wide")

COUNTRIES = {
    "Russia": {"wb": "RUS", "fx": "RUBUSD=X", "iso": "RU"},
    "China": {"wb": "CHN", "fx": "CNYUSD=X", "iso": "CN"},
    "Iran": {"wb": "IRN", "fx": "IRRUSD=X", "iso": "IR"},
    "Turkey": {"wb": "TUR", "fx": "TRYUSD=X", "iso": "TR"},
}

# ---------------------------------------------------------
# DATA FETCHERS
# ---------------------------------------------------------

def fetch_fx(pair):
    try:
        df = yf.download(pair, period="3y", interval="1d")
        if df.empty:
            return None
        return df["Close"].resample("M").last()
    except:
        return None

def fetch_gdp(wb):
    try:
        url = f"https://api.worldbank.org/v2/country/{wb}/indicator/NY.GDP.PCAP.CD?format=json"
        r = requests.get(url).json()
        data = r[1]
        df = pd.DataFrame(data)[["date", "value"]].dropna()
        df["date"] = pd.to_datetime(df["date"] + "-01-01")
        return df.set_index("date")["value"].resample("A").last()
    except:
        return None

def fetch_trends(iso):
    try:
        py = TrendReq()
        py.build_payload(["protest"], timeframe="today 5-y", geo=iso)
        df = py.interest_over_time()
        if df.empty:
            return None
        return df["protest"].resample("M").mean()
    except:
        return None

# ---------------------------------------------------------
# SCALING
# ---------------------------------------------------------

def scale(series, inverse=False):
    # ensure valid numeric series
    if series is None:
        return None
    if not isinstance(series, pd.Series):
        return None
    if series.empty:
        return None
    if len(series.dropna()) == 0:
        return None

    s = series.astype(float)

    low = float(s.min())
    high = float(s.max())

    # avoid division by zero
    if abs(high - low) < 1e-9:
        return pd.Series([0.5] * len(s), index=s.index)

    scaled = (s - low) / (high - low)
    return 1 - scaled if inverse else scaled


def safe_scale(series, inverse=False):
    try:
        result = scale(series, inverse=inverse)
        return result
    except Exception:
        return None

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.title("Authoritarian Regime Effectiveness — Lite Dashboard")
st.write("Prosty, odporny model. Skala: 0..1 (1 = silniejszy reżim).")

sel = st.sidebar.multiselect("Wybierz kraje", list(COUNTRIES.keys()), default=["Russia", "China"])
start = st.sidebar.date_input("Start date", dt.date.today() - dt.timedelta(days=365 * 3))

# ---------------------------------------------------------
# PROCESSING
# ---------------------------------------------------------

results = {}

for c in sel:
    meta = COUNTRIES[c]

    fx = fetch_fx(meta["fx"])
    gdp = fetch_gdp(meta["wb"])
    tr = fetch_trends(meta["iso"])

    if fx is not None:
        fx = fx[fx.index >= pd.to_datetime(start)]
    if gdp is not None:
        gdp = gdp[gdp.index >= pd.to_datetime(start)]
    if tr is not None:
        tr = tr[tr.index >= pd.to_datetime(start)]

    # Placeholder model variables (until real ones are added)
    corruption = tr
    legitimacy = tr
    stability = tr
    gov = tr

    gov_s = safe(gov)
    econ_s = safe(-fx)  # lower FX = stronger currency
    corr_s = safe(-corruption)
    gdp_s = safe(gdp)
    infl_s = safe(None)  # placeholder until inflation added

    # Latest values (for radar/panel)
    values = [
        gov_s.iloc[-1] if gov_s is not None else 0.5,
        econ_s.iloc[-1] if econ_s is not None else 0.5,
        corr_s.iloc[-1] if corr_s is not None else 0.5,
        gdp_s.iloc[-1] if gdp_s is not None else 0.5,
        infl_s.iloc[-1] if infl_s is not None else 0.5,
    ]

    df = pd.concat(
        [
            econ_s.rename("economic") if econ_s is not None else None,
            legitimacy.rename("legitimacy") if legitimacy is not None else None,
            stability.rename("stability") if stability is not None else None,
        ],
        axis=1,
    )

    df["composite"] = df.mean(axis=1)
    results[c] = df

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

if len(results) == 0:
    st.warning("Brak danych. Wybierz kraj i kliknij 'Update'.")
else:
    st.header("Composite Score Comparison")
    comp = pd.concat({k: v["composite"] for k, v in results.items()}, axis=1)
    st.line_chart(comp)

    for c, df in results.items():
        st.subheader(c)
        st.dataframe(df.tail(12))

