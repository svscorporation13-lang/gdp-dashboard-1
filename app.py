import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from pytrends.request import TrendReq
import requests

st.set_page_config(page_title="Authoritarian Monitor — Lite", layout="wide")

COUNTRIES = {
    "Russia": {"wb":"RUS", "fx":"RUBUSD=X", "iso":"RU"},
    "China": {"wb":"CHN", "fx":"CNYUSD=X", "iso":"CN"},
    "Iran": {"wb":"IRN", "fx":"IRRUSD=X", "iso":"IR"},
    "Turkey": {"wb":"TUR", "fx":"TRYUSD=X", "iso":"TR"},
}

def fetch_fx(pair):
    try:
        df = yf.download(pair, period="3y", interval="1d")
        if df.empty: return None
        df = df["Close"].resample("M").last()
        return df
    except:
        return None

def fetch_gdp(wb):
    try:
        url = f"https://api.worldbank.org/v2/country/{wb}/indicator/NY.GDP.PCAP.CD?format=json"
        r = requests.get(url).json()
        data = r[1]
        df = pd.DataFrame(data)[["date","value"]].dropna()
        df["date"] = pd.to_datetime(df["date"]+"-01-01")
        df = df.set_index("date")["value"].resample("A").last()
        return df
    except:
        return None

def fetch_trends(iso):
    try:
        py = TrendReq()
        py.build_payload(["protest"], timeframe="today 5-y", geo=iso)
        df = py.interest_over_time()
        if df.empty: return None
        return df["protest"].resample("M").mean()
    except:
        return None

def scale(series, inverse=False):
    low = series.min()
    high = series.max()

    # NEW: prevent division by zero when high == low
    if high == low:
        return pd.Series([0.5] * len(series), index=series.index)

    scaled = (series - low) / (high - low)
    return 1 - scaled if inverse else scaled


st.title("Authoritarian Regime Effectiveness — Lite Dashboard")

sel = st.sidebar.multiselect("Countries", list(COUNTRIES.keys()), default=["Russia","China"])

start = dt.date.today() - dt.timedelta(days=365*3)

results = {}

for c in sel:
    meta = COUNTRIES[c]

    fx = fetch_fx(meta["fx"])
    gdp = fetch_gdp(meta["wb"])
    tr = fetch_trends(meta["iso"])

    if fx is not None: fx = fx[fx.index >= pd.to_datetime(start)]
    if gdp is not None: gdp = gdp[gdp.index >= pd.to_datetime(start)]
    if tr is not None: tr = tr[tr.index >= pd.to_datetime(start)]

# SAFE WRAPPER – prevents errors when a dataset has only 1 value
def safe_scale(series, inverse=False):
    if series is None or len(series) <= 1:
        return None
    return scale(series, inverse=inverse)

# Apply scaling safely
gov = safe_scale(gov)
econ = safe_scale(-fx, inverse=False)
corr = safe_scale(-corruption, inverse=False)
gdp_s = safe_scale(gdp)
infl = safe_scale(-inflation, inverse=True)

values = [
    gov.iloc[-1] if gov is not None else 0.5,
    econ.iloc[-1] if econ is not None else 0.5,
    corr.iloc[-1] if corr is not None else 0.5,
    gdp_s.iloc[-1] if gdp_s is not None else 0.5,
    infl.iloc[-1] if infl is not None else 0.5,
]


    df = pd.concat([econ.rename("economic"), legit.rename("legitimacy"), stability.rename("stability")], axis=1)

    df["composite"] = df.mean(axis=1)

    results[c] = df

st.header("Composite Score Comparison")

comp = pd.concat({k: v["composite"] for k,v in results.items()}, axis=1)
st.line_chart(comp)

for c, df in results.items():
    st.subheader(c)
    st.dataframe(df.tail(12))
