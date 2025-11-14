# app.py
# Authoritarian Monitor — Lite (Rebuilt, robust)
# Save as app.py in your repo and deploy to Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import requests

st.set_page_config(page_title="Authoritarian Monitor — Lite", layout="wide")

# ========== CONFIG ==========
COUNTRIES = {
    "Russia": {"wb": "RUS", "fx": "RUBUSD=X", "iso": "RU"},
    "China": {"wb": "CHN", "fx": "CNYUSD=X", "iso": "CN"},
    "Iran": {"wb": "IRN", "fx": "IRRUSD=X", "iso": "IR"},
    "Turkey": {"wb": "TUR", "fx": "TRYUSD=X", "iso": "TR"},
}
NEUTRAL = 0.5  # default fallback for missing indicators

# ========== DEPENDENCY-SAFE IMPORTS ==========
# yfinance and pytrends may not be present in environment at import time on Cloud;
# import lazily inside functions and handle exceptions gracefully.

# ========== CACHING FETCHERS ==========
@st.cache_data(ttl=60 * 60 * 6)  # cache 6 hours
def fetch_fx(pair: str, period: str = "3y"):
    try:
        import yfinance as yf
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
    try:
        from pytrends.request import TrendReq
        py = TrendReq(hl="en-US", tz=0)
        kw = ["protest"]
        # pytrends expects geo like 'RU' or '' (global)
        py.build_payload(kw, timeframe=timeframe, geo=iso if iso else "")
        df = py.interest_over_time()
        if df is None or df.empty:
            return None
        s = df["protest"].resample("M").mean()
        return s
    except Exception:
        return None

# ========== NORMALIZATION & HELPERS ==========
def safe_minmax_scale(series: pd.Series, inverse: bool = False):
    """Map series to 0..1 safely. If insufficient data -> return None."""
    if series is None:
        return None
    s = pd.Series(series).dropna().astype(float)
    if s.empty or s.shape[0] < 2:
        return None
    low, high = s.min(), s.max()
    if high == low:
        return pd.Series(np.repeat(0.5, len(s)), index=s.index)
    scaled = (s - low) / (high - low)
    return (1 - scaled) if inverse else scaled

def last_or_neutral(s):
    """Return last value from series or NEUTRAL if None."""
    if s is None:
        return NEUTRAL
    try:
        v = float(s.iloc[-1])
        if np.isnan(v):
            return NEUTRAL
        return float(v)
    except Exception:
        return NEUTRAL

# ========== CORE: build indicators per country ==========
def build_indicators_for_country(meta: dict, start_date: dt.date):
    """Return DataFrame (monthly) with columns: economic, legitimacy, stability, composite"""
    # 1) Fetch raw series
    fx = fetch_fx(meta.get("fx"))
    gdp = fetch_gdp_per_capita(meta.get("wb"))
    trends = fetch_trends_protest(meta.get("iso"))

    # Restrict to start_date
    try:
        if fx is not None:
            fx = fx[fx.index >= pd.to_datetime(start_date)]
        if gdp is not None:
            gdp = gdp[gdp.index >= pd.to_datetime(start_date)]
        if trends is not None:
            trends = trends[trends.index >= pd.to_datetime(start_date)]
    except Exception:
        pass

    # 2) Normalize to 0..1 (higher = better)
    # economic: combine FX strength and GDP per capita (if available)
    econ_fx = safe_minmax_scale(-fx) if fx is not None else None  # negative because weaker FX -> worse (we invert)
    econ_gdp = safe_minmax_scale(gdp) if gdp is not None else None

    # Merge monthly frame
    frames = []
    if econ_fx is not None:
        frames.append(econ_fx.rename("econ_fx"))
    if econ_gdp is not None:
        frames.append(econ_gdp.rename("econ_gdp"))
    if frames:
        df = pd.concat(frames, axis=1)
        # economic score = mean of available economic submetrics
        df["economic"] = df.mean(axis=1)
    else:
        df = pd.DataFrame(index=pd.date_range(start=start_date, periods=1, freq="M"))
        df["economic"] = np.nan

    # legitimacy: inverse of protest interest (more searches/events -> lower legitimacy)
    legit = safe_minmax_scale(-trends) if trends is not None else None
    if legit is not None:
        # align
        df = df.join(legit.rename("legitimacy"), how="outer")
    else:
        df["legitimacy"] = np.nan

    # stability: proxy from GDP per capita (higher = more stable)
    stab = safe_minmax_scale(gdp) if gdp is not None else None
    if stab is not None:
        df = df.join(stab.rename("stability"), how="outer")
    else:
        df["stability"] = np.nan

    # Fill timeline and forward-fill reasonable values
    df = df.sort_index()
    df = df.resample("M").last().ffill().bfill()

    # If entire column is NaN -> keep as NaN so composite logic uses NEUTRAL fallback later
    # Composite: mean of available indicators (economic, legitimacy, stability)
    def composite_row(row):
        vals = []
        for col in ["economic", "legitimacy", "stability"]:
            v = row.get(col)
            if pd.isna(v):
                continue
            vals.append(v)
        if not vals:
            return np.nan
        return float(np.mean(vals))

    df["composite"] = df.apply(composite_row, axis=1)

    return df[["economic", "legitimacy", "stability", "composite"]]

# ========== UI & orchestration ==========
st.title("Authoritarian Regime Effectiveness — Lite Dashboard")
st.markdown("Porównanie: prosty, odporny model. Wartości 0..1 (1 = lepiej dla reżimu).")

# Sidebar
selected = st.sidebar.multiselect("Wybierz kraje", list(COUNTRIES.keys()), default=["Russia", "China"])
start = st.sidebar.date_input("Start date", value=dt.date.today() - dt.timedelta(days=365 * 3))
fetch = st.sidebar.button("Fetch / Update now")

# Data container
results = {}

if fetch:
    st.sidebar.info("Pobieram dane — proszę czekać (może zająć kilkanaście sekund).")
    for c in selected:
        meta = COUNTRIES[c]
        try:
            dfc = build_indicators_for_country(meta, start)
            if dfc is None or dfc.empty:
                st.sidebar.warning(f"Brak danych dla {c}")
                continue
            # ensure composite uses neutral default if NaN
            dfc["composite_filled"] = dfc["composite"].fillna(NEUTRAL)
            results[c] = dfc
            # save CSV to cache for manual download
            csv = dfc.to_csv(index=True)
            st.sidebar.download_button(f"Pobierz CSV: {c}", data=csv, file_name=f"{c}_indicators.csv")
        except Exception as e:
            st.sidebar.error(f"Błąd dla {c}: {e}")

# If not fetched this session, attempt to show nothing or instruct to click fetch
if not results:
    st.warning("Brak aktualnych danych. Kliknij 'Fetch / Update now' w panelu bocznym, aby pobrać dane.")
    st.stop()

# Show comparison chart
st.header("Composite (porównanie)")
comp_df = pd.concat({k: v["composite_filled"] for k, v in results.items()}, axis=1)
comp_df.columns = list(results.keys())
st.line_chart(comp_df.fillna(method="ffill"))

# Show per-country panels
for country, dfc in results.items():
    st.subheader(country)
    st.dataframe(dfc.tail(12).round(3))
    st.line_chart(dfc[["economic", "legitimacy", "stability", "composite_filled"]].fillna(NEUTRAL))

st.caption("Uwaga: to narzędzie prototypowe. Aby zwiększyć precyzję wskaźników politycznych (represje, elite defections) zintegrować ACLED/GDELT/oxford sanctions.")
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

    # --- brakujące dane modelowe ---
    # (prosty placeholder, dopóki nie dodasz prawdziwych źródeł)
    corruption = tr  
    legitimacy = tr  
    stability = tr  
    gov = tr  

    # --- bezpieczne skalowanie ---
    def safe(series, inverse=False):
        if series is None or len(series) <= 1:
            return None
        return scale(series, inverse=inverse)

    gov_s  = safe(gov)
    econ_s = safe(-fx)            # silniejsza waluta = lepiej
    corr_s = safe(-corruption)    # mniej korupcji = lepiej
    gdp_s  = safe(gdp)
    infl_s = safe(-fx, inverse=True)  # uproszczone

    # --- zbuduj dataframe dla kraju ---
    df = pd.DataFrame({
        "economic": econ_s,
        "legitimacy": legitimacy,
        "stability": stability
    })

    df["composite"] = df.mean(axis=1)

    results[c] = df
