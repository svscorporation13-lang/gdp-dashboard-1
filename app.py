for c in sel:
    meta = COUNTRIES[c]

    fx  = fetch_fx(meta["fx"])
    gdp = fetch_gdp(meta["wb"])
    tr  = fetch_trends(meta["iso"])

    # --- odfiltrowanie do zakresu dat ---
    if fx is not None:
        fx = fx[fx.index >= pd.to_datetime(start)]

    if gdp is not None:
        gdp = gdp[gdp.index >= pd.to_datetime(start)]

    if tr is not None:
        tr = tr[tr.index >= pd.to_datetime(start)]

    # --- placeholder danych modelowych (dopóki nie dodasz prawdziwych źródeł) ---
    corruption = tr
    legitimacy = tr
    stability  = tr
    gov        = tr

    # --- bezpieczne skalowanie (chroni przed błędami) ---
    def safe(series, inverse=False):
        if series is None or len(series) <= 1:
            return None
        return scale(series, inverse=inverse)

    gov_s  = safe(gov)
    econ_s = safe(-fx)             # silniejsza waluta = lepiej
    corr_s = safe(-corruption)     # mniej korupcji = lepiej
    gdp_s  = safe(gdp)             # wzrost PKB = lepiej
    infl_s = safe(-fx, inverse=True)  # placeholder inflacji

    # --- zbuduj dataframe ---
    df = pd.DataFrame({
        "economic": econ_s,
        "legitimacy": legitimacy,
        "stability": stability,
    })

    df["composite"] = df.mean(axis=1)

    results[c] = df
