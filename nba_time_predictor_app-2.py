import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble        import RandomForestRegressor
from sklearn.linear_model    import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

st.set_page_config(
    page_title="NBA Time Remaining Predictor",
    page_icon="🏀",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Data loading & model training — cached so it only runs once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_data_and_train():
    import gdown

    file_id = "1K5JQOG3G_LnvASRGkXJ6jmEtMEBVuxEK"
    url     = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, "pbp.csv", quiet=True)

    pbp = pd.read_csv("pbp.csv")
    pbp["timeActual"] = pd.to_datetime(pbp["timeActual"])
    pbp = pbp.sort_values(["game_id", "timeActual"]).reset_index(drop=True)

    pbp["seconds_elapsed"] = (
        pbp.groupby("game_id")["timeActual"]
           .transform(lambda x: (x - x.iloc[0]).dt.total_seconds())
    )
    pbp["duration"] = pbp.groupby("game_id")["seconds_elapsed"].transform("max")
    pbp = pbp[pbp["duration"] <= 20000].copy()

    def clock_to_seconds(clock_str):
        try:
            s   = str(clock_str).strip()
            iso = re.match(r"PT(?:(\d+)M)?(?:([\d.]+)S)?", s)
            if iso:
                return float(iso.group(1) or 0) * 60 + float(iso.group(2) or 0)
            parts = s.split(":")
            return int(parts[0]) * 60 + float(parts[1])
        except Exception:
            return np.nan
    

    pbp["clock_seconds"] = pbp["clock"].apply(clock_to_seconds)

    pbp["scoreHome"]      = pd.to_numeric(pbp["scoreHome"], errors="coerce").fillna(0)
    pbp["scoreAway"]      = pd.to_numeric(pbp["scoreAway"], errors="coerce").fillna(0)
    pbp["scoring_margin"] = (pbp["scoreHome"] - pbp["scoreAway"]).abs()

    PERIOD_DUR = {**{q: 720 for q in range(1, 5)}, **{ot: 300 for ot in range(5, 20)}}
    pbp["period_dur"]         = pbp["period"].map(PERIOD_DUR).fillna(300)
    pbp["game_clock_elapsed"] = pbp.apply(
        lambda r: (r["period"] - 1) * 720 + r["period_dur"] - r["clock_seconds"]
                  if r["period"] <= 4
                  else 2880 + (r["period"] - 5) * 300 + r["period_dur"] - r["clock_seconds"],
        axis=1,
    )

    pbp["last_four_min"] = ((pbp["period"] == 4) & (pbp["clock_seconds"] <= 240)).astype(int)
    pbp["last_two_min"]  = ((pbp["period"] == 4) & (pbp["clock_seconds"] <= 120)).astype(int)
    pbp["is_close_game"] = (pbp["scoring_margin"] <= 5).astype(int)

    desc = pbp["description"].str.lower().fillna("")
    pbp["is_foul"]     = desc.str.contains("foul|violation").astype(int)
    pbp["is_timeout"]  = desc.str.contains("timeout").astype(int)
    pbp["is_turnover"] = desc.str.contains("turnover").astype(int)
    pbp["is_review"]   = desc.str.contains("review|challenge").astype(int)
    pbp["is_ft"]       = desc.str.contains("free throw").astype(int)

    for col in ["is_foul", "is_timeout", "is_turnover", "is_review", "is_ft"]:
        pbp[f"{col}_cum"] = pbp.groupby("game_id")[col].cumsum()

    pbp["foul_rate_20"] = (
        pbp.groupby("game_id")["is_foul"]
           .transform(lambda x: x.rolling(20, min_periods=1).sum())
    )

    for col in ["is_foul", "is_timeout", "is_turnover", "is_ft"]:
        pbp[f"{col}_last4"]     = np.where(pbp["last_four_min"], pbp[col], 0)
        pbp[f"{col}_last4_cum"] = pbp.groupby("game_id")[f"{col}_last4"].cumsum()

    pbp["event_idx"] = pbp.groupby("game_id").cumcount() + 1
    pbp["pace"]      = pbp["event_idx"] / (pbp["game_clock_elapsed"].clip(lower=1) / 60)
    pbp["real_vs_clock_ratio"] = pbp["seconds_elapsed"] / pbp["game_clock_elapsed"].clip(lower=1)

    FEATURES = [
        "clock_seconds", "game_clock_elapsed",
        "scoring_margin", "is_close_game", "last_four_min", "last_two_min",
        "foul_rate_20", "is_foul_cum", "is_foul_last4_cum",
        "is_timeout_last4_cum", "is_ft_last4_cum", "is_turnover_last4_cum",
        "is_review_cum", "teamId", "pace", "real_vs_clock_ratio",
    ]
    TARGET = "real_remaining"

    q4           = pbp[pbp["period"] == 4].copy()
    q4[TARGET]   = q4["duration"] - q4["seconds_elapsed"]
    q4           = q4[q4[TARGET] >= 0]
    
    # We add game_id to model_df so we can split by game
    model_df     = q4[["game_id"] + FEATURES + [TARGET]].dropna()

    margin_bucket = pd.cut(model_df["scoring_margin"], bins=[-1, 5, 15, 100], labels=[0, 1, 2])
    valid_idx     = margin_bucket.notna() & model_df[TARGET].notna()
    
    # Filter only valid rows
    model_df = model_df.loc[valid_idx]
    
    X = model_df[FEATURES]
    y = model_df[TARGET]
    game_ids_valid = model_df["game_id"]

    # Ensure entire games are put into train or test, avoiding data leakage 
    # and ensuring there are actually untouched games to select from.
    unique_games = game_ids_valid.unique()
    train_games, test_games = train_test_split(unique_games, test_size=0.20, random_state=42)

    # Filter data based on the game-level split
    train_mask = game_ids_valid.isin(train_games)
    X_train, y_train = X[train_mask], y[train_mask]

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    lr = Ridge(alpha=1.0)
    lr.fit(X_train_sc, y_train)

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Return train_games alongside the other items
    return pbp, rf, lr, scaler, FEATURES, set(train_games)


def predict(pbp, rf, lr, scaler, FEATURES, game_id, minutes_remaining, seconds_remaining, model):
    clock_secs = minutes_remaining * 60 + seconds_remaining
    game_rows  = pbp[pbp["game_id"] == game_id].sort_values("seconds_elapsed")

    if game_rows.empty:
        return None, f"Game ID '{game_id}' not found."

    q4_rows = game_rows[game_rows["period"] == 4].copy()
    if q4_rows.empty:
        return None, f"No Q4 data for game '{game_id}'."

    q4_rows["_diff"] = (q4_rows["clock_seconds"] - clock_secs).abs()
    closest          = q4_rows.loc[q4_rows["_diff"].idxmin()]

    row = {
        "clock_seconds":         clock_secs,
        "game_clock_elapsed":    3 * 720 + (720 - clock_secs),
        "scoring_margin":        closest["scoring_margin"],
        "is_close_game":         int(closest["scoring_margin"] <= 5),
        "last_four_min":         int(clock_secs <= 240),
        "last_two_min":          int(clock_secs <= 120),
        "foul_rate_20":          closest["foul_rate_20"],
        "is_foul_cum":           closest["is_foul_cum"],
        "is_foul_last4_cum":     closest["is_foul_last4_cum"],
        "is_timeout_last4_cum":  closest["is_timeout_last4_cum"],
        "is_ft_last4_cum":       closest["is_ft_last4_cum"],
        "is_turnover_last4_cum": closest["is_turnover_last4_cum"],
        "is_review_cum":         closest["is_review_cum"],
        "pace":                  closest["pace"],
        "real_vs_clock_ratio":   closest["real_vs_clock_ratio"],
        "teamId":                closest["teamId"],
    }

    X_input = pd.DataFrame([row])[FEATURES]

    if model == "rf":
        pred_remaining = float(rf.predict(X_input)[0])
    else:
        pred_remaining = float(lr.predict(scaler.transform(X_input))[0])

    pred_remaining   = max(0, pred_remaining)
    actual_remaining = max(0, game_rows["seconds_elapsed"].max() - closest["seconds_elapsed"])

    return {
        "scoring_margin":   int(closest["scoring_margin"]),
        "is_clutch":        closest["scoring_margin"] <= 5,
        "foul_last4":       int(closest["is_foul_last4_cum"]),
        "timeout_last4":    int(closest["is_timeout_last4_cum"]),
        "real_vs_clock":    float(closest["real_vs_clock_ratio"]),
        "pred_remaining":   pred_remaining,
        "actual_remaining": actual_remaining,
        "error_secs":       pred_remaining - actual_remaining,
    }, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🏀 NBA Q4 Time Remaining Predictor")
st.caption("Predict how much real time is left based on game-clock position and live context.")

with st.spinner("Loading data and training models — this takes ~30 seconds the first time…"):
    # Unpack the set of training games
    pbp, rf, lr, scaler, FEATURES, train_games = load_data_and_train()

st.success(f"Ready — {pbp['game_id'].nunique():,} games loaded.", icon="✅")

game_matchups = (
    pbp.groupby("game_id")["teamTricode"]
    .apply(lambda x: sorted(set(x.dropna().astype(str))))
    .apply(lambda teams: " vs ".join(teams) if len(teams) >= 2 else "Unknown matchup")
    .to_dict()
)

pbp["game_date"] = pd.to_datetime(pbp["timeActual"]).dt.date

game_dates = pbp.groupby("game_id")["game_date"].first().to_dict()

all_ids = pbp["game_id"].unique().tolist()
test_ids = [gid for gid in all_ids if gid not in train_games]

col_id, col_model = st.columns([1, 1])
h1, h2, h3, h4, h5 = st.columns([3, 1.5, 1, 1, 1])
h1.caption("Game")
h2.caption("Model")
h3.caption("Min")
h4.caption("Sec")
h5.caption("")
# --- COMPACT CONTROL BAR ---

col_game, col_model, col_min, col_sec, col_btn = st.columns([3, 1.5, 1, 1, 1])

with col_game:
    game_id = st.selectbox(
        "Game",
        options=test_ids,
        format_func=lambda gid: f"{game_matchups.get(gid, gid)} | {game_dates.get(gid, '')}",
        label_visibility="collapsed"
    )

with col_model:
    model = st.selectbox(
        "Model",
        options=["rf", "lr"],
        format_func=lambda x: "RF" if x == "rf" else "Ridge",
        label_visibility="collapsed"
    )

with col_min:
    minutes_remaining = st.number_input(
        "Min",
        min_value=0,
        max_value=12,
        value=4,
        step=1,
        label_visibility="collapsed"
    )

with col_sec:
    seconds_remaining = st.number_input(
        "Sec",
        min_value=0,
        max_value=59,
        value=0,
        step=1,
        label_visibility="collapsed"
    )

with col_btn:
    run = st.button("Predict")

if run:
    with st.spinner("Running prediction…"):
        result, err_msg = predict(
            pbp, rf, lr, scaler, FEATURES,
            game_id, minutes_remaining, seconds_remaining, model,
        )

    if err_msg:
        st.error(err_msg)
    else:
        st.divider()

        # --- COMPACT RESULTS UI ---

        def fmt(secs):
            m, s = divmod(int(secs), 60)
            return f"{m}:{s:02d}"

        top1, top2 = st.columns([1, 1])

        # LEFT: Game context
        with top1:
            st.markdown("### Game Context")

            c1, c2 = st.columns(2)
            c1.metric("Margin", f"{result['scoring_margin']} pts")
            c2.metric("Clutch", "Yes" if result["is_clutch"] else "No")

            c3, c4 = st.columns(2)
            c3.metric("Fouls (L4M)", result["foul_last4"])
            c4.metric("Timeouts (L4M)", result["timeout_last4"])

            st.metric("Real/Clock", f"{result['real_vs_clock']:.2f}×")

        # RIGHT: Prediction
        with top2:
            st.markdown("### Prediction")

            p1, p2 = st.columns(2)
            p1.metric("Predicted", fmt(result["pred_remaining"]))
            p2.metric("Actual", fmt(result["actual_remaining"]))

            err = result["error_secs"]
            st.metric("Error", f"{err:+.0f}s", delta=f"{err/60:+.1f} min", delta_color="inverse")

        # BOTTOM: slim progress bar
        clock_left = minutes_remaining * 60 + seconds_remaining
        st.progress((720 - clock_left) / 720)

        # Footer (tiny)
        st.caption(f"{model.upper()} · {game_matchups.get(game_id, game_id)}")