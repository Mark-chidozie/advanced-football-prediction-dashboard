# -*- coding: utf-8 -*-
# Advanced Football Predictor – multi-page Streamlit app
# Now with a dedicated "Bet Insights" page (sidebar nav) driven by Poisson xG.

from __future__ import annotations
import os, time, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    HAVE_XGB = False

# ============================== CONFIG ==============================
st.set_page_config(page_title="Advanced Football Predictor", page_icon="⚽", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FD_TOKEN = st.secrets.get("FOOTBALL_DATA_TOKEN") or os.environ.get("FOOTBALL_DATA_TOKEN")
AF_KEY   = st.secrets.get("API_FOOTBALL_KEY")    or os.environ.get("API_FOOTBALL_KEY")

FD_ROOT  = "https://api.football-data.org/v4"
FD_HDRS  = {"X-Auth-Token": FD_TOKEN} if FD_TOKEN else {}

AF_ROOT  = "https://v3.football.api-sports.io"
AF_HDRS  = {"x-rapidapi-key": AF_KEY, "x-rapidapi-host": "v3.football.api-sports.io"} if AF_KEY else {}

# competitions for team search (you already expanded this)
LEAGUES = ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "WC", "CL", "EL", "EC",
           "WCQ", "AAL", "BJL", "DFB", "ABL", "BSA", "ELC", "BSA", "B1", "B2"]

WIN, DRAW, LOSS = "#22c55e", "#fbbf24", "#ef4444"
LABEL_MAP = {0: "Home win", 1: "Draw", 2: "Away win"}

LOGREG_FEATURES = [
    "home_gf_pg", "home_ga_pg", "home_gd_pg",
    "away_gf_pg", "away_ga_pg", "away_gd_pg"
]

# heuristic factor for corners from total xG
CORNERS_PER_XG = 3.0

# ------------------------------ CSS ------------------------------
st.markdown(
    """
<style>
.stApp{
  background:#ffffff;
  color:#0f172a;
}
.block-container{
  max-width:1400px;
  padding-top:.6rem;
}
.section-title{
  font-weight:800;
  font-size:1.1rem;
  margin:.6rem 0 .2rem;
}
.subtle{opacity:.8}
.divider{height:1px;background:#e5e7eb;margin:1rem 0;}
.crest{
  width:60px;
  height:60px;
  object-fit:contain;
  vertical-align:middle;
}
.teamname{
  font-weight:900;
  font-size:22px;
  display:flex;
  gap:10px;
  align-items:center;
}
.bigxg{
  font-weight:900;
  font-size:38px;
}
.card{
  background:#f8fafc;
  border:1px solid #e5e7eb;
  border-radius:14px;
  padding:14px;
}
.player-card{
  background:#0b1220;
  color:#e6edf3;
  border-radius:16px;
  padding:12px;
  border:1px solid #141b2e;
}
.player-badge{
  background:#10b981;
  border-radius:999px;
  padding:0 8px;
  margin-left:8px;
  font-weight:700;
}
.small-note{
  font-size:0.8rem;
  opacity:0.8;
}

/* BET INSIGHTS panel styling (green digital board look) */
.bet-panel{
  background:radial-gradient(circle at top,#064e3b 0%,#020617 65%);
  border-radius:18px;
  padding:16px 18px 14px 18px;
  color:#e5e7eb;
  border:1px solid rgba(34,197,94,.6);
  box-shadow:0 16px 40px rgba(15,23,42,.8);
}
.bet-header{
  display:flex;
  justify-content:space-between;
  align-items:center;
  font-size:0.9rem;
  letter-spacing:.06em;
  text-transform:uppercase;
  margin-bottom:6px;
}
.bet-title{
  font-weight:800;
}
.bet-engine{
  font-size:0.65rem;
  padding:2px 8px;
  border-radius:999px;
  border:1px solid rgba(148,163,184,.8);
  text-transform:uppercase;
}
.bet-subtitle{
  font-size:0.75rem;
  opacity:.85;
  margin-bottom:6px;
}
.bet-section-title{
  font-size:0.78rem;
  text-transform:uppercase;
  letter-spacing:.05em;
  margin:8px 0 4px;
}
.bet-table{
  width:100%;
  border-collapse:collapse;
  font-size:0.78rem;
}
.bet-table th,
.bet-table td{
  padding:4px 6px;
  border-bottom:1px solid rgba(148,163,184,.18);
}
.bet-table th{
  font-weight:600;
  text-align:center;
  background:rgba(15,23,42,.85);
}
.bet-table td.label{
  text-align:left;
}
.bet-table td.val{
  text-align:center;
  font-weight:600;
  color:#22c55e;
}
.bet-footnote{
  margin-top:6px;
  font-size:0.7rem;
  opacity:.8;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================== BASIC HELPERS ==============================
def fd_get(url: str, params: Optional[Dict]=None) -> Dict:
    r = requests.get(url, headers=FD_HDRS, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def af_get(path: str, params: Optional[Dict]=None) -> Dict:
    r = requests.get(f"{AF_ROOT}{path}", headers=AF_HDRS, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=6*3600, show_spinner=True)
def build_team_index() -> pd.DataFrame:
    rows: List[dict] = []
    if not FD_HDRS:
        return pd.DataFrame()
    for code in LEAGUES:
        try:
            d = fd_get(f"{FD_ROOT}/competitions/{code}/teams")
            for t in d.get("teams", []):
                rows.append(
                    {
                        "id": t["id"],
                        "name": t["name"],
                        "crest": t.get("crest"),
                        "league": code,
                    }
                )
            time.sleep(0.12)
        except Exception:
            pass
    return (
        pd.DataFrame(rows)
        .drop_duplicates("id")
        .sort_values("name")
        .reset_index(drop=True)
    )

@st.cache_data(ttl=1800)
def team_last_matches(team_id: int, n: int = 20) -> pd.DataFrame:
    try:
        d = fd_get(f"{FD_ROOT}/teams/{team_id}/matches",
                   params={"status": "FINISHED", "limit": n})
        items = d.get("matches", [])
    except Exception:
        items = []
    rows: List[dict] = []
    for m in items:
        ft = (m.get("score") or {}).get("fullTime") or {"home": None, "away": None}
        if ft["home"] is None or ft["away"] is None:
            continue
        rows.append(
            {
                "date": m["utcDate"],
                "homeId": m["homeTeam"]["id"],
                "awayId": m["awayTeam"]["id"],
                "homeTeam": m["homeTeam"]["name"],
                "awayTeam": m["awayTeam"]["name"],
                "hg": int(ft["home"]),
                "ag": int(ft["away"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("date")
        .reset_index(drop=True)
    )

@st.cache_data(ttl=900)
def build_training_dataset(home_id: int, away_id: int) -> pd.DataFrame:
    """Rolling features from last 7 matches for both teams."""
    dfH = team_last_matches(home_id, 20)
    dfA = team_last_matches(away_id, 20)
    base = (
        pd.concat([dfH, dfA], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    if base.empty:
        return pd.DataFrame()

    feats: List[dict] = []
    for i, r in base.iterrows():

        def roll(tid: int) -> Tuple[float, float, float]:
            past = base.iloc[:i]
            sel = past[(past["homeId"] == tid) | (past["awayId"] == tid)].tail(7)
            if sel.empty:
                return (1.4, 1.4, 0.0)
            gf = ga = 0
            for _, x in sel.iterrows():
                if x["homeId"] == tid:
                    gf += x["hg"]
                    ga += x["ag"]
                else:
                    gf += x["ag"]
                    ga += x["hg"]
            n = len(sel)
            return (gf / n, ga / n, (gf - ga) / n)

        hgf, hga, hgd = roll(r["homeId"])
        agf, aga, agd = roll(r["awayId"])

        feats.append(
            {
                "date": r["date"],
                "homeTeam": r["homeTeam"],
                "awayTeam": r["awayTeam"],
                "home_gf_pg": hgf,
                "home_ga_pg": hga,
                "home_gd_pg": hgd,
                "away_gf_pg": agf,
                "away_ga_pg": aga,
                "away_gd_pg": agd,
                "hg": r["hg"],
                "ag": r["ag"],
            }
        )

    return pd.DataFrame(feats)

# ============================== POISSON MODEL ==============================
def poisson_probs(mu_h: float, mu_a: float, cap: int = 8) -> Tuple[float, float, float]:
    """Independent Poisson for (home, away) -> (p_home_win, p_draw, p_away_win)."""
    home_p = [math.exp(-mu_h) * mu_h**k / math.factorial(k) for k in range(cap + 1)]
    away_p = [math.exp(-mu_a) * mu_a**k / math.factorial(k) for k in range(cap + 1)]
    home_p[-1] += max(0.0, 1.0 - sum(home_p))
    away_p[-1] += max(0.0, 1.0 - sum(away_p))

    p_home = p_draw = p_away = 0.0
    for i, pi in enumerate(home_p):
        for j, pj in enumerate(away_p):
            p = pi * pj
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    s = p_home + p_draw + p_away
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / s, p_draw / s, p_away / s

def mu_from_fixed_math(row: pd.Series, home_adv: float) -> Tuple[float, float]:
    """
    Poisson xG maths (keep this exactly as before):
      mu_home ~ 0.6*home_scored + 0.4*away_conceded + home_adv
      mu_away ~ 0.6*away_scored + 0.4*home_conceded
    """
    mu_h = max(0.1, 0.6 * row.home_gf_pg + 0.4 * row.away_ga_pg + home_adv)
    mu_a = max(0.1, 0.6 * row.away_gf_pg + 0.4 * row.home_ga_pg)
    return mu_h, mu_a

def poisson_pmf_array(mu: float, cap: int = 12) -> np.ndarray:
    """Single-team or total-goals distribution up to cap (last bucket soaks tail)."""
    arr = [math.exp(-mu) * mu**k / math.factorial(k) for k in range(cap + 1)]
    arr[-1] += max(0.0, 1.0 - sum(arr))
    return np.asarray(arr, dtype=float)

def poisson_joint(mu_h: float, mu_a: float, cap: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return home_dist, away_dist, and full joint matrix."""
    home = poisson_pmf_array(mu_h, cap)
    away = poisson_pmf_array(mu_a, cap)
    joint = np.outer(home, away)
    return home, away, joint

def evaluate_poisson_on_df(df: pd.DataFrame, home_adv: float) -> dict:
    preds: List[List[float]] = []
    y: List[int] = []
    for _, r in df.iterrows():
        mu_h, mu_a = mu_from_fixed_math(r, home_adv)
        ph, pd_, pa = poisson_probs(mu_h, mu_a)
        preds.append([ph, pd_, pa])
        y.append(0 if r.hg > r.ag else 1 if r.hg == r.ag else 2)
    preds_arr = np.asarray(preds)
    y_arr = np.asarray(y)
    yhat = preds_arr.argmax(1)

    acc = accuracy_score(y_arr, yhat) * 100
    f1 = f1_score(y_arr, yhat, average="macro") * 100
    try:
        ll = log_loss(y_arr, preds_arr) * 100
    except ValueError:
        ll = float("nan")
    try:
        auc_macro = roc_auc_score(y_arr, preds_arr, multi_class="ovr", average="macro") * 100
    except ValueError:
        auc_macro = float("nan")

    return {
        "model_name": "Poisson (fixed maths)",
        "kind": "classifier",
        "acc": acc,
        "f1": f1,
        "logloss": ll,
        "auc": auc_macro,
        "rows": int(len(df)),
        "y_true": y_arr,
        "y_pred": yhat,
        "proba": preds_arr,
        "feature_importance": None,
    }

# ============================== ML MODELS ==============================
def time_split(df: pd.DataFrame):
    d2 = df.copy()
    d2["_ord"] = pd.to_datetime(d2["date"])
    d2 = d2.sort_values("_ord").reset_index(drop=True)
    if len(d2) < 6:
        return d2.index, d2.index
    cut = max(1, int(0.8 * len(d2)))
    tr_idx, te_idx = d2.index[:cut], d2.index[cut:]
    if len(te_idx) < 3:
        tr_idx, te_idx = d2.index, d2.index
    return tr_idx, te_idx

def train_classifier(df: pd.DataFrame, kind: str, name: str) -> dict:
    X = df[LOGREG_FEATURES].copy()
    y = df.apply(lambda r: 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2), axis=1)
    tr_idx, te_idx = time_split(df)
    Xtr, ytr = X.loc[tr_idx], y.loc[tr_idx]
    Xte, yte = X.loc[te_idx], y.loc[te_idx]

    feature_importance = None

    if kind == "logreg":
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        clf = LogisticRegression(max_iter=500, multi_class="multinomial")
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)
        coef = np.abs(clf.coef_).mean(axis=0)
        feature_importance = dict(zip(LOGREG_FEATURES, coef))

    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        feature_importance = dict(zip(LOGREG_FEATURES, clf.feature_importances_))

    elif kind == "xgb":
        if not HAVE_XGB:
            raise RuntimeError("XGBoost not installed.")
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr.values, ytr.values)
        proba = clf.predict_proba(Xte.values)
        try:
            fi = clf.feature_importances_
            feature_importance = dict(zip(LOGREG_FEATURES, fi))
        except Exception:
            feature_importance = None

    elif kind == "svm":
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        clf = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")

    proba_arr = np.asarray(proba)
    yte_arr = np.asarray(yte)
    yhat = proba_arr.argmax(1)

    acc = accuracy_score(yte_arr, yhat) * 100
    f1  = f1_score(yte_arr, yhat, average="macro") * 100
    try:
        ll = log_loss(yte_arr, proba_arr) * 100
    except ValueError:
        ll = float("nan")
    try:
        auc_macro = roc_auc_score(yte_arr, proba_arr, multi_class="ovr", average="macro") * 100
    except ValueError:
        auc_macro = float("nan")

    return {
        "model_name": name,
        "kind": "classifier",
        "acc": acc,
        "f1": f1,
        "logloss": ll,
        "auc": auc_macro,
        "rows": int(len(df)),
        "y_true": yte_arr,
        "y_pred": yhat,
        "proba": proba_arr,
        "feature_importance": feature_importance,
    }

def train_regressor(df: pd.DataFrame, kind: str, name: str) -> dict:
    X = df[LOGREG_FEATURES].copy()
    y_home = df["hg"].astype(float)
    y_away = df["ag"].astype(float)
    tr_idx, te_idx = time_split(df)
    Xtr, Xte = X.loc[tr_idx], X.loc[te_idx]
    yh_tr, yh_te = y_home.loc[tr_idx], y_home.loc[te_idx]
    ya_tr, ya_te = y_away.loc[tr_idx], y_away.loc[te_idx]

    if kind == "rf":
        reg_h = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        reg_a = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=43,
            n_jobs=-1,
        )
        reg_h.fit(Xtr, yh_tr)
        reg_a.fit(Xtr, ya_tr)
        yh_pred = reg_h.predict(Xte)
        ya_pred = reg_a.predict(Xte)
        fi = reg_h.feature_importances_ + reg_a.feature_importances_
        feature_importance = dict(zip(LOGREG_FEATURES, fi))

    elif kind == "xgb":
        if not HAVE_XGB:
            raise RuntimeError("XGBoost not installed.")
        reg_h = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        reg_a = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=43,
        )
        reg_h.fit(Xtr.values, yh_tr.values)
        reg_a.fit(Xtr.values, ya_tr.values)
        yh_pred = reg_h.predict(Xte.values)
        ya_pred = reg_a.predict(Xte.values)
        try:
            fi = reg_h.feature_importances_ + reg_a.feature_importances_
            feature_importance = dict(zip(LOGREG_FEATURES, fi))
        except Exception:
            feature_importance = None

    elif kind == "svr":
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        reg_h = SVR(kernel="rbf", C=2.0, gamma="scale")
        reg_a = SVR(kernel="rbf", C=2.0, gamma="scale")
        reg_h.fit(Xtr_s, yh_tr)
        reg_a.fit(Xtr_s, ya_tr)
        yh_pred = reg_h.predict(Xte_s)
        ya_pred = reg_a.predict(Xte_s)
        feature_importance = None
    else:
        raise ValueError(f"Unknown regressor kind: {kind}")

    mae_home = mean_absolute_error(yh_te, yh_pred)
    mae_away = mean_absolute_error(ya_te, ya_pred)
    rmse_home = mean_squared_error(yh_te, yh_pred, squared=False)
    rmse_away = mean_squared_error(ya_te, ya_pred, squared=False)

    yh_round = np.rint(yh_pred)
    ya_round = np.rint(ya_pred)
    y_true_cls = np.where(
        yh_te.values > ya_te.values,
        0,
        np.where(yh_te.values == ya_te.values, 1, 2),
    )
    y_hat_cls = np.where(
        yh_round > ya_round,
        0,
        np.where(yh_round == ya_round, 1, 2),
    )
    acc = accuracy_score(y_true_cls, y_hat_cls) * 100
    f1 = f1_score(y_true_cls, y_hat_cls, average="macro") * 100

    return {
        "model_name": name,
        "kind": "regressor",
        "acc": acc,
        "f1": f1,
        "logloss": float("nan"),
        "auc": float("nan"),
        "rows": int(len(df)),
        "y_true": y_true_cls,
        "y_pred": y_hat_cls,
        "proba": None,
        "feature_importance": feature_importance,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "rmse_home": rmse_home,
        "rmse_away": rmse_away,
    }

# ============================== VISUAL HELPERS ==============================
def slim_strength_bar(pH: float, pA: float) -> go.Figure:
    tot = max(1e-9, pH + pA)
    h = 100 * pH / tot
    a = 100 - h
    fig = go.Figure()
    fig.add_bar(x=[h], y=[""], orientation="h", marker_color=WIN,
                text=[f"{h:.0f}%"], textposition="inside")
    fig.add_bar(x=[a], y=[""], orientation="h", marker_color=LOSS,
                text=[f"{a:.0f}%"], textposition="inside")
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        height=54,
        showlegend=False,
    )
    return fig

def gf_ga_gd_chart(home_vals, away_vals, labels=("GF", "GA", "GD")) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=labels, y=home_vals, name="Home", marker_color=WIN)
    fig.add_bar(x=labels, y=away_vals, name="Away", marker_color=LOSS)
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=260,
    )
    return fig

def plot_feature_importance(fi: Dict[str, float], title: str) -> go.Figure:
    if not fi:
        return go.Figure()
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure()
    fig.add_bar(x=vals, y=names, orientation="h")
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=10, t=40, b=20),
        height=300,
    )
    return fig

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[LABEL_MAP[c] for c in [0, 1, 2]],
            y=[LABEL_MAP[c] for c in [0, 1, 2]],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Confusion matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    return fig

def plot_reliability_curve(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> go.Figure:
    if proba is None or proba.ndim != 2 or proba.shape[1] < 1:
        return go.Figure()

    p = proba[:, 0]
    y_bin = (y_true == 0).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1
    xs: List[float] = []
    ys: List[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        xs.append(p[mask].mean())
        ys.append(y_bin[mask].mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash"), name="Perfect"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Model"))
    fig.update_layout(
        title="Reliability diagram (Home win probability)",
        xaxis_title="Predicted probability",
        yaxis_title="Observed frequency",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    return fig

# ============================== SIDEBAR NAV ==============================
st.sidebar.header("Navigation")
nav = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Bet Insights",               # NEW PAGE
        "Confusion Matrix & Metrics",
        "Feature Importance",
        "Match Simulator",
        "xG Confidence Interval",
        "Time Series Trends",
        "Model Comparison",
        "Reliability Diagram",
        "Downloads",
        "Individual players (API-FOOTBALL)",
    ],
    label_visibility="collapsed",
)

# ============================== SHARED TEAM PICKERS ==============================
if not FD_TOKEN:
    st.error("Add FOOTBALL_DATA_TOKEN to secrets.toml")
    st.stop()

teams_df = build_team_index()
if teams_df.empty:
    st.error("Could not fetch teams. Check plan/token.")
    st.stop()

names = teams_df["name"].tolist()
default_home = next((n for n in names if "Barcelona" in n), names[0])
default_away = next(
    (n for n in names if "Real Madrid" in n and n != default_home),
    names[min(1, len(names) - 1)],
)

col = st.columns([1.6, 1.6, 0.9, 1.0, 1.6])
with col[0]:
    st.markdown("**Home team**")
    home_name = st.selectbox(
        "",
        names,
        index=names.index(default_home) if default_home in names else 0,
        label_visibility="collapsed",
    )
with col[1]:
    st.markdown("**Away team**")
    away_name = st.selectbox(
        "",
        [n for n in names if n != home_name],
        index=0,
        label_visibility="collapsed",
    )
with col[2]:
    n_last = st.slider("Form window", 5, 15, 7)
with col[3]:
    home_adv = st.slider("Home advantage (+μ)", 0.00, 0.60, 0.25, 0.05)
with col[4]:
    model_choice = st.selectbox(
        "Model",
        [
            "Poisson model (fixed maths)",
            "Logistic Regression (classifier)",
            "Random Forest Classifier",
            "Random Forest Regressor",
            "XGBoost Classifier",
            "XGBoost Regressor",
            "SVM (SVC classifier)",
            "SVR (regressor)",
        ],
    )

home_id = int(teams_df[teams_df["name"] == home_name]["id"].iloc[0])
away_id = int(teams_df[teams_df["name"] == away_name]["id"].iloc[0])
crest_h = teams_df[teams_df["name"] == home_name]["crest"].iloc[0]
crest_a = teams_df[teams_df["name"] == away_name]["crest"].iloc[0]

dfH_full = team_last_matches(home_id, 20)
dfA_full = team_last_matches(away_id, 20)
if dfH_full.empty or dfA_full.empty:
    st.warning("Not enough recent data for one or both teams.")
    st.stop()

def roll_end(df: pd.DataFrame, tid: int, w: int) -> Tuple[float, float, float]:
    sel = df[(df["homeId"] == tid) | (df["awayId"] == tid)].tail(w)
    if sel.empty:
        return (1.4, 1.4, 0.0)
    gf = ga = 0
    for _, x in sel.iterrows():
        if x["homeId"] == tid:
            gf += x["hg"]
            ga += x["ag"]
        else:
            gf += x["ag"]
            ga += x["hg"]
    n = len(sel)
    return (gf / n, ga / n, (gf - ga) / n)

hgf, hga, hgd = roll_end(dfH_full, home_id, n_last)
agf, aga, agd = roll_end(dfA_full, away_id, n_last)
live = pd.Series(
    {
        "home_gf_pg": hgf,
        "home_ga_pg": hga,
        "away_gf_pg": agf,
        "away_ga_pg": aga,
    }
)
mu_h, mu_a = mu_from_fixed_math(live, home_adv)
pH, pD, pA = poisson_probs(mu_h, mu_a)

# ============================== PAGE: OVERVIEW ==============================
if nav == "Overview":
    topL, topM, topR = st.columns([1, 1, 1])
    with topL:
        st.markdown(
            f"<div class='teamname'><img class='crest' src='{crest_h}'/>{home_name}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='bigxg'>{mu_h:.2f}</div><span class='subtle'>Home xG (Poisson)</span>",
            unsafe_allow_html=True,
        )
    with topM:
        st.plotly_chart(slim_strength_bar(pH, pA), use_container_width=True)
        st.markdown(
            f"<div class='small-note' style='text-align:center'>"
            f"Win {pH*100:.1f}% • Draw {pD*100:.1f}% • Away win {pA*100:.1f}%"
            f"</div>",
            unsafe_allow_html=True,
        )
    with topR:
        st.markdown(
            f"<div style='display:flex;gap:10px;align-items:center;justify-content:flex-end' "
            f"class='teamname'><span style='text-align:right'>{away_name}</span>"
            f"<img class='crest' src='{crest_a}'/></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:right' class='bigxg'>{mu_a:.2f}</div>"
            f"<div style='text-align:right' class='subtle'>Away xG (Poisson)</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>Team stats (GF / GA / GD) over recent window</div>",
        unsafe_allow_html=True,
    )
    fig_stats = gf_ga_gd_chart(
        home_vals=(hgf, hga, hgd),
        away_vals=(agf, aga, agd),
    )
    st.plotly_chart(fig_stats, use_container_width=True)
    st.caption(
        "Overview page: quick view of xG and basic stats. "
        "Use the other pages on the left for deeper analysis."
    )

# ============================== PAGE: BET INSIGHTS ==========================
elif nav == "Bet Insights":
    st.markdown("### 🎯 Bet Insights (markets from xG)")

    # Build Poisson distributions for goals and corners
    GOAL_CAP = 10
    TOTAL_CAP = 14
    home_dist, away_dist, joint = poisson_joint(mu_h, mu_a, cap=GOAL_CAP)
    total_dist = poisson_pmf_array(mu_h + mu_a, cap=TOTAL_CAP)

    # approximate half xG split
    mu_h_half = mu_h * 0.5
    mu_a_half = mu_a * 0.5
    total_half_dist = poisson_pmf_array(mu_h_half + mu_a_half, cap=8)

    # approximate corners from total xG
    mu_corners = CORNERS_PER_XG * (mu_h + mu_a)
    corners_dist = poisson_pmf_array(mu_corners, cap=25)

    def pct(p: float) -> str:
        p = max(0.0, min(1.0, float(p)))
        return f"{p*100:.0f}%"

    def prob_over(dist: np.ndarray, line: float) -> float:
        k = int(math.floor(line + 1e-9)) + 1  # e.g. 0.5 -> 1
        k = min(k, len(dist) - 1)
        return float(dist[k:].sum())

    def prob_under(dist: np.ndarray, line: float) -> float:
        k = int(math.floor(line + 1e-9))      # e.g. 0.5 -> 0
        k = min(k, len(dist) - 1)
        return float(dist[: k + 1].sum())

    def dc_prob(dc: str, line: float, direction: str) -> float:
        thr_under = int(math.floor(line + 1e-9))
        thr_over = thr_under + 1
        cap = joint.shape[0] - 1
        acc = 0.0
        for i in range(cap + 1):
            for j in range(cap + 1):
                total = i + j
                if dc == "HD":      # Home/Draw
                    in_dc = i >= j
                elif dc == "HA":    # Home/Away
                    in_dc = i != j
                else:               # "DA": Draw/Away
                    in_dc = i <= j
                if direction == "under":
                    cond = in_dc and (total <= thr_under)
                else:
                    cond = in_dc and (total >= thr_over)
                if cond:
                    acc += joint[i, j]
        return acc

    def or_total_prob(side: str, line: float, direction: str) -> float:
        """Home or Over 2.5, Draw or Over 2.5, Away or Under 2.5 etc."""
        thr_under = int(math.floor(line + 1e-9))
        thr_over = thr_under + 1
        cap = joint.shape[0] - 1
        acc = 0.0
        for i in range(cap + 1):
            for j in range(cap + 1):
                total = i + j
                if side == "home":
                    base = i > j
                elif side == "draw":
                    base = i == j
                else:  # away
                    base = i < j
                if direction == "over":
                    tot_cond = total >= thr_over
                else:
                    tot_cond = total <= thr_under
                if base or tot_cond:
                    acc += joint[i, j]
        return acc

    # ---------- TEAM GOALS (first upload style) ----------
    team_rows: List[Tuple[str, float]] = []
    for label, dist, team_label in [
        (home_name, home_dist, "home"),
        (away_name, away_dist, "away"),
    ]:
        o05 = prob_over(dist, 0.5)
        u05 = prob_under(dist, 0.5)
        o15 = prob_over(dist, 1.5)
        u15 = prob_under(dist, 1.5)
        team_rows.extend(
            [
                (f"{label} over 0.5 goals", o05),
                (f"{label} under 0.5 goals", u05),
                (f"{label} over 1.5 goals", o15),
                (f"{label} under 1.5 goals", u15),
            ]
        )

    # ---------- FULL-TIME OVER / UNDER ----------
    ft_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ft_rows: List[Tuple[str, float, float]] = []
    for ln in ft_lines:
        ft_rows.append(
            (
                f"{ln:.1f} goals",
                prob_over(total_dist, ln),
                prob_under(total_dist, ln),
            )
        )

    # ---------- 1ST HALF / 2ND HALF OVER / UNDER (using total_half_dist) ----------
    half_lines = [0.5, 1.5, 2.5]
    half_rows: List[Tuple[str, float, float]] = []
    for ln in half_lines:
        half_rows.append(
            (
                f"{ln:.1f} goals",
                prob_over(total_half_dist, ln),
                prob_under(total_half_dist, ln),
            )
        )

    # ---------- DOUBLE CHANCE & OVER / UNDER (1.5, 2.5, 3.5) ----------
    dc_labels = [("Home/Draw", "HD"), ("Home/Away", "HA"), ("Draw/Away", "DA")]
    dc_lines = [1.5, 2.5, 3.5]
    dc_tables: Dict[float, List[Tuple[str, float, float]]] = {}
    for ln in dc_lines:
        rows_ln: List[Tuple[str, float, float]] = []
        for text, key in dc_labels:
            p_under = dc_prob(key, ln, "under")
            p_over = dc_prob(key, ln, "over")
            rows_ln.append((text, p_under, p_over))
        dc_tables[ln] = rows_ln

    # ---------- "TEAM OR OVER / UNDER 2.5" (Yes/No) ----------
    or_rows: List[Tuple[str, float, float]] = []
    for side_label, key in [("Home team or over 2.5", "home_over"),
                            ("Home team or under 2.5", "home_under"),
                            ("Draw or over 2.5", "draw_over"),
                            ("Draw or under 2.5", "draw_under"),
                            ("Away team or over 2.5", "away_over"),
                            ("Away team or under 2.5", "away_under")]:
        side, direction = side_label.split(" or ")[0].split()[0].lower(), "over"
        if "under" in side_label:
            direction = "under"
        if "Draw" in side_label:
            side = "draw"
        elif "Away" in side_label:
            side = "away"
        p_yes = or_total_prob(side, 2.5, "over" if "over" in side_label else "under")
        p_no = 1.0 - p_yes
        or_rows.append((side_label, p_yes, p_no))

    # ---------- MULTIGOALS ----------
    def multigoal_probs(dist: np.ndarray) -> List[Tuple[str, float]]:
        probs: List[Tuple[str, float]] = []
        # 1-2
        p_12 = float(dist[1:3].sum())
        # 1-3
        p_13 = float(dist[1:4].sum())
        # 2-3
        p_23 = float(dist[2:4].sum())
        # 4+
        p_4p = float(dist[4:].sum())
        probs.extend(
            [
                ("1-2 goals", p_12),
                ("1-3 goals", p_13),
                ("2-3 goals", p_23),
                ("4+ goals", p_4p),
            ]
        )
        return probs

    home_multi = multigoal_probs(home_dist)
    away_multi = multigoal_probs(away_dist)

    # ---------- CORNERS OVER / UNDER ----------
    corner_lines = [6.5, 7.5, 8.5, 9.5, 10.5]
    corner_rows: List[Tuple[str, float, float]] = []
    for ln in corner_lines:
        corner_rows.append(
            (
                f"{ln:.1f} corners",
                prob_over(corners_dist, ln),
                prob_under(corners_dist, ln),
            )
        )

    # ---------- RENDER BET PANEL ----------
    html = ""
    html += "<div class='bet-panel'>"
    html += (
        "<div class='bet-header'>"
        "<div class='bet-title'>BET INSIGHTS</div>"
        "<div class='bet-engine'>Poisson xG engine</div>"
        "</div>"
    )
    html += f"<div class='bet-subtitle'>Match: {home_name} vs {away_name}</div>"

    # Team goals table
    html += "<div class='bet-section-title'>Team goals (full time)</div>"
    html += "<table class='bet-table'>"
    html += "<tr><th>Market</th><th>Confidence</th></tr>"
    for label, p in team_rows:
        html += (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='val'>{pct(p)}</td></tr>"
        )
    html += "</table>"

    # Full-time Over/Under
    html += "<div class='bet-section-title'>Full time – over / under</div>"
    html += "<table class='bet-table'>"
    html += "<tr><th>Line</th><th>Over</th><th>Under</th></tr>"
    for label, pov, pun in ft_rows:
        html += (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='val'>{pct(pov)}</td>"
            f"<td class='val'>{pct(pun)}</td></tr>"
        )
    html += "</table>"

    # 1st half OU (using same probs as half_rows)
    html += "<div class='bet-section-title'>1st half – over / under</div>"
    html += "<table class='bet-table'>"
    html += "<tr><th>Line</th><th>Over</th><th>Under</th></tr>"
    for label, pov, pun in half_rows:
        html += (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='val'>{pct(pov)}</td>"
            f"<td class='val'>{pct(pun)}</td></tr>"
        )
    html += "</table>"

    # 2nd half OU – same xG split (you can change later)
    html += "<div class='bet-section-title'>2nd half – over / under</div>"
    html += "<table class='bet-table'>"
    html += "<tr><th>Line</th><th>Over</th><th>Under</th></tr>"
    for label, pov, pun in half_rows:
        html += (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='val'>{pct(pov)}</td>"
            f"<td class='val'>{pct(pun)}</td></tr>"
        )
    html += "</table>"

    # Double chance & OU
    for ln in dc_lines:
        html += (
            f"<div class='bet-section-title'>Double chance & over/under – {ln:.1f}</div>"
        )
        html += "<table class='bet-table'>"
        html += "<tr><th></th><th>Under</th><th>Over</th></tr>"
        for text, p_under, p_over in dc_tables[ln]:
            html += (
                f"<tr><td class='label'>{text}</td>"
                f"<td class='val'>{pct(p_under)}</td>"
                f"<td class='val'>{pct(p_over)}</td></tr>"
            )
        html += "</table>"

    # Home/Draw/Away OR over/under 2.5
    html += "<div class='bet-section-title'>Team or over / under 2.5</div>"
    html += "<table class='bet-table'>"
    html += "<tr><th>Market</th><th>Yes</th><th>No</th></tr>"
    for label, p_yes, p_no in or_rows:
        html += (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='val'>{pct(p_yes)}</td>"
            f"<td class='val'>{pct(p_no)}</td></tr>"
        )
    html += "</table>"

    # Home multigoals
    html += f"<div class='bet-section-title'>Home multigoals – {home_name}</div>"
    html += "<table class='bet-table'>"
    html += "<tr><th>Range</th><th>Confidence</th></tr>"
    for label, p in home_multi:
        html += (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='val'>{pct(p)}</td></tr>"
        )
    html += "</table>"

    # Away multigoals
    html += f"<div class='bet-section-title'>Away multigoals – {away_name}</div>"
    html += "<table class='bet-table'>"
    html += "<tr><th>Range</th><th>Confidence</th></tr>"
    for label, p in away_multi:
        html += (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='val'>{pct(p)}</td></tr>"
        )
    html += "</table>"

    # Corners OU
    html += "<div class='bet-section-title'>Corners – over / under</div>"
    html += "<table class='bet-table'>"
    html += "<tr><th>Line</th><th>Over</th><th>Under</th></tr>"
    for label, pov, pun in corner_rows:
        html += (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='val'>{pct(pov)}</td>"
            f"<td class='val'>{pct(pun)}</td></tr>"
        )
    html += "</table>"

    html += (
        "<div class='bet-footnote'>Probabilities are derived from expected goals "
        "(Poisson model). Halves assume xG is split evenly. Corners use a simple "
        "heuristic based on total xG. Educational only – not betting advice.</div>"
    )

    html += "</div>"  # end .bet-panel

    st.markdown(html, unsafe_allow_html=True)

# ============================== PAGE: CONFUSION & METRICS ====================
elif nav == "Confusion Matrix & Metrics":
    st.markdown("### 📊 Confusion Matrix & Performance Metrics")
    df_train = build_training_dataset(home_id, away_id)
    if df_train.empty:
        st.error("No training data available for these teams.")
    else:
        if st.button("Train / evaluate selected model", use_container_width=True):
            if model_choice == "Poisson model (fixed maths)":
                res = evaluate_poisson_on_df(df_train, home_adv)
            elif model_choice == "Logistic Regression (classifier)":
                res = train_classifier(df_train, "logreg", "Logistic Regression")
            elif model_choice == "Random Forest Classifier":
                res = train_classifier(df_train, "rf", "Random Forest Classifier")
            elif model_choice == "XGBoost Classifier":
                if not HAVE_XGB:
                    st.error("XGBoost not installed. Run `pip install xgboost`.")
                    st.stop()
                res = train_classifier(df_train, "xgb", "XGBoost Classifier")
            elif model_choice == "SVM (SVC classifier)":
                res = train_classifier(df_train, "svm", "SVM (SVC classifier)")
            elif model_choice == "Random Forest Regressor":
                res = train_regressor(df_train, "rf", "Random Forest Regressor")
            elif model_choice == "XGBoost Regressor":
                if not HAVE_XGB:
                    st.error("XGBoost not installed. Run `pip install xgboost`.")
                    st.stop()
                res = train_regressor(df_train, "xgb", "XGBoost Regressor")
            elif model_choice == "SVR (regressor)":
                res = train_regressor(df_train, "svr", "SVR (regressor)")
            else:
                st.error("Unknown model.")
                st.stop()

            if res["kind"] == "classifier":
                c = st.columns(4)
                c[0].metric("Accuracy", f"{res['acc']:.1f}%")
                c[1].metric("F1 (macro)", f"{res['f1']:.1f}%")
                c[2].metric(
                    "AUC (macro)",
                    f"{res['auc']:.1f}%" if not math.isnan(res["auc"]) else "N/A",
                )
                c[3].metric(
                    "Log loss (↓)",
                    f"{res['logloss']:.2f}" if not math.isnan(res["logloss"]) else "N/A",
                )
                st.caption(f"Rows: {res['rows']}")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Accuracy from predicted W/D/L", f"{res['acc']:.1f}%")
                c1.metric("F1 (macro)", f"{res['f1']:.1f}%")
                c2.metric(
                    "Home MAE / RMSE",
                    f"{res['mae_home']:.2f} / {res['rmse_home']:.2f}",
                )
                c2.metric(
                    "Away MAE / RMSE",
                    f"{res['mae_away']:.2f} / {res['rmse_away']:.2f}",
                )
                st.caption(
                    f"Rows: {res['rows']} (AUC/Log-loss not applicable for regressors)"
                )

            st.plotly_chart(
                plot_confusion(res["y_true"], res["y_pred"]), use_container_width=True
            )

# ============================== PAGE: FEATURE IMPORTANCE =====================
elif nav == "Feature Importance":
    st.markdown("### 🧠 Feature Importance (which stats matter most?)")
    df_train = build_training_dataset(home_id, away_id)
    if df_train.empty:
        st.error("No training data available for these teams.")
    else:
        if st.button(
            "Train & compute feature importance", use_container_width=True
        ):
            if model_choice == "Logistic Regression (classifier)":
                res = train_classifier(df_train, "logreg", "Logistic Regression")
            elif model_choice == "Random Forest Classifier":
                res = train_classifier(df_train, "rf", "Random Forest Classifier")
            elif model_choice == "XGBoost Classifier":
                if not HAVE_XGB:
                    st.error("XGBoost not installed. Run `pip install xgboost`.")
                    st.stop()
                res = train_classifier(df_train, "xgb", "XGBoost Classifier")
            elif model_choice == "Random Forest Regressor":
                res = train_regressor(df_train, "rf", "Random Forest Regressor")
            elif model_choice == "XGBoost Regressor":
                if not HAVE_XGB:
                    st.error("XGBoost not installed. Run `pip install xgboost`.")
                    st.stop()
                res = train_regressor(df_train, "xgb", "XGBoost Regressor")
            else:
                st.info(
                    "Feature importance is not available for this model choice."
                )
                st.stop()

            if res["feature_importance"] is None:
                st.info("This model does not expose feature importance.")
            else:
                st.plotly_chart(
                    plot_feature_importance(
                        res["feature_importance"],
                        f"Feature importance – {res['model_name']}",
                    ),
                    use_container_width=True,
                )

                items = sorted(
                    res["feature_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                top3 = [name for name, _ in items[:3]]
                st.markdown("#### 🔍 Explainable AI Summary")
                st.markdown(
                    f"> **Model selected:** {res['model_name']}  \n"
                    f"> Most influential features (for predicting result):"
                )
                for i, feat in enumerate(top3, start=1):
                    st.markdown(f"{i}. **{feat}**")

# ============================== PAGE: MATCH SIMULATOR =======================
elif nav == "Match Simulator":
    st.markdown("### 🎲 Match Simulator (Monte-Carlo using Poisson xG)")
    st.write(f"Current Poisson xG – Home: **{mu_h:.2f}**, Away: **{mu_a:.2f}**")
    sims = st.slider("Number of simulations", 500, 10000, 2000, step=500)

    rng = np.random.default_rng(42)
    home_goals = rng.poisson(mu_h, size=sims)
    away_goals = rng.poisson(mu_a, size=sims)

    home_win = (home_goals > away_goals).mean() * 100
    draw = (home_goals == away_goals).mean() * 100
    away_win = (home_goals < away_goals).mean() * 100
    btts = (np.logical_and(home_goals > 0, away_goals > 0)).mean() * 100
    over15 = (home_goals + away_goals > 1.5).mean() * 100
    over25 = (home_goals + away_goals > 2.5).mean() * 100
    over35 = (home_goals + away_goals > 3.5).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Home win probability", f"{home_win:.1f}%")
    c2.metric("Draw probability", f"{draw:.1f}%")
    c3.metric("Away win probability", f"{away_win:.1f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("BTTS (both teams score)", f"{btts:.1f}%")
    c5.metric("Over 2.5 goals", f"{over25:.1f}%")
    c6.metric("Over 3.5 goals", f"{over35:.1f}%")

    sc_counts = pd.Series(list(zip(home_goals, away_goals))).value_counts()
    df_scores = sc_counts.rename("Count").reset_index()
    df_scores.columns = ["HomeGoals", "AwayGoals", "Count"]
    df_scores["Prob_%"] = df_scores["Count"] * 100 / sims
    df_scores = df_scores.head(10)

    fig_cs = go.Figure()
    fig_cs.add_bar(
        x=[f"{r.HomeGoals}-{r.AwayGoals}" for _, r in df_scores.iterrows()],
        y=df_scores["Prob_%"],
    )
    fig_cs.update_layout(
        title="Most probable correct scores",
        xaxis_title="Scoreline",
        yaxis_title="Probability (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    st.plotly_chart(fig_cs, use_container_width=True)
    st.dataframe(df_scores, use_container_width=True)

# ============================== PAGE: xG CONFIDENCE INTERVAL =================
elif nav == "xG Confidence Interval":
    st.markdown("### 📏 xG Estimates with 95% Confidence Interval")
    ci_low_h = mu_h - 1.96 * math.sqrt(mu_h)
    ci_high_h = mu_h + 1.96 * math.sqrt(mu_h)
    ci_low_a = mu_a - 1.96 * math.sqrt(mu_a)
    ci_high_a = mu_a + 1.96 * math.sqrt(mu_a)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='teamname'><img class='crest' src='{crest_h}'/>{home_name}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Home xG", f"{mu_h:.2f}")
        st.write(f"95% CI ≈ **[{ci_low_h:.2f}, {ci_high_h:.2f}]**")
    with c2:
        st.markdown(
            f"<div style='display:flex;gap:10px;align-items:center;justify-content:flex-end' "
            f"class='teamname'><span style='text-align:right'>{away_name}</span>"
            f"<img class='crest' src='{crest_a}'/></div>",
            unsafe_allow_html=True,
        )
        st.metric("Away xG", f"{mu_a:.2f}")
        st.write(f"95% CI ≈ **[{ci_low_a:.2f}, {ci_high_a:.2f}]**")

    st.markdown(
        "<div class='small-note'>Confidence interval uses Poisson variance (√μ). "
        "It shows the uncertainty around each xG estimate.</div>",
        unsafe_allow_html=True,
    )

# ============================== PAGE: TIME SERIES TRENDS =====================
elif nav == "Time Series Trends":
    st.markdown("### 📈 Time Series Trends (GF & GA per match)")

    def make_ts(df: pd.DataFrame, team_id: int, name: str) -> go.Figure:
        df2 = df[(df["homeId"] == team_id) | (df["awayId"] == team_id)].copy()
        if df2.empty:
            return go.Figure()
        df2["date"] = pd.to_datetime(df2["date"])
        df2["gf"] = np.where(df2["homeId"] == team_id, df2["hg"], df2["ag"])
        df2["ga"] = np.where(df2["homeId"] == team_id, df2["ag"], df2["hg"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df2["date"], y=df2["gf"],
                                 mode="lines+markers", name="GF"))
        fig.add_trace(go.Scatter(x=df2["date"], y=df2["ga"],
                                 mode="lines+markers", name="GA"))
        fig.update_layout(
            title=name,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            height=250,
        )
        return fig

    ts1, ts2 = st.columns(2)
    with ts1:
        st.plotly_chart(make_ts(dfH_full, home_id, home_name),
                        use_container_width=True)
    with ts2:
        st.plotly_chart(make_ts(dfA_full, away_id, away_name),
                        use_container_width=True)

# ============================== PAGE: MODEL COMPARISON =======================
elif nav == "Model Comparison":
    st.markdown("### 📊 Model Comparison Dashboard")
    df_train = build_training_dataset(home_id, away_id)
    if df_train.empty:
        st.error("No training data available for these teams.")
    else:
        if st.button(
            "Run comparison (Poisson, Logistic, RF, XGBoost, SVM)",
            use_container_width=True,
        ):
            results: List[dict] = []
            results.append(evaluate_poisson_on_df(df_train, home_adv))
            results.append(train_classifier(df_train, "logreg", "Logistic Regression"))
            results.append(train_classifier(df_train, "rf", "Random Forest Classifier"))
            if HAVE_XGB:
                results.append(train_classifier(df_train, "xgb", "XGBoost Classifier"))
            results.append(train_classifier(df_train, "svm", "SVM (SVC classifier)"))

            rows: List[dict] = []
            for r in results:
                rows.append(
                    {
                        "Model": r["model_name"],
                        "Accuracy (%)": round(r["acc"], 1),
                        "F1 (macro %)": round(r["f1"], 1),
                        "AUC (macro %)": (
                            round(r["auc"], 1)
                            if not math.isnan(r["auc"])
                            else None
                        ),
                        "Log loss (↓)": (
                            round(r["logloss"], 2)
                            if not math.isnan(r["logloss"])
                            else None
                        ),
                        "Rows": r["rows"],
                    }
                )
            df_cmp = pd.DataFrame(rows)
            st.dataframe(df_cmp, use_container_width=True)
            st.session_state["cmp_table"] = df_cmp

# ============================== PAGE: RELIABILITY DIAGRAM ====================
elif nav == "Reliability Diagram":
    st.markdown("### 📐 Reliability Diagram (Calibration Plot)")
    df_train = build_training_dataset(home_id, away_id)
    if df_train.empty:
        st.error("No training data available for these teams.")
    else:
        if st.button(
            "Train model & draw calibration plot", use_container_width=True
        ):
            if model_choice == "Poisson model (fixed maths)":
                res = evaluate_poisson_on_df(df_train, home_adv)
            elif model_choice == "Logistic Regression (classifier)":
                res = train_classifier(df_train, "logreg", "Logistic Regression")
            elif model_choice == "Random Forest Classifier":
                res = train_classifier(df_train, "rf", "Random Forest Classifier")
            elif model_choice == "XGBoost Classifier":
                if not HAVE_XGB:
                    st.error("XGBoost not installed. Run `pip install xgboost`.")
                    st.stop()
                res = train_classifier(df_train, "xgb", "XGBoost Classifier")
            elif model_choice == "SVM (SVC classifier)":
                res = train_classifier(df_train, "svm", "SVM (SVC classifier)")
            else:
                st.info("Reliability diagram only applies to classifier models.")
                st.stop()

            fig_rel = plot_reliability_curve(res["y_true"], res["proba"])
            st.plotly_chart(fig_rel, use_container_width=True)

# ============================== PAGE: DOWNLOADS ==============================
elif nav == "Downloads":
    st.markdown("### 💾 Downloads")
    df_train = build_training_dataset(home_id, away_id)
    if df_train.empty:
        st.error("No training data available for these teams.")
    else:
        csv_train = df_train.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download training dataset (CSV)",
            data=csv_train,
            file_name=f"training_{home_id}_{away_id}.csv",
            mime="text/csv",
        )

        poiss = evaluate_poisson_on_df(df_train.tail(14), home_adv)
        probs = pd.DataFrame(poiss["proba"], columns=["P_home", "P_draw", "P_away"])
        csv_probs = probs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Poisson probabilities (CSV)",
            data=csv_probs,
            file_name="poisson_probabilities.csv",
            mime="text/csv",
        )

        if "cmp_table" in st.session_state:
            cmp_csv = st.session_state["cmp_table"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download last model comparison table (CSV)",
                data=cmp_csv,
                file_name="model_comparison.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "Run the **Model Comparison** page first to enable comparison download."
            )

# ===================== PAGE: INDIVIDUAL PLAYERS (API-FOOTBALL) ===============
else:
    # (unchanged from your previous version – kept as-is)
    st.markdown("### 👤 Individual player stats — API-FOOTBALL")
    if not AF_KEY:
        st.error("Add API_FOOTBALL_KEY to `.streamlit/secrets.toml` to use this tab.")
        st.stop()

    season = st.selectbox(
        "Season", list(range(2019, 2031)), index=list(range(2019, 2031)).index(2025)
    )

    with st.expander("If your API-FOOTBALL team IDs differ, type them here"):
        af_home_id = st.text_input("API-FOOTBALL home team id (optional)", "")
        af_away_id = st.text_input("API-FOOTBALL away team id (optional)", "")

    @st.cache_data(ttl=3600)
    def af_resolve_team_id_by_name(team_name: str) -> Optional[int]:
        try:
            r = af_get("/teams", params={"search": team_name})
            for t in r.get("response", []):
                nm = t["team"]["name"]
                if team_name.lower() in nm.lower() or nm.lower() in team_name.lower():
                    return int(t["team"]["id"])
        except Exception:
            pass
        return None

    if not af_home_id:
        hid = af_resolve_team_id_by_name(home_name)
        af_home_id = str(hid) if hid else ""
    if not af_away_id:
        aid = af_resolve_team_id_by_name(away_name)
        af_away_id = str(aid) if aid else ""

    # ---- rest of the player / XI / standings panel remains exactly as before ----
    # (You can paste your previous "Individual players (API-FOOTBALL)" code here.)
