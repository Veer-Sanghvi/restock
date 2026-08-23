"""Weekly demand forecasting that ends in an inventory decision.

Public rebuild of the supply chain half of my 2022 Corizo ML internship.
A global LightGBM model forecasts next-week demand for the best-selling
SKUs of a real retailer (UCI Online Retail), backtested with a rolling
origin against the two baselines every ops team already uses: last week's
number, and a 4-week moving average. Accuracy is then translated into the
unit that matters on a shop floor: safety stock at a 95 percent service
level.

Outputs: metrics.json + figures/.
"""
import json

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
RNG = 42
BACKTEST_WEEKS = 12
Z_95 = 1.645  # one-sided 95 percent service level

df = pd.read_csv("data/weekly_demand.csv", parse_dates=["week"])
df = df.sort_values(["sku", "week"]).reset_index(drop=True)

# the final calendar week of the export is partial; drop it
df = df[df["week"] < df["week"].max()]

# start each sku at its first sale so pre-launch zeros don't teach the
# model that demand was ever truly zero
df = df[df.groupby("sku")["units"].cumsum() > 0].reset_index(drop=True)

# ------------------------------------------------------------- features
df["lag1"] = df.groupby("sku")["units"].shift(1)
df["lag2"] = df.groupby("sku")["units"].shift(2)
df["lag3"] = df.groupby("sku")["units"].shift(3)
df["lag4"] = df.groupby("sku")["units"].shift(4)
df["ma4"] = df.groupby("sku")["units"].shift(1).rolling(4).mean().values
df["woy"] = df["week"].dt.isocalendar().week.astype(int)
df["sku_id"] = df["sku"].astype("category").cat.codes
df = df.dropna().reset_index(drop=True)

FEATURES = ["lag1", "lag2", "lag3", "lag4", "ma4", "woy", "sku_id"]

# ------------------------------------------- rolling-origin backtest
weeks = sorted(df["week"].unique())
test_weeks = weeks[-BACKTEST_WEEKS:]
rows = []
for wk in test_weeks:
    train = df[df["week"] < wk]
    test = df[df["week"] == wk].copy()
    model = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.06,
                              n_estimators=300, min_child_samples=20,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RNG, verbose=-1)
    model.fit(train[FEATURES], train["units"],
              categorical_feature=["sku_id"])
    test["pred_lgbm"] = np.clip(model.predict(test[FEATURES]), 0, None)
    test["pred_naive"] = test["lag1"]
    test["pred_ma4"] = test["ma4"]
    rows.append(test)
bt = pd.concat(rows, ignore_index=True)

def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))

MODELS = {"naive_last_week": "pred_naive", "moving_average_4wk": "pred_ma4",
          "lightgbm": "pred_lgbm"}
overall = {name: {"rmse": rmse(bt["units"], bt[col]),
                  "mae": float(np.mean(np.abs(bt["units"] - bt[col])))}
           for name, col in MODELS.items()}

# ---------------------------------- accuracy translated into inventory
# per-sku forecast error sigma sets safety stock: ss = z * sigma. Summed
# over the catalog, the difference between models is stock you hold, or
# don't, for the same service level.
per_sku = {}
for name, col in MODELS.items():
    sig = bt.groupby("sku").apply(
        lambda g, c=col: rmse(g["units"], g[c]), include_groups=False)
    per_sku[name] = sig
safety = {name: float(Z_95 * s.sum()) for name, s in per_sku.items()}

results = {
    "n_skus": int(df["sku"].nunique()),
    "backtest_weeks": BACKTEST_WEEKS,
    "one_step_ahead": True,
    "overall": overall,
    "safety_stock_units_at_95pct": {k: round(v) for k, v in safety.items()},
    "safety_stock_saved_vs_naive": {
        "moving_average_4wk": round(safety["naive_last_week"]
                                    - safety["moving_average_4wk"]),
        "lightgbm": round(safety["naive_last_week"] - safety["lightgbm"]),
    },
}
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# --------------------------------------------------------------- figures
DARK = {"bg": "#0d1117", "surface": "#161b22", "ink": "#c9d1d9",
        "muted": "#8b949e", "accent": "#39d353", "link": "#58a6ff",
        "warn": "#d29922"}

def dark_ax(ax):
    ax.set_facecolor(DARK["bg"])
    for s in ax.spines.values():
        s.set_color(DARK["muted"])
    ax.tick_params(colors=DARK["muted"])
    ax.xaxis.label.set_color(DARK["ink"])
    ax.yaxis.label.set_color(DARK["ink"])
    ax.title.set_color(DARK["ink"])

def dark_legend(ax):
    leg = ax.legend(facecolor=DARK["surface"], edgecolor=DARK["muted"])
    for t in leg.get_texts():
        t.set_color(DARK["ink"])

# 1) one SKU's story: history + backtest forecasts
top_sku = df.groupby("sku")["units"].sum().idxmax()
hist = df[df["sku"] == top_sku]
bts = bt[bt["sku"] == top_sku]
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150, facecolor=DARK["bg"])
dark_ax(ax)
ax.plot(hist["week"], hist["units"], color=DARK["ink"], lw=1.6,
        label="actual demand")
ax.plot(bts["week"], bts["pred_naive"], color=DARK["muted"], lw=1.4, ls="--",
        label="naive (last week)")
ax.plot(bts["week"], bts["pred_lgbm"], color=DARK["accent"], lw=2,
        label="LightGBM, one week ahead")
ax.axvspan(bts["week"].min(), bts["week"].max(), color=DARK["surface"])
ax.set_xlabel("week")
ax.set_ylabel("units per week")
ax.set_title(f"Best-selling SKU {top_sku}: 12-week rolling backtest")
dark_legend(ax)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("figures/backtest-top-sku.png", facecolor=DARK["bg"])

# 2) the decision: safety stock required at 95 percent service
fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150, facecolor=DARK["bg"])
dark_ax(ax)
names = ["naive_last_week", "moving_average_4wk", "lightgbm"]
labels = ["naive\n(last week)", "4-week\nmoving average", "LightGBM"]
vals = [safety[n] for n in names]
colors = [DARK["muted"], DARK["link"], DARK["accent"]]
bars = ax.bar(labels, vals, color=colors, alpha=0.9)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:,.0f}",
            ha="center", color=DARK["ink"], fontsize=11)
ax.set_ylabel(f"safety stock [units, {df.sku.nunique()} SKUs]")
ax.set_title("Stock you must hold for 95% service, by forecast method")
fig.tight_layout()
fig.savefig("figures/safety-stock.png", facecolor=DARK["bg"])
print("figures written")
