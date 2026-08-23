"""Build the weekly demand table from the raw UCI Online Retail export.

Raw source: UCI ML Repository, Online Retail (ID 352), a UK online
retailer's transactions from Dec 2010 to Dec 2011. The raw 23 MB xlsx is
not committed; this script turns it into data/weekly_demand.csv (small,
committed) so the modeling step is reproducible without the download.

Usage: python prepare_data.py /path/to/"Online Retail.xlsx"
"""
import sys

import pandas as pd

TOP_N = 25

raw = pd.read_excel(sys.argv[1])

# Cleaning decisions, each one visible:
# - drop cancellations (invoice numbers starting with C) and negative
#   quantities (returns); demand here means what customers ordered
# - drop rows without a stock code or description
# - drop non-product stock codes (postage, manuals, bank charges)
raw = raw[~raw["InvoiceNo"].astype(str).str.startswith("C")]
raw = raw[raw["Quantity"] > 0]
raw = raw.dropna(subset=["StockCode", "Description"])
raw["StockCode"] = raw["StockCode"].astype(str)
raw = raw[raw["StockCode"].str.match(r"^\d{5}[A-Za-z]?$")]

raw["week"] = raw["InvoiceDate"].dt.to_period("W-SUN").dt.start_time

# top sellers by total units, for series stable enough to forecast weekly
top = (raw.groupby("StockCode")["Quantity"].sum()
       .sort_values(ascending=False).head(TOP_N).index)
weekly = (raw[raw["StockCode"].isin(top)]
          .groupby(["StockCode", "week"])["Quantity"].sum()
          .reset_index()
          .rename(columns={"StockCode": "sku", "Quantity": "units"}))

# dense weekly grid per sku (missing week = zero demand that week)
full = []
for sku, g in weekly.groupby("sku"):
    idx = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="W-MON")
    s = g.set_index("week")["units"].reindex(idx, fill_value=0)
    full.append(pd.DataFrame({"sku": sku, "week": idx, "units": s.values}))
out = pd.concat(full, ignore_index=True)
assert len(out) > 1000 and out["sku"].nunique() == TOP_N, \
    "rebuild sanity check failed: source export may have changed"
out.to_csv("data/weekly_demand.csv", index=False)
print(f"wrote data/weekly_demand.csv: {out.sku.nunique()} skus x "
      f"{out.week.nunique()} weeks, {len(out)} rows")
