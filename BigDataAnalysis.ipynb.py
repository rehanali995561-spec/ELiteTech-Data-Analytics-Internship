"""
Generates a large synthetic global e-commerce transactions dataset.
Partitioned into multiple Parquet files (one per month) to simulate a
realistic big-data source that arrives in chunks.

Total size target: ~20 million rows across 24 partitions.
"""
import numpy as np
import pandas as pd
import os
import time

OUT_DIR = "/home/claude/bigdata_project/data/transactions"
os.makedirs(OUT_DIR, exist_ok=True)

RNG_SEED = 42
N_PARTITIONS = 24                # one per month over 2 years
ROWS_PER_PARTITION = 850_000      # ~20.4M rows total

CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Books",
              "Sports & Outdoors", "Beauty & Health", "Toys & Games", "Grocery"]
CATEGORY_PRICE_RANGE = {
    "Electronics": (30, 1500), "Clothing": (10, 200), "Home & Garden": (15, 800),
    "Books": (5, 60), "Sports & Outdoors": (10, 400), "Beauty & Health": (5, 150),
    "Toys & Games": (5, 250), "Grocery": (2, 80),
}
REGIONS = ["North America", "Europe", "Asia Pacific", "South America", "Africa", "Middle East"]
REGION_WEIGHTS = [0.32, 0.28, 0.22, 0.09, 0.05, 0.04]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Wallet", "Bank Transfer", "Cash on Delivery"]
GENDERS = ["Male", "Female", "Other"]

start_date = pd.Timestamp("2024-01-01")

overall_start = time.time()
total_rows = 0

for month_idx in range(N_PARTITIONS):
    rng = np.random.default_rng(RNG_SEED + month_idx)
    n = ROWS_PER_PARTITION

    month_start = start_date + pd.DateOffset(months=month_idx)
    days_in_month = (month_start + pd.DateOffset(months=1) - month_start).days

    category = rng.choice(CATEGORIES, size=n, p=[0.22, 0.18, 0.13, 0.08, 0.12, 0.11, 0.09, 0.07])
    unit_price = np.empty(n, dtype=np.float32)
    for cat in CATEGORIES:
        mask = category == cat
        lo, hi = CATEGORY_PRICE_RANGE[cat]
        unit_price[mask] = rng.uniform(lo, hi, size=mask.sum())

    quantity = rng.integers(1, 6, size=n).astype(np.int16)
    discount_pct = rng.choice([0, 0, 0, 5, 10, 15, 20, 25, 30], size=n).astype(np.int8)
    region = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=[0.34, 0.22, 0.18, 0.13, 0.08, 0.05])
    customer_age = rng.integers(16, 75, size=n).astype(np.int8)
    customer_gender = rng.choice(GENDERS, size=n, p=[0.47, 0.47, 0.06])

    day_offset = rng.integers(0, days_in_month, size=n)
    seconds_offset = rng.integers(0, 86400, size=n)
    order_date = month_start + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(seconds_offset, unit="s")

    # Weekend/holiday-season demand boost baked in via slightly higher qty in Nov/Dec
    if month_start.month in (11, 12):
        quantity = np.clip(quantity + rng.integers(0, 2, size=n), 1, 8).astype(np.int16)

    is_returned = rng.choice([0, 1], size=n, p=[0.93, 0.07]).astype(np.int8)

    df = pd.DataFrame({
        "order_id": np.arange(total_rows, total_rows + n, dtype=np.int64),
        "customer_id": rng.integers(1, 2_500_000, size=n, dtype=np.int64),
        "product_id": rng.integers(1, 50_000, size=n, dtype=np.int32),
        "category": category,
        "quantity": quantity,
        "unit_price": unit_price.round(2),
        "discount_pct": discount_pct,
        "order_date": order_date,
        "region": region,
        "payment_method": payment_method,
        "customer_age": customer_age,
        "customer_gender": customer_gender,
        "is_returned": is_returned,
    })

    fname = os.path.join(OUT_DIR, f"transactions_{month_start.strftime('%Y_%m')}.parquet")
    df.to_parquet(fname, index=False, engine="pyarrow", compression="snappy")
    total_rows += n
    print(f"[{month_idx+1:>2}/{N_PARTITIONS}] wrote {fname}  rows={n:,}  total={total_rows:,}")

elapsed = time.time() - overall_start
size_bytes = sum(
    os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)
)
print(f"\nDone. {total_rows:,} rows across {N_PARTITIONS} files in {elapsed:.1f}s")
print(f"Total on-disk size: {size_bytes / 1e6:.1f} MB")
