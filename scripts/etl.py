
from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILE_MAP = {
    "customers": "customers.csv",
    "transactions": "transactions.csv",
    "interactions": "interactions.csv",
    "campaigns": "campaigns.csv",
    "reviews": "customer_reviews_complete.csv",
    "support_tickets": "support_tickets.csv",
}

DATE_COLUMNS = {
    "customers": ["registration_date"],
    "transactions": ["transaction_date"],
    "interactions": ["interaction_date"],
    "campaigns": ["start_date", "end_date"],
    "reviews": ["transaction_date", "review_date"],
    "support_tickets": ["submission_date", "resolution_date"],
}

NUMERIC_COLUMNS = {
    "customers": ["age", "zip_code"],
    "transactions": ["quantity", "price", "discount_applied"],
    "interactions": ["duration"],
    "campaigns": ["budget", "impressions", "clicks", "conversions", "conversion_rate", "roi"],
    "reviews": ["rating"],
    "support_tickets": ["resolution_time_hours", "customer_satisfaction_score"],
}

TEXT_STANDARDIZATION = {
    "preferred_channel": {
        "in-store": "In-Store", "instore": "In-Store", "store": "In-Store",
        "email": "Email", "sms": "SMS", "phone": "Phone"
    },
    "channel": {
        "web": "Website", "website": "Website", "mobile app": "Mobile App",
        "app": "Mobile App", "social media": "Social Media", "social": "Social Media",
        "email": "Email"
    },
    "interaction_type": {
        "view_product": "View Product", "view product": "View Product",
        "add_to_cart": "Add to Cart", "cart_add": "Add to Cart",
        "click_ad": "Click Ad", "click ad": "Click Ad", "review": "Review",
        "wishlist_add": "Wishlist Add", "purchase": "Purchase"
    },
    "payment_method": {
        "credit card": "Credit Card", "paypal": "PayPal", "cash": "Cash",
        "gift card": "Gift Card", "debit card": "Debit Card", "bank transfer": "Bank Transfer"
    },
    "resolution_status": {
        "resolved": "Resolved", "closed": "Closed", "pending": "Pending", "open": "Open"
    },
    "priority": {
        "low": "Low", "medium": "Medium", "high": "High", "urgent": "Urgent"
    }
}


def standardize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def normalize_text_value(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    return " ".join(value.split())


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in TEXT_STANDARDIZATION.items():
        if col in df.columns:
            def mapper(x):
                if pd.isna(x):
                    return np.nan
                key = str(x).strip().lower()
                return mapping.get(key, " ".join(str(x).strip().split()).title())
            df[col] = df[col].apply(mapper)
    return df


def cap_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    caps = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = int(((s < lower) | (s > upper)).sum())
        s = s.clip(lower=lower, upper=upper)
        after = int(((s < lower) | (s > upper)).sum())
        df[col] = s
        caps[col] = {
            "lower_cap": None if pd.isna(lower) else float(lower),
            "upper_cap": None if pd.isna(upper) else float(upper),
            "values_capped": before - after,
        }
    return df, caps


def clean_dataframe(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"dataset": name, "initial_rows": int(len(df)), "initial_columns": int(df.shape[1])}
    df = standardize_colnames(df)

    # strip object fields
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(normalize_text_value)

    # parse dates
    for col in DATE_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # numeric conversion
    for col in NUMERIC_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_before = df.isna().sum().to_dict()

    # duplicate removal
    before_dupes = len(df)
    id_cols = [c for c in df.columns if c.endswith("_id")]
    if id_cols:
        subset = [id_cols[0]]
        df = df.drop_duplicates(subset=subset, keep="first")
    else:
        df = df.drop_duplicates()
    report["duplicates_removed"] = int(before_dupes - len(df))

    # standardize categories
    df = standardize_categories(df)

    # fill categorical missing
    for col in df.select_dtypes(include="object").columns:
        if col == "notes":
            df[col] = df[col].fillna("No notes provided")
        else:
            df[col] = df[col].fillna("Unknown")

    # fill numeric missing
    for col in df.select_dtypes(include=["number"]).columns:
        if name == "transactions" and col == "discount_applied":
            df[col] = df[col].fillna(0)
        elif name == "campaigns" and col in {"clicks", "conversions", "impressions"}:
            df[col] = df[col].fillna(0)
        else:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    # domain fixes
    if name == "customers":
        if "age" in df.columns:
            df["age"] = df["age"].clip(lower=18, upper=90)
            df["age_group"] = pd.cut(
                df["age"],
                bins=[17, 24, 34, 44, 54, 64, 100],
                labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
            )
        if "preferred_channel" in df.columns:
            df["preferred_channel"] = df["preferred_channel"].replace("Unknown", "Email")
        if "zip_code" in df.columns:
            df["zip_code"] = df["zip_code"].round(0).astype("Int64").astype(str)
    elif name == "transactions":
        for col in ["quantity", "price", "discount_applied"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        df, caps = cap_outliers_iqr(df, ["quantity", "price", "discount_applied"])
        report["outlier_caps"] = caps
        if {"quantity", "price"}.issubset(df.columns):
            df["gross_sales"] = df["quantity"] * df["price"]
            df["net_sales"] = (df["gross_sales"] - df["discount_applied"]).clip(lower=0)
        if "transaction_date" in df.columns:
            df["transaction_year"] = df["transaction_date"].dt.year
            df["transaction_month"] = df["transaction_date"].dt.to_period("M").astype(str)
            df["transaction_quarter"] = df["transaction_date"].dt.to_period("Q").astype(str)
    elif name == "interactions":
        if "duration" in df.columns:
            df["duration"] = df["duration"].clip(lower=0)
            df, caps = cap_outliers_iqr(df, ["duration"])
            report["outlier_caps"] = caps
        if "interaction_date" in df.columns:
            df["interaction_month"] = df["interaction_date"].dt.to_period("M").astype(str)
    elif name == "campaigns":
        for col in ["budget", "impressions", "clicks", "conversions"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        if {"start_date", "end_date"}.issubset(df.columns):
            df["campaign_duration_days"] = (df["end_date"] - df["start_date"]).dt.days.clip(lower=0)
        if {"clicks", "impressions"}.issubset(df.columns):
            df["ctr"] = np.where(df["impressions"] > 0, (df["clicks"] / df["impressions"]) * 100, 0)
        if {"conversions", "clicks"}.issubset(df.columns):
            df["click_to_conversion_rate"] = np.where(df["clicks"] > 0, (df["conversions"] / df["clicks"]) * 100, 0)
        if {"budget", "conversions"}.issubset(df.columns):
            df["cost_per_conversion"] = np.where(df["conversions"] > 0, df["budget"] / df["conversions"], 0)
    elif name == "reviews":
        if "rating" in df.columns:
            df["rating"] = df["rating"].clip(lower=1, upper=5)
        if {"review_date", "transaction_date"}.issubset(df.columns):
            df["review_lag_days"] = (df["review_date"] - df["transaction_date"]).dt.days
            df["review_lag_days"] = df["review_lag_days"].clip(lower=0)
    elif name == "support_tickets":
        if "resolution_time_hours" in df.columns:
            df["resolution_time_hours"] = df["resolution_time_hours"].clip(lower=0)
            df, caps = cap_outliers_iqr(df, ["resolution_time_hours"])
            report["outlier_caps"] = caps
        if "customer_satisfaction_score" in df.columns:
            df["customer_satisfaction_score"] = df["customer_satisfaction_score"].clip(lower=1, upper=5)
        if {"submission_date", "resolution_date"}.issubset(df.columns):
            df["resolution_lag_hours_calc"] = (
                (df["resolution_date"] - df["submission_date"]).dt.total_seconds() / 3600
            )
            df["resolution_lag_hours_calc"] = df["resolution_lag_hours_calc"].clip(lower=0)

    missing_after = df.isna().sum().to_dict()

    report["final_rows"] = int(len(df))
    report["final_columns"] = int(df.shape[1])
    report["missing_before"] = {k: int(v) for k, v in missing_before.items() if int(v) > 0}
    report["missing_after"] = {k: int(v) for k, v in missing_after.items() if int(v) > 0}
    report["columns"] = list(df.columns)
    return df, report


def build_master_table(customers, transactions, interactions, reviews, tickets):
    tx = transactions.groupby("customer_id", as_index=False).agg(
        total_orders=("transaction_id", "count"),
        total_quantity=("quantity", "sum"),
        total_gross_sales=("gross_sales", "sum"),
        total_net_sales=("net_sales", "sum"),
        avg_order_value=("net_sales", "mean"),
        avg_discount=("discount_applied", "mean"),
        last_transaction_date=("transaction_date", "max")
    )
    ix = interactions.groupby("customer_id", as_index=False).agg(
        total_interactions=("interaction_id", "count"),
        avg_interaction_duration=("duration", "mean"),
        distinct_sessions=("session_id", "nunique")
    )
    rv = reviews.groupby("customer_id", as_index=False).agg(
        review_count=("review_id", "count"),
        avg_rating=("rating", "mean")
    )
    st = tickets.groupby("customer_id", as_index=False).agg(
        total_tickets=("ticket_id", "count"),
        avg_resolution_hours=("resolution_time_hours", "mean"),
        avg_satisfaction=("customer_satisfaction_score", "mean")
    )
    master = customers.merge(tx, on="customer_id", how="left")
    master = master.merge(ix, on="customer_id", how="left")
    master = master.merge(rv, on="customer_id", how="left")
    master = master.merge(st, on="customer_id", how="left")

    fill_zero_cols = [
        "total_orders", "total_quantity", "total_gross_sales", "total_net_sales", "avg_order_value",
        "avg_discount", "total_interactions", "avg_interaction_duration", "distinct_sessions",
        "review_count", "avg_rating", "total_tickets", "avg_resolution_hours", "avg_satisfaction"
    ]
    for col in fill_zero_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0)
    return master


def main():
    cleaned = {}
    reports = {}
    for name, filename in FILE_MAP.items():
        df = pd.read_csv(RAW_DIR / filename)
        clean_df, report = clean_dataframe(name, df)
        cleaned[name] = clean_df
        reports[name] = report

    # Save cleaned datasets
    save_names = {
        "customers": "customers_clean.csv",
        "transactions": "transactions_clean.csv",
        "interactions": "interactions_clean.csv",
        "campaigns": "campaigns_clean.csv",
        "reviews": "reviews_clean.csv",
        "support_tickets": "support_tickets_clean.csv",
    }
    for key, outname in save_names.items():
        cleaned[key].to_csv(PROCESSED_DIR / outname, index=False)

    # Integrated table
    master = build_master_table(
        cleaned["customers"], cleaned["transactions"], cleaned["interactions"],
        cleaned["reviews"], cleaned["support_tickets"]
    )
    master.to_csv(PROCESSED_DIR / "customer_master_table.csv", index=False)

    overall = {
        "datasets_processed": list(FILE_MAP.keys()),
        "master_table_rows": int(len(master)),
        "master_table_columns": int(master.shape[1]),
        "per_dataset_report": reports,
        "assumptions": [
            "Missing categorical values filled with 'Unknown' except preferred_channel defaulted to 'Email'.",
            "Missing numeric values imputed with median unless zero is more appropriate for counts/discounts.",
            "Outliers capped using IQR method instead of deleted to preserve business records.",
            "Negative or logically invalid numeric values clipped to zero or valid ranges."
        ]
    }

    with open(REPORTS_DIR / "etl_report.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, default=str)

    summary_lines = [
        "ETL SUMMARY",
        "=" * 60,
        f"Datasets processed: {', '.join(FILE_MAP.keys())}",
        f"Customer master table shape: {master.shape}",
        "",
    ]
    for name, report in reports.items():
        summary_lines.append(f"{name.upper()}:")
        summary_lines.append(f"  Initial rows/cols: {report['initial_rows']} / {report['initial_columns']}")
        summary_lines.append(f"  Final rows/cols:   {report['final_rows']} / {report['final_columns']}")
        summary_lines.append(f"  Duplicates removed: {report['duplicates_removed']}")
        if report["missing_before"]:
            summary_lines.append(f"  Missing before: {report['missing_before']}")
        if report["missing_after"]:
            summary_lines.append(f"  Missing after: {report['missing_after']}")
        summary_lines.append("")
    (REPORTS_DIR / "etl_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("ETL complete. Cleaned files and reports saved.")


if __name__ == "__main__":
    main()
