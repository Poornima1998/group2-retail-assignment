
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
REPORTS_DIR = BASE_DIR / "reports"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 6)

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    customers = pd.read_csv(PROCESSED_DIR / "customers_clean.csv", parse_dates=["registration_date"])
    transactions = pd.read_csv(PROCESSED_DIR / "transactions_clean.csv", parse_dates=["transaction_date"])
    interactions = pd.read_csv(PROCESSED_DIR / "interactions_clean.csv", parse_dates=["interaction_date"])
    campaigns = pd.read_csv(PROCESSED_DIR / "campaigns_clean.csv", parse_dates=["start_date", "end_date"])
    reviews = pd.read_csv(PROCESSED_DIR / "reviews_clean.csv", parse_dates=["transaction_date", "review_date"])
    tickets = pd.read_csv(PROCESSED_DIR / "support_tickets_clean.csv", parse_dates=["submission_date", "resolution_date"])
    master = pd.read_csv(PROCESSED_DIR / "customer_master_table.csv", parse_dates=["registration_date", "last_transaction_date"])

    # Tables
    sales_by_category = transactions.groupby("product_category", as_index=False)["net_sales"].sum().sort_values("net_sales", ascending=False)
    sales_by_month = transactions.groupby("transaction_month", as_index=False)["net_sales"].sum().sort_values("transaction_month")
    payments = transactions["payment_method"].value_counts().rename_axis("payment_method").reset_index(name="count")
    rating_by_category = reviews.groupby("product_category", as_index=False)["rating"].mean().sort_values("rating", ascending=False)
    issue_counts = tickets["issue_category"].value_counts().rename_axis("issue_category").reset_index(name="count")
    campaign_perf = campaigns.groupby("campaign_type", as_index=False).agg(
        avg_roi=("roi", "mean"),
        avg_ctr=("ctr", "mean"),
        total_conversions=("conversions", "sum"),
        avg_cost_per_conversion=("cost_per_conversion", "mean"),
    ).sort_values("avg_roi", ascending=False)
    channel_counts = interactions["channel"].value_counts().rename_axis("channel").reset_index(name="count")

    sales_by_category.to_csv(TABLES_DIR / "sales_by_category.csv", index=False)
    sales_by_month.to_csv(TABLES_DIR / "monthly_sales.csv", index=False)
    payments.to_csv(TABLES_DIR / "payment_method_distribution.csv", index=False)
    rating_by_category.to_csv(TABLES_DIR / "rating_by_category.csv", index=False)
    issue_counts.to_csv(TABLES_DIR / "support_issue_counts.csv", index=False)
    campaign_perf.to_csv(TABLES_DIR / "campaign_performance_by_type.csv", index=False)
    channel_counts.to_csv(TABLES_DIR / "interaction_channel_counts.csv", index=False)

    # Plots
    top_cat = sales_by_category.head(10)
    plt.figure()
    plt.bar(top_cat["product_category"], top_cat["net_sales"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Product Categories by Net Sales")
    plt.xlabel("Product Category")
    plt.ylabel("Net Sales")
    save_plot(PLOTS_DIR / "top_categories_by_sales.png")

    plt.figure()
    plt.plot(sales_by_month["transaction_month"], sales_by_month["net_sales"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Net Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Net Sales")
    save_plot(PLOTS_DIR / "monthly_sales_trend.png")

    plt.figure()
    plt.bar(payments["payment_method"], payments["count"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Payment Method Distribution")
    plt.xlabel("Payment Method")
    plt.ylabel("Number of Transactions")
    save_plot(PLOTS_DIR / "payment_method_distribution.png")

    plt.figure()
    plt.hist(customers["age"].dropna(), bins=20)
    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    save_plot(PLOTS_DIR / "customer_age_distribution.png")

    plt.figure()
    plt.bar(channel_counts["channel"], channel_counts["count"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Interaction Channel Distribution")
    plt.xlabel("Channel")
    plt.ylabel("Count")
    save_plot(PLOTS_DIR / "interaction_channel_distribution.png")

    plt.figure()
    plt.scatter(campaigns["budget"], campaigns["conversions"])
    plt.title("Campaign Budget vs Conversions")
    plt.xlabel("Budget")
    plt.ylabel("Conversions")
    save_plot(PLOTS_DIR / "campaign_budget_vs_conversions.png")

    plt.figure()
    plt.bar(issue_counts["issue_category"], issue_counts["count"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Support Ticket Count by Issue Category")
    plt.xlabel("Issue Category")
    plt.ylabel("Ticket Count")
    save_plot(PLOTS_DIR / "support_issue_counts.png")

    plt.figure()
    plt.bar(rating_by_category["product_category"].head(10), rating_by_category["rating"].head(10))
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Rated Product Categories")
    plt.xlabel("Product Category")
    plt.ylabel("Average Rating")
    save_plot(PLOTS_DIR / "top_rated_categories.png")

    # Correlation matrix
    corr_cols = ["age", "total_orders", "total_net_sales", "avg_order_value", "total_interactions", "avg_rating", "total_tickets", "avg_satisfaction"]
    corr_df = master[corr_cols].corr(numeric_only=True)
    corr_df.to_csv(TABLES_DIR / "customer_master_correlations.csv")
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_df, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation Heatmap (Customer-Level Features)")
    save_plot(PLOTS_DIR / "customer_feature_correlation_heatmap.png")

    # Text summary
    summary = []
    summary.append("EDA SUMMARY")
    summary.append("=" * 70)
    summary.append(f"Customers: {len(customers):,}")
    summary.append(f"Transactions: {len(transactions):,}")
    summary.append(f"Interactions: {len(interactions):,}")
    summary.append(f"Campaigns: {len(campaigns):,}")
    summary.append(f"Reviews: {len(reviews):,}")
    summary.append(f"Support tickets: {len(tickets):,}")
    summary.append("")
    summary.append(f"Total net sales: {transactions['net_sales'].sum():,.2f}")
    summary.append(f"Average order value: {transactions['net_sales'].mean():,.2f}")
    summary.append(f"Top category by sales: {sales_by_category.iloc[0]['product_category']} ({sales_by_category.iloc[0]['net_sales']:.2f})")
    summary.append(f"Most used payment method: {payments.iloc[0]['payment_method']} ({payments.iloc[0]['count']})")
    summary.append(f"Most common interaction channel: {channel_counts.iloc[0]['channel']} ({channel_counts.iloc[0]['count']})")
    summary.append(f"Best campaign type by avg ROI: {campaign_perf.iloc[0]['campaign_type']} ({campaign_perf.iloc[0]['avg_roi']:.2f})")
    summary.append(f"Most common support issue: {issue_counts.iloc[0]['issue_category']} ({issue_counts.iloc[0]['count']})")
    summary.append(f"Highest rated category: {rating_by_category.iloc[0]['product_category']} ({rating_by_category.iloc[0]['rating']:.2f})")
    summary.append("")
    summary.append("Operational insights:")
    summary.append("- Focus sales planning around the highest-grossing product categories.")
    summary.append("- The dominant payment method and interaction channel should receive priority in UX and marketing optimization.")
    summary.append("- Campaign types with superior ROI and lower cost per conversion should receive proportionally larger budgets.")
    summary.append("- Support issue categories with heavy ticket volumes should be addressed through process improvements and proactive communication.")
    (REPORTS_DIR / "insights_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("EDA complete. Tables, plots, and summary saved.")

if __name__ == "__main__":
    main()
