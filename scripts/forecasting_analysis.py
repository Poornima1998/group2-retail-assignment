import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_forecasting():
    # 1. SETUP PATHS
    # We use the absolute path to ensure it runs correctly on your current machine
    base_path = r'C:\Users\ASUS TUF F15\group2-retail-assignment'
    data_path = os.path.join(base_path, 'outputs', 'tables', 'monthly_sales_series.csv')
    output_plot_path = os.path.join(base_path, 'outputs', 'plots', 'forecast_comparison.png')
    output_csv_path = os.path.join(base_path, 'outputs', 'tables', 'forecast_metrics_summary.csv')

    # 2. LOAD DATA
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print("Error: Data file not found. Please check the path.")
        return

    df = pd.read_csv(data_path)
    df['month'] = pd.to_datetime(df['month'])
    df = df.set_index('month').sort_index()

    # 3. TRAIN-TEST SPLIT (80/20)
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    # 4. BASELINE MODEL: 3-MONTH MOVING AVERAGE
    train['ma3_forecast'] = train['net_sales'].rolling(window=3, min_periods=1).mean()
    test['ma3_forecast'] = train['ma3_forecast'].iloc[-1]
    
    ma3_mae = mean_absolute_error(test['net_sales'], test['ma3_forecast'])
    ma3_rmse = np.sqrt(mean_squared_error(test['net_sales'], test['ma3_forecast']))

    # 5. PROPHET MODEL
    print("Training Prophet model...")
    prophet_df = train.reset_index().rename(columns={'month': 'ds', 'net_sales': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=len(test), freq='MS')
    forecast = model.predict(future)
    prophet_forecast = forecast[forecast['ds'].isin(test.index)]['yhat'].values

    prophet_mae = mean_absolute_error(test['net_sales'], prophet_forecast)
    prophet_rmse = np.sqrt(mean_squared_error(test['net_sales'], prophet_forecast))

    # 6. SAVE METRICS
    results = pd.DataFrame({
        'Model': ['3-Month Moving Avg', 'Prophet'],
        'MAE': [ma3_mae, prophet_mae],
        'RMSE': [ma3_rmse, prophet_rmse]
    })
    results.to_csv(output_csv_path, index=False)
    print(f"Metrics saved to: {output_csv_path}")
    print(results.round(2))

    # 7. VISUALIZATION
    print("Generating forecast plot...")
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['net_sales'], 'b-', label='Actual Sales', linewidth=2)
    plt.plot(test.index, test['ma3_forecast'], 'r--', label='MA3 Baseline')
    plt.plot(test.index, prophet_forecast, 'g-', label='Prophet Forecast', linewidth=2)
    
    plt.title('Retail Sales Forecasting: Prophet vs Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    print(f"Plot saved to: {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    run_forecasting()