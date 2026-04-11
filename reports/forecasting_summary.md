# Forecasting Summary - Member 4

## Dataset
- `monthly_sales_series.csv`: 60 months net sales (2020-2025)
- Derived from cleaned transaction data.

## Forecasting Approach
- **Train/Test**: 48 months train, 12 months test (time-based split)
- **Models**: 3-month moving average (baseline) + Prophet
- **Metrics**: MAE, RMSE

## Results
| Model | MAE | RMSE |
|-------|-----|------|
| 3-Month MA | [178477.70] | [247634.68] |
| Prophet | [157954.95] | [174796.80] |

- The high RMSE is due to the model being surprised by the unprecedented sales peak in late 2024, which exceeded historical seasonal patterns.

## Business Implications
1. **Inventory**: Forecast peaks in Nov-Dec for holiday stocking
2. **Staffing**: Plan extra support for high-sales months
3. **Campaign timing**: Target high-demand periods
4. **Budgeting**: Predict revenue for next 6 months

## Limitations & Future Work
- Monthly granularity limits daily planning
- Add exogenous variables (campaigns, holidays)
- Test other models (SARIMA, LSTM)




