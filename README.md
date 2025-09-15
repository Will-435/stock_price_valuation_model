# Stock Forecasting & Valuation Pipeline

This project combines **machine learning** with **financial valuation**.  
It predicts short-term stock **drift** (expected 1-day return) using a Random Forest model and compares those forecasts to a **Discounted Cash Flow (DCF)** intrinsic valuation. The output is a directional recommendation (LONG, SHORT, or HOLD) based on whether the market price appears under- or over-valued.

---

## Project Highlights

- **End-to-end ML pipeline**
  - Data cleaning (duplicate removal, NaN handling, type conversion)
  - Encoding categorical variables with one-hot encoding
  - Converting dates into numeric format (timestamp ordinals)

- **Time-series awareness**
  - Chronological train/test split (no look-ahead bias)
  - Hyperparameter tuning with `TimeSeriesSplit` and `RandomizedSearchCV`

- **Feature engineering**
  - Lagged returns (yesterday, week, month)
  - Rolling averages (mean week/month)
  - Rolling volatility (std dev week/month)

- **Modeling**
  - Random Forest Regressor with randomized hyperparameter search
  - Evaluation via out-of-sample **Mean Absolute Error (MAE)**

- **Finance integration**
  - Simple DCF model for equity value per share
  - Comparison of ML-predicted market price vs intrinsic DCF price
  - Recommendation logic: LONG if undervalued, SHORT if overvalued

---

## Repository Structure

```
├── data_cleaner.py      # Functions for NaN handling, duplicate removal, encoding
├── data_analysis.py     # Feature engineering, model tuning, drift calculation
├── dcf_model.py         # Simple DCF implementation
├── final.py             # Main entrypoint combining ML + DCF
├── csv_Files/           # Example stock data (e.g. Apple)
└── README.md            # This file
```

---

##  How It Works

1. **Data import & cleaning**
   - Remove duplicates
   - Handle NaNs (drop or impute median)
   - Convert dates to timestamps
   - Encode string features into numerical form

2. **Feature engineering**
   - Add lagged returns, rolling means, and rolling volatilities

3. **Model training**
   - Random Forest tuned with `RandomizedSearchCV`
   - Cross-validated using `TimeSeriesSplit`

4. **Forecasting**
   - Predict **tomorrow’s price** and compute **1-day drift (μ)**  
     \[
     \mu_{1d} = \ln\left(\frac{P_{t+1}}{P_t}\right)
     \]

5. **Valuation (DCF)**
   - Project revenues, EBITDA margins, tax, capex, NWC, etc.
   - Apply discount rate (WACC) and terminal growth
   - Calculate implied equity value per share

6. **Decision**
   - If ML forecast > DCF price → SHORT (overvalued)
   - If ML forecast < DCF price → LONG (undervalued)
   - Else → HOLD

---

## Example Output

```text
Mean Absolute Error: 2.34
Predicted price tomorrow: $187.42
Predicted drift (μ): +0.65%

DCF implied value: $180.00
Stock is overvalued by $7.42 → Recommended action: SHORT
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended versions (to avoid breaking changes):

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
scipy>=1.11
```

### Run the pipeline
```bash
python final.py
```

---

## Applications

- **Investment Banking (IBD)** → understand DCF valuation and compare with market prices  
- **Risk** → volatility-based features, time-aware cross-validation, leakage prevention  
- **Data Science** → end-to-end pipeline with preprocessing, feature engineering, and model tuning  
- **Quant Research** → μ (expected return) forecasting, volatility modeling, and decision rules  

---

## Limitations

- DCF parameters (WACC, margins, growth rates) are placeholders — not investment advice.  
- Model currently predicts 1-day price levels; targeting returns directly would improve stationarity.  
- Transaction costs, market impact, and multi-day horizons not yet included.  

---

## Next Steps

- Train directly on **returns** instead of price levels  
- Add separate model for **volatility forecasting**  
- Implement **purged/embargoed CV** for multi-day horizons  
- Backtest strategy including **transaction costs**  
- Expand DCF with real financial statement data  

---

## Disclaimer
This project is for **educational purposes only**.  
It is **not financial advice** and should not be used for actual trading decisions.
