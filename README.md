# Stock Forecasting and Valuation Pipeline

This project couples a short-horizon machine learning forecast with a discounted cash flow valuation, returning a directional trade recommendation (LONG, SHORT, or HOLD) based on whether the model-predicted price appears to disagree with intrinsic value.

## What the pipeline does

The pipeline is built around the principle that a single price forecast, taken in isolation, carries very little decision-useful information for an investor. A model can predict tomorrow's price with reasonable accuracy and still leave the user no closer to a sensible position because the forecast tells them nothing about whether the prevailing price is justified by the underlying economics of the firm. To address this, the pipeline runs two parallel valuations of the same security: a Random Forest regressor that produces a one-day-ahead price forecast (and from this an implied log-drift, μ), and a discounted cash flow model that produces an intrinsic equity value per share from a set of forecast revenues, margins, and capital assumptions. The two prices are then fed to a small C++ executable (`cpp/trade_logic_program`) that compares them and emits the directional recommendation. Splitting the decision logic into a separate compiled binary is overkill for a model of this size, but it demonstrates inter-process communication between Python and C++ and keeps the trading rule isolated from the modelling code.

## Time-series discipline

Look-ahead leakage is the single most common reason that backtested equity models appear strong in development and collapse out of sample. The pipeline therefore avoids `train_test_split` (which shuffles and would mix future bars into training data) and instead sorts the dataframe chronologically and slices it at a configurable train fraction. Hyperparameter tuning uses `TimeSeriesSplit` rather than ordinary k-fold so that every cross-validation fold respects the arrow of time. All engineered features (lagged returns, rolling means, rolling standard deviations) are shifted by one bar so that no row's features ever depend on its own target value.

## Repository layout

```
stock_price_valuation_model/
├── README.md
├── requirements.txt
├── data/
│   └── apple_stock_data.csv      # example daily bars
├── src/
│   ├── data_cleaner.py           # dedup, NaN handling, encoding, date conversion
│   ├── data_analysis.py          # feature engineering, RF tuning, train/test split
│   ├── dcf_model.py              # DCF valuation
│   └── main.py                   # pipeline entry point
└── cpp/
    ├── trade_logic_program.cpp   # decision rule source
    └── trade_logic_program       # compiled binary (rebuild if needed)
```

## Getting started

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If the C++ binary will not run on your platform, rebuild it:

```bash
cd cpp && g++ -O2 -o trade_logic_program trade_logic_program.cpp && chmod +x trade_logic_program
```

Then run the pipeline from the project root:

```bash
python src/main.py
```

## Example output

```
mae: 2.34
price tmr: 187.42
drift 1d: +0.0065
dcf price: 180.00
Stock is overvalued by $7.42. Recommended action: SHORT.
```

## Known limitations

The DCF assumptions in `src/main.py` (WACC, EBITDA margins, growth, capex intensity) are illustrative placeholders rather than carefully sourced inputs from the firm's filings, so the intrinsic price should be read as a worked example, not as a defensible valuation of Apple. The Random Forest currently targets price levels rather than returns, which makes the target non-stationary and biases the model toward recent regime levels; switching to log-returns is the recommended first improvement. Transaction costs, slippage, and multi-day horizons are not modelled, and the decision rule contains a hard-coded $90 buffer in the overvalued branch that should be reviewed before any practical use.

## Disclaimer

This project is for educational purposes only. It is not financial advice and must not be used to take real positions.
