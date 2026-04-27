import subprocess
from pathlib import Path

import pandas as pd

import data_cleaner
import data_analysis
import dcf_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'apple_stock_data.csv'
TRADE_LOGIC_BINARY = PROJECT_ROOT / 'cpp' / 'trade_logic_program'

TARGET_COLUMN = 'close_price'

REVENUES = [1_000_000_000, 1_100_000_000, 1_210_000_000, 1_331_000_000, 1_464_000_000]
EBITDA_MARGINS = [0.60, 0.68, 0.7, 0.7, 0.7]

DCF_ASSUMPTIONS = {
    "wacc": 0.09,
    "terminal_growth_rate": 0.025,
    "tax_rate": 0.21,
    "d_a_percent_revenue": 0.03,
    "nwc_percent_revenue": 0.06,
    "capex_percent_revenue": 0.05,
    "net_debt": 1000,
    "shares_outstanding": 100_000_000,
}


def run_pipeline():
    """Run the full clean -> features -> ML -> DCF -> recommendation pipeline."""
    raw = pd.read_csv(DATA_PATH)

    print("Cleaning data...")
    deduped = data_cleaner.drop_duplicate_rows_cols(raw)
    nan_handled = data_cleaner.nan_handling(deduped)
    dated = data_cleaner.convert_dates_to_timestamp(nan_handled)
    cleaned = data_cleaner.encode_strings(dated)

    print('Extracting key features from the cleaned data...')
    feature_frame = data_analysis.include_stock_analysis(cleaned)

    print('Running the model and DCF analysis...')
    price_tomorrow, drift_one_day, mae = data_analysis.run_model(
        feature_frame, TARGET_COLUMN
    )

    _, _, _, dcf_implied_price = dcf_model.calculate_dcf(
        REVENUES, EBITDA_MARGINS, **DCF_ASSUMPTIONS
    )

    print(f"mae: {mae:.4f}")
    print(f"price tmr: {price_tomorrow:.2f}")
    print(f"drift 1d: {drift_one_day:+.4f}")
    print(f"dcf price: {dcf_implied_price:.2f}")

    command = [str(TRADE_LOGIC_BINARY), str(dcf_implied_price), str(price_tomorrow)]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout.strip())
    except FileNotFoundError:
        print("trade logic binary not found, build cpp/trade_logic_program.cpp first")


if __name__ == '__main__':
    run_pipeline()
