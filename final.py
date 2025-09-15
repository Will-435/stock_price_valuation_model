import pandas as pd

# Import the functions from other files in the directory
import data_cleaner
import data_analysis
import dcf_model


# Import the data as a csv file and convert it into a pandas dataframe
df = pd.read_csv('csv_Files/apple_stock_data.csv')
target = 'close_price'

# Clean the data
df_0 = data_cleaner.drop_duplicate_rows_cols(df)

df_1 = data_cleaner.nan_handling(df_0)

df_2 = data_cleaner.convert_dates_to_timestamp(df_1)

df_clean = data_cleaner.encode_strings(df_2)

# Insert calculated values and analysis to help the model
df_3 = data_analysis.include_stock_analysis(df_clean)

price_tmr, drift_1, mae_prediction = data_analysis.run_model(df_3, target)


# DCF model parematers
# Bare in mind that these are guessed and are not accurately calculated, therefore thsi isn't accurate trading advice

revenues = [1000000000, 1100000000, 1210000000, 1331000000, 1464000000]  # 10% growth
ebitda_margins = [0.60, 0.68, 0.7, 0.7, 0.7]

assumptions = {
    "wacc": 0.09,
    "terminal_growth_rate": 0.025,
    "tax_rate": 0.21,
    "d_a_percent_revenue": 0.03,
    "nwc_percent_revenue": 0.06,
    "capex_percent_revenue": 0.05,
    "net_debt": 1000,
    "shares_outstanding": 100_000_000
}

# Run the DCf model with our paramaters 
dcf_df, enterprise_value, equity_value, dcf_implied_price = dcf_model.calculate_dcf(revenues, ebitda_margins, **assumptions)

if dcf_implied_price < price_tmr:
    delta = price_tmr - dcf_implied_price
    print(f'Stock is overvalued by ${delta}. Recomended action: SHORT.')

elif dcf_implied_price > price_tmr:
    delta = dcf_implied_price - price_tmr
    print(f'Stock is undervalued by ${delta}. Recomended action: LONG.')

else:
    print(f'Stock is correctly priced. No recomended trading action.')