import pandas as pd


def calculate_dcf(revenues, ebitda_margins, tax_rate, d_a_percent_revenue, nwc_percent_revenue, capex_percent_revenue, wacc, terminal_growth_rate, net_debt, shares_outstanding):

    
    """
    Calculates the enterprise value and implied share price using a DCF model.
    """

    forecast_years = len(revenues)

    data = {
        'year': range(1, forecast_years + 1),
        'revenue': revenues,
        'ebitda_margin': ebitda_margins
        }

    df = pd.DataFrame(data)

    # Step 1: Calculate Unlevered Free Cash Flow (UFCF) for the forecast period
    df['ebitda'] = df['revenue'] * df['ebitda_margin']
    df['d_and_a'] = df['revenue'] * d_a_percent_revenue
    df['ebit'] = df['ebitda'] - df['d_and_a']
    df['nopat'] = df['ebit'] * (1 - tax_rate)  # Net Operating Profit After Tax
    df['capex'] = df['revenue'] * capex_percent_revenue
    df['nwc'] = df['revenue'] * nwc_percent_revenue
    df['change_in_nwc'] = df['nwc'].diff().fillna(df['nwc'].iloc[0]) # Handle the first year

    df['fcff'] = df['nopat'] + df['d_and_a'] - df['capex'] - df['change_in_nwc']

    # Step 2: Calculate Terminal Value
    last_year_fcff = df['fcff'].iloc[-1]
    terminal_value = last_year_fcff * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

    # Step 3: Discount all cash flows to their present value
    df['discount_factor'] = [(1 + wacc) ** year for year in df['year']]
    df['pv_fcff'] = df['fcff'] / df['discount_factor']

    pv_terminal_value = terminal_value / ((1 + wacc) ** forecast_years)
    
    # Step 4: Calculate Enterprise Value and Equity Value
    enterprise_value = df['pv_fcff'].sum() + pv_terminal_value
    equity_value = enterprise_value - net_debt
    implied_share_price = equity_value / shares_outstanding

    return df, enterprise_value, equity_value, implied_share_price