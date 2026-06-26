import pandas as pd


def calculate_dcf(revenues,
                  ebitda_margins,
                  tax_rate,
                  d_a_percent_revenue,
                  nwc_percent_revenue,
                  capex_percent_revenue,
                  wacc,
                  terminal_growth_rate,
                  net_debt,
                  shares_outstanding):
    """
    Run a discounted cash flow valuation to recover the implied share price.
    Free cash flow to the firm is forecast for each year, a terminal value
    captures the period beyond the forecast, both are discounted at the
    weighted average cost of capital, and net debt is removed to reach equity.

    INPUTS:
        * revenues, an iterable of forecast revenues
        * ebitda_margins, an iterable of forecast EBITDA margins
        * tax_rate, d_a_percent_revenue, nwc_percent_revenue, capex_percent_revenue, scalar assumption rates
        * wacc, the weighted average cost of capital
        * terminal_growth_rate, the perpetual growth rate beyond the forecast
        * net_debt, total net debt subtracted from enterprise value
        * shares_outstanding, the share count used for the per share price

    OUTPUTS:
        * the per year working dataframe
        * the enterprise value
        * the equity value
        * the implied share price
    """
    forecast_years = len(revenues)

    df = pd.DataFrame({
        'year': range(1, forecast_years + 1),
        'revenue': revenues,
        'ebitda_margin': ebitda_margins,
    })

    df['ebitda'] = df['revenue'] * df['ebitda_margin']
    df['d_and_a'] = df['revenue'] * d_a_percent_revenue
    df['ebit'] = df['ebitda'] - df['d_and_a']
    df['nopat'] = df['ebit'] * (1 - tax_rate)
    df['capex'] = df['revenue'] * capex_percent_revenue
    df['nwc'] = df['revenue'] * nwc_percent_revenue
    df['change_in_nwc'] = df['nwc'].diff().fillna(df['nwc'].iloc[0])
    df['fcff'] = df['nopat'] + df['d_and_a'] - df['capex'] - df['change_in_nwc']

    final_year_fcff = df['fcff'].iloc[-1]
    terminal_value = (
        final_year_fcff * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
    )

    df['discount_factor'] = [(1 + wacc) ** year for year in df['year']]
    df['pv_fcff'] = df['fcff'] / df['discount_factor']

    pv_terminal_value = terminal_value / ((1 + wacc) ** forecast_years)

    enterprise_value = df['pv_fcff'].sum() + pv_terminal_value
    equity_value = enterprise_value - net_debt
    implied_share_price = equity_value / shares_outstanding

    return df, enterprise_value, equity_value, implied_share_price
