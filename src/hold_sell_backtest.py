"""
Evaluates the forecasting model as a hold or sell engine rather than as a next
day direction predictor. The engine is meant to disagree with the market and
tell a buy and hold investor when to step aside, so per day directional accuracy
is the wrong test. This module instead follows the model's hold and sell signals
as a hold or cash strategy and asks whether it exits before drawdowns and
improves risk adjusted return against simply holding the stock.

The model predicts the next day close from lagged momentum, moving average and
volatility features. A sell signal is raised whenever the predicted next close
sits below the current close, meaning the model expects a decline and an investor
should move to cash. Signals are produced walk forward: the model is only ever
trained on data observable before the day it predicts, so there is no look ahead.
"""


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'apple_stock_data.csv'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
VISUALS_DIR = PROJECT_ROOT / 'visuals'
METRICS_PATH = PROCESSED_DIR / 'hold_sell_metrics.parquet'
EQUITY_PLOT_PATH = VISUALS_DIR / 'hold_sell_equity_curves.png'
DRAWDOWN_PLOT_PATH = VISUALS_DIR / 'hold_sell_drawdowns.png'

WEEK_WINDOW = 5
MONTH_WINDOW = 20

INITIAL_TRAIN_FRACTION = 0.5
REFIT_INTERVAL_DAYS = 21

RF_N_ESTIMATORS = 300
RF_MIN_SAMPLES_LEAF = 2
RANDOM_STATE = 1

TRADING_DAYS_PER_YEAR = 252
FORWARD_RETURN_WINDOW = 5

FIGURE_DPI = 150
EQUITY_FIGSIZE = (11, 6)
DRAWDOWN_FIGSIZE = (11, 4.5)
STRATEGY_COLOUR = "green"
BUY_HOLD_COLOUR = "blue"
DRAWDOWN_COLOUR = "red"

FEATURE_COLUMNS = [
    'yesterday_return', 'return_week', 'return_month',
    'mean_week', 'mean_month', 'vol_week', 'vol_month', 'close_price',
]


def load_price_series(csv_path = None):
    """
    Load the raw Apple price history and return a clean, date sorted frame with
    the close price. The source CSV is the raw external input so it is read as
    CSV rather than Parquet.

    INPUTS:
        * csv_path, path to the raw Apple price CSV

    OUTPUTS:
        * dataframe with a datetime date column and the close price, sorted by date
    """
    raw_frame = pd.read_csv(csv_path)
    raw_frame['date'] = pd.to_datetime(raw_frame['date'], errors = 'coerce')
    raw_frame = raw_frame.dropna(subset = ['date']).sort_values('date').reset_index(drop = True)
    return raw_frame[['date', 'close_price']]


def add_features(price_frame = None):
    """
    Add the lagged momentum, moving average and volatility features used by the
    model, plus the next day close as the prediction target. Every feature is
    shifted by one day so only past information is used, which prevents look
    ahead leakage.

    INPUTS:
        * price_frame, frame with date and close_price

    OUTPUTS:
        * frame with the feature columns, a target_close column and no missing rows
    """
    frame = price_frame.copy()
    close = frame['close_price']

    frame['yesterday_return'] = close.pct_change().shift(1)
    frame['return_week'] = close.pct_change(WEEK_WINDOW).shift(1)
    frame['return_month'] = close.pct_change(MONTH_WINDOW).shift(1)
    frame['mean_week'] = close.rolling(WEEK_WINDOW).mean().shift(1)
    frame['mean_month'] = close.rolling(MONTH_WINDOW).mean().shift(1)
    frame['vol_week'] = close.rolling(WEEK_WINDOW).std().shift(1)
    frame['vol_month'] = close.rolling(MONTH_WINDOW).std().shift(1)

    frame['target_close'] = close.shift(-1)
    frame['next_return'] = close.pct_change().shift(-1)

    return frame.dropna().reset_index(drop = True)


def generate_walk_forward_signals(feature_frame = None):
    """
    Produce a hold or sell signal for every day in the evaluation window by
    predicting the next close walk forward. The model starts trained on the first
    fraction of the history and is refitted at a fixed interval on all data
    observable up to the prediction day. A sell signal is raised when the
    predicted next close is below the current close.

    INPUTS:
        * feature_frame, output of add_features with features and target_close

    OUTPUTS:
        * dataframe over the evaluation window with date, close price, next
          return, the predicted close and a hold flag, where hold is 1 to stay
          invested and 0 to move to cash
    """
    feature_matrix = feature_frame[FEATURE_COLUMNS].values
    target_values = feature_frame['target_close'].values
    close_values = feature_frame['close_price'].values

    row_count = len(feature_frame)
    evaluation_start = int(row_count * INITIAL_TRAIN_FRACTION)

    model = RandomForestRegressor(
        n_estimators = RF_N_ESTIMATORS,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        random_state = RANDOM_STATE,
        n_jobs = -1,
    )

    predicted_closes = np.full(row_count, np.nan)
    days_since_refit = REFIT_INTERVAL_DAYS
    for current_index in range(evaluation_start, row_count):
        if days_since_refit >= REFIT_INTERVAL_DAYS:
            # Train only on rows strictly before the prediction day.
            model.fit(feature_matrix[:current_index], target_values[:current_index])
            days_since_refit = 0
        predicted_closes[current_index] = model.predict(feature_matrix[current_index].reshape(1, -1))[0]
        days_since_refit += 1

    evaluation_slice = feature_frame.iloc[evaluation_start:].copy()
    evaluation_slice['predicted_close'] = predicted_closes[evaluation_start:]
    evaluation_slice['hold'] = (evaluation_slice['predicted_close'] >= evaluation_slice['close_price']).astype(int)
    return evaluation_slice[['date', 'close_price', 'next_return', 'predicted_close', 'hold']].reset_index(drop = True)


def annualised_return(daily_returns = None):
    """Annualise a daily return series by compounding the mean over a trading year."""
    return (1.0 + daily_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1.0


def annualised_volatility(daily_returns = None):
    """Annualise the standard deviation of a daily return series."""
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns = None):
    """Annualised Sharpe ratio of a daily return series against a zero cash rate."""
    if daily_returns.std() == 0:
        return np.nan
    return daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity_curve = None):
    """Largest peak to trough fall of an equity curve, returned as a negative fraction."""
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return drawdown.min()


def run_backtest(signal_frame = None):
    """
    Turn the hold and sell signals into a hold or cash strategy and compare it to
    buy and hold. On a hold day the strategy earns the next day return, on a sell
    day it earns nothing because it sits in cash. Builds both equity curves and
    their drawdown paths.

    INPUTS:
        * signal_frame, output of generate_walk_forward_signals

    OUTPUTS:
        * frame with the strategy and buy and hold daily returns, equity curves
          and drawdown paths
    """
    frame = signal_frame.copy()
    frame['strategy_return'] = frame['hold'] * frame['next_return']
    frame['buy_hold_return'] = frame['next_return']

    frame['strategy_equity'] = (1.0 + frame['strategy_return']).cumprod()
    frame['buy_hold_equity'] = (1.0 + frame['buy_hold_return']).cumprod()

    frame['strategy_drawdown'] = frame['strategy_equity'] / frame['strategy_equity'].cummax() - 1.0
    frame['buy_hold_drawdown'] = frame['buy_hold_equity'] / frame['buy_hold_equity'].cummax() - 1.0
    return frame


def summarise_exit_timing(signal_frame = None):
    """
    Measure whether sell signals actually precede weakness. Compares the average
    forward return after a sell signal with the average after a hold signal, and
    reports the share of sell days that were followed by a fall over the forward
    window. A useful sell engine should show lower, ideally negative, forward
    returns after a sell than after a hold.

    INPUTS:
        * signal_frame, output of generate_walk_forward_signals

    OUTPUTS:
        * dictionary of forward return after sell, forward return after hold and
          the fraction of sell days followed by a decline
    """
    frame = signal_frame.copy()
    forward_return = (
        frame['close_price'].shift(-FORWARD_RETURN_WINDOW) / frame['close_price'] - 1.0
    )
    sell_mask = (frame['hold'] == 0) & forward_return.notna()
    hold_mask = (frame['hold'] == 1) & forward_return.notna()

    forward_after_sell = forward_return[sell_mask].mean()
    forward_after_hold = forward_return[hold_mask].mean()
    sell_decline_rate = (forward_return[sell_mask] < 0).mean()
    return {
        'forward_return_after_sell': float(forward_after_sell),
        'forward_return_after_hold': float(forward_after_hold),
        'sell_days_followed_by_decline': float(sell_decline_rate),
    }


def build_metrics_table(backtest_frame = None, timing_summary = None):
    """
    Collect the headline strategy and buy and hold metrics into a single table
    for saving and printing.

    INPUTS:
        * backtest_frame, output of run_backtest
        * timing_summary, output of summarise_exit_timing

    OUTPUTS:
        * single row dataframe of the comparison metrics
    """
    strategy = backtest_frame['strategy_return']
    buy_hold = backtest_frame['buy_hold_return']
    time_in_market = backtest_frame['hold'].mean()

    record = {
        'evaluation_days': len(backtest_frame),
        'time_in_market': float(time_in_market),
        'strategy_annual_return': float(annualised_return(strategy)),
        'buy_hold_annual_return': float(annualised_return(buy_hold)),
        'strategy_annual_vol': float(annualised_volatility(strategy)),
        'buy_hold_annual_vol': float(annualised_volatility(buy_hold)),
        'strategy_sharpe': float(sharpe_ratio(strategy)),
        'buy_hold_sharpe': float(sharpe_ratio(buy_hold)),
        'strategy_max_drawdown': float(max_drawdown(backtest_frame['strategy_equity'])),
        'buy_hold_max_drawdown': float(max_drawdown(backtest_frame['buy_hold_equity'])),
    }
    record.update(timing_summary)
    return pd.DataFrame([record])


def plot_equity_curves(backtest_frame = None, output_path = None):
    """
    Plot the strategy equity curve against buy and hold so the timing of exits is
    visible.

    INPUTS:
        * backtest_frame, output of run_backtest
        * output_path, file path for the saved figure

    OUTPUTS:
        * none, the figure is written to disk
    """
    figure_handle, axis_handle = plt.subplots(figsize = EQUITY_FIGSIZE)
    axis_handle.plot(backtest_frame['date'], backtest_frame['buy_hold_equity'], color = BUY_HOLD_COLOUR, label = "buy and hold")
    axis_handle.plot(backtest_frame['date'], backtest_frame['strategy_equity'], color = STRATEGY_COLOUR, label = "hold or sell strategy")
    axis_handle.set_title("Hold or sell strategy versus buy and hold")
    axis_handle.set_xlabel("date")
    axis_handle.set_ylabel("growth of one unit")
    axis_handle.legend()
    figure_handle.tight_layout()
    figure_handle.savefig(output_path, dpi = FIGURE_DPI)
    plt.close(figure_handle)


def plot_drawdowns(backtest_frame = None, output_path = None):
    """
    Plot the drawdown path of the strategy against buy and hold to show whether
    the sell signals reduce the depth of losses.

    INPUTS:
        * backtest_frame, output of run_backtest
        * output_path, file path for the saved figure

    OUTPUTS:
        * none, the figure is written to disk
    """
    figure_handle, axis_handle = plt.subplots(figsize = DRAWDOWN_FIGSIZE)
    axis_handle.plot(backtest_frame['date'], backtest_frame['buy_hold_drawdown'], color = BUY_HOLD_COLOUR, label = "buy and hold")
    axis_handle.plot(backtest_frame['date'], backtest_frame['strategy_drawdown'], color = DRAWDOWN_COLOUR, label = "hold or sell strategy")
    axis_handle.set_title("Drawdown path: strategy versus buy and hold")
    axis_handle.set_xlabel("date")
    axis_handle.set_ylabel("drawdown")
    axis_handle.legend()
    figure_handle.tight_layout()
    figure_handle.savefig(output_path, dpi = FIGURE_DPI)
    plt.close(figure_handle)


def main():
    """Run the hold or sell evaluation end to end and report the comparison metrics."""
    PROCESSED_DIR.mkdir(parents = True, exist_ok = True)
    VISUALS_DIR.mkdir(parents = True, exist_ok = True)

    price_frame = load_price_series(csv_path = DATA_PATH)
    feature_frame = add_features(price_frame = price_frame)
    signal_frame = generate_walk_forward_signals(feature_frame = feature_frame)
    backtest_frame = run_backtest(signal_frame = signal_frame)
    timing_summary = summarise_exit_timing(signal_frame = signal_frame)

    metrics_table = build_metrics_table(backtest_frame = backtest_frame, timing_summary = timing_summary)
    metrics_table.to_parquet(METRICS_PATH, index = False)

    plot_equity_curves(backtest_frame = backtest_frame, output_path = EQUITY_PLOT_PATH)
    plot_drawdowns(backtest_frame = backtest_frame, output_path = DRAWDOWN_PLOT_PATH)

    print("hold or sell engine evaluation")
    for column in metrics_table.columns:
        print(f"  {column}: {metrics_table.iloc[0][column]:.4f}")
    print(f"saved metrics to {METRICS_PATH}")
    print(f"saved plots to {EQUITY_PLOT_PATH.name} and {DRAWDOWN_PLOT_PATH.name}")


if __name__ == '__main__':
    main()
