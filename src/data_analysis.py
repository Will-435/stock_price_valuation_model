from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import randint
import numpy as np


TRAIN_FRACTION = 0.9
WEEK_WINDOW = 5
MONTH_WINDOW = 20
DEFAULT_CV_SPLITS = 5
DEFAULT_SEARCH_ITERATIONS = 200
DEFAULT_RANDOM_STATE = 1
TUNING_ITERATIONS = 40

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(200, 1200),
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "max_leaf_nodes": randint(10, 10000),
}


def random_forest_tuning(train_x, train_y, n_splits = DEFAULT_CV_SPLITS, n_iter = DEFAULT_SEARCH_ITERATIONS, random_state = DEFAULT_RANDOM_STATE):
    """
    Tune a random forest regressor with a randomised hyperparameter search.
    Cross validation uses a time series split so future data never leaks into
    the training folds.

    INPUTS:
        * train_x, train_y, the training features and target
        * n_splits, the number of time series cross validation folds
        * n_iter, the number of random parameter combinations to try
        * random_state, the seed for reproducibility

    OUTPUTS:
        * the fitted best estimator
        * the dictionary of best hyperparameters
        * the best cross validated mean absolute error
    """
    time_series_cv = TimeSeriesSplit(n_splits = n_splits)
    base_estimator = RandomForestRegressor(random_state = random_state, n_jobs = -1)

    search = RandomizedSearchCV(
        estimator = base_estimator,
        param_distributions = PARAM_DISTRIBUTIONS,
        n_iter = n_iter,
        scoring = "neg_mean_absolute_error",
        cv = time_series_cv,
        n_jobs = -1,
        verbose = 1,
        random_state = random_state,
    )
    search.fit(train_x, train_y)

    return search.best_estimator_, search.best_params_, -search.best_score_


def retrieve_train_test(df, features, target):
    """Split a time ordered dataframe into chronological train and test sets."""
    time_ordered_df = df.sort_values('date')
    split_index = int(len(time_ordered_df) * TRAIN_FRACTION)

    train = time_ordered_df.iloc[:split_index]
    test = time_ordered_df.iloc[split_index:]

    return train[features], test[features], train[target], test[target]


def include_stock_analysis(df):
    """Add lagged returns, rolling means and rolling volatilities, all shifted to use past data only."""
    df = df.sort_values('date').copy()

    df["yday_return"] = df['close_price'].pct_change().shift(1)
    df['return_week'] = df['close_price'].pct_change(WEEK_WINDOW).shift(1)
    df['return_month'] = df['close_price'].pct_change(MONTH_WINDOW).shift(1)

    df['mean_week'] = df['close_price'].rolling(WEEK_WINDOW).mean().shift(1)
    df['mean_month'] = df['close_price'].rolling(MONTH_WINDOW).mean().shift(1)

    df['vol_week'] = df['close_price'].rolling(WEEK_WINDOW).std().shift(1)
    df['vol_month'] = df['close_price'].rolling(MONTH_WINDOW).std().shift(1)

    return df.dropna()


def run_model(df, target):
    """
    Train and evaluate the tuned random forest, then predict the next day
    price and the implied one day drift.

    INPUTS:
        * df, the cleaned and feature engineered dataframe
        * target, the name of the target column

    OUTPUTS:
        * the predicted next day price
        * the log return between today's price and the prediction
        * the mean absolute error on the held out test set
    """
    features = [col for col in df.columns if col != target]
    feature_frame = df[features]

    price_today = df[target].iloc[-1]

    train_x, test_x, train_y, test_y = retrieve_train_test(df, features, target)

    final_model, _, _ = random_forest_tuning(
        train_x, train_y,
        n_splits = DEFAULT_CV_SPLITS,
        n_iter = TUNING_ITERATIONS,
    )
    final_model.fit(train_x, train_y)

    test_predictions = final_model.predict(test_x)
    out_of_sample_mae = mean_absolute_error(test_y, test_predictions)

    latest_features = feature_frame.tail(1)
    price_tomorrow = float(final_model.predict(latest_features)[0])

    drift_one_day = np.log(price_tomorrow / float(price_today))

    return price_tomorrow, drift_one_day, out_of_sample_mae
