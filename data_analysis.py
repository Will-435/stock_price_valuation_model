from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import randint
import numpy as np
"""
This function tries different values of max leaf nodes to find the optimum using mean error.
Creating its model ensures we don't test one model once, resulting in overfitting.
This is important for building confidence in our final model.

* train_test_split mixes past and future data - causing leakage.
* We will order the dates and then split them into test, train manually.
"""


def random_forest_tuning(train_x, train_y, n_splits = 5, n_iter = 200, random_state = 1):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Reasonable ranges for time-series tabular data
    param_dict = {
        "n_estimators": randint(200, 1200),
        "max_depth": randint(3, 30),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        # keep max_leaf_nodes in play if you like:
        "max_leaf_nodes": randint(10, 10000)
    }

    base = RandomForestRegressor(random_state = random_state, n_jobs = -1)

    search = RandomizedSearchCV(

        estimator = base,
        param_distributions = param_dict,
        n_iter  =n_iter,
        scoring = "neg_mean_absolute_error",
        cv = tscv,
        n_jobs = -1,
        verbose = 1,
        random_state=random_state
    )
    search.fit(train_x, train_y)

    return search.best_estimator_, search.best_params_, -search.best_score_


"""
This function will intake the dataframe and its defined x and y and split the
data frame into training data, and testing data. train_test_split is not appropriate
because it will mix dates in the training data that will lead to leakage and hinder the
models ability to find patterns in a day-to-day sequence.
"""

def retrieve_train_test(df, features, target):

    time_ordered_df = df.sort_values('date')

# Test before train to avoid obvious progression that skew model results
    train = time_ordered_df.iloc[:int(len(time_ordered_df) * 0.9)]
    test = time_ordered_df.iloc[int(len(time_ordered_df) * 0.9):]

    train_x = train[features]
    test_x = test[features]
    train_y = train[target]
    test_y = test[target]

    return train_x, test_x, train_y, test_y


def include_stock_analysis(df):

    df = df.sort_values('date').copy()

# Here we consider the percentage return of the last day, week and month
    df["yday_return"] = df['close_price'].pct_change().shift(1)
    df['return_week'] = df['close_price'].pct_change(5).shift(1)
    df['return_month'] = df['close_price'].pct_change(20).shift(1)

# Here we consider the the mean closing price of the last week and month
    df['mean_week']  = df["close_price"].rolling(5).mean().shift(1)
    df['mean_month'] = df["close_price"].rolling(20).mean().shift(1) 

# Here we consider the standrad deviation of the closing price over the last week and month
    df['vol_week'] = df['close_price'].rolling(5).std().shift(1)
    df['vol_month'] = df['close_price'].rolling(20).std().shift(1)

# Remove the new, unnecessary rows
    df = df.dropna()

    return df

def run_model(df_0, target):

    features = [col for col in df_0.columns if col != target]

    x = df_0[features]
    y = df_0[target]

    price_tday = df_0[target].iloc[-1]

    train_x, test_x, train_y, test_y =  retrieve_train_test(df_0, features, target)

    # This will itterate through different paramater combinations and find the best
    final_model, best_params, cv_mae = random_forest_tuning(train_x, train_y, n_splits=5, n_iter=40)

    final_model.fit(train_x, train_y)

    prediction_y = final_model.predict(test_x)

    mae_prediction = mean_absolute_error(test_y, prediction_y)

    x_tmr = x.tail(1)
    price_tmr = float(final_model.predict(x_tmr))

# math.log() calculates the natural log by default
    drift_1 = np.log( price_tmr / float(price_tday) )

    return  price_tmr, drift_1, mae_prediction