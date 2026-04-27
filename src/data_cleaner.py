import pandas as pd
from sklearn.preprocessing import OneHotEncoder


NAN_DROP_THRESHOLD = 0.2
DATE_FORMAT = '%Y-%m-%d'


def drop_duplicate_rows_cols(df):
    """Remove duplicated rows and duplicated columns from a dataframe."""
    deduped_rows = df.drop_duplicates()
    deduped_cols = deduped_rows.T.drop_duplicates().T
    return deduped_cols


def nan_handling(df):
    """
    Handle NaN values in a dataframe.

    Inputs:
        df: pandas DataFrame possibly containing NaN values.
    Outputs:
        df: pandas DataFrame with NaN values handled per column.
    Operates per column: string columns drop NaN rows; numeric columns
    drop rows when the NaN proportion exceeds NAN_DROP_THRESHOLD,
    otherwise NaN values are imputed with the column median.
    """
    for col in df.columns:
        nan_count = df[col].isna().sum()
        column_size = df[col].size

        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df = df.dropna(subset=[col])
        elif (pd.api.types.is_any_real_numeric_dtype(df[col])
              and nan_count > column_size * NAN_DROP_THRESHOLD):
            df = df.dropna(subset=[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def convert_dates_to_timestamp(df):
    """Convert any string-formatted date column to its ordinal day count."""
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            try:
                ordinal_days = (
                    pd.to_datetime(df[col], format=DATE_FORMAT)
                    .map(pd.Timestamp.toordinal)
                )
                df[col] = ordinal_days
            except Exception:
                pass

    return df


def encode_strings(df):
    """
    One-hot encode all genuine string columns in the dataframe.

    Inputs:
        df: pandas DataFrame.
    Outputs:
        cleaned_df: dataframe with categorical columns replaced by
        boolean indicator columns produced by OneHotEncoder.
    Numeric columns stored as strings are converted to numeric in place
    rather than being one-hot encoded.
    """
    categorical_cols = []

    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                categorical_cols.append(col)

    if categorical_cols:
        encoder = (
            OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            .set_output(transform='pandas')
        )
        encoded = encoder.fit_transform(df[categorical_cols])
        numeric_only = df.drop(columns=categorical_cols)
        cleaned_df = pd.concat([numeric_only, encoded], axis=1)
    else:
        cleaned_df = df.copy()

    return cleaned_df
