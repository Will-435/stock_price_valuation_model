import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""
This function will intake a data frame and remove all rows and columns 
that are duplicates within the data frame. 
Cleaning the data this way improves processing time by cutting out unnecessary work.
"""
def drop_duplicate_rows_cols(df):

    # remove duplicated rows
    df_1 = df.drop_duplicates()

    # remove duplicated columns
    df_2_transpose = df_1.T.drop_duplicates()
    df_2 = df_2_transpose.T

    return df_2


"""
This function inputs a dataframe with not a number (NaN) elements and handles them accordingly.
If NaN elements are below 20%, we replace them with the column mean in order to make the most of our 
data without skewing the results. If the NaN elements are over 20%, their rows are removed.
This is necessary to stop errors later on when performing calculations through built-in functions.
"""
def nan_handling(df):

    columns = df.columns
    for col in columns:

        # creating a boolean matrix, where True represents NaN element
        boolean_values = df[col].isna()
        NaN_sum = boolean_values.sum()
        num_elements = df[col].size

        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df = df.dropna(subset=[col]) 

        elif pd.api.types.is_any_real_numeric_dtype(df[col]) and NaN_sum > num_elements * 0.2:
            # remove all rows with NaN elements
            df = df.dropna(subset=[col])
    
        else:
            # fills that column with its respective median from the df.median() series
            df[col] = df[col].fillna(df[col].median())

    return df


"""
This function takes a dataframe with dates as strings and converts them to timestamp objects.
Iterating over each column is fast. Only if the element fits the condition, a date will be converted.
This is necessary for the date column to be interpreted in our model as numeric, which is essential.
""" 
def convert_dates_to_timestamp(df):

    columns = df.columns
    for col in columns:
        # if the elements of the column are possibly a string 
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            try:
                counted_days = pd.to_datetime(df[col], format = '%Y-%m-%d').map(pd.Timestamp.toordinal)
                df[col] = counted_days

            except Exception:
                pass

    return df


"""
This function takes a data frame with columns of strings and replaces those columns with
a series of boolean columns.
The one hot encoder library assigns each unique string a boolean value.
This is necessary so that our model can associate certain sectors with patterns in our data.
We don't need to consider dates; they have already been turned into integers.
"""
def encode_strings(df):

    cat_cols = []
    columns = df.columns

# Create a list of catagorical columns
    for col in columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):

# Making sure were not turning any numbers stored as strings into boolean columns
            try:
                df[col] = pd.to_numeric(df[col])
            
            except Exception:
                cat_cols.append(col)


    if cat_cols:
 # Define the encoder
        encode = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore').set_output(transform = 'pandas')

# Create a data frame of boolean columns 
        ohe_transform = encode.fit_transform(df[cat_cols])

# Drops the original catagorical columns
        df_num = df.drop(columns=cat_cols)

        df_clean = pd.concat([df_num, ohe_transform], axis=1)
        
    else:
        df_clean = df.copy()

    return df_clean


"""
This function tries different values of max leaf nodes to find the optimum using mean error.
Creating its model ensures we don't test one model once, resulting in overfitting.
This is important for building confidence in our final model.

* train_test_split mixes past and future data - causing leakage.
* We will order the dates and then split them into test, train manually.
"""