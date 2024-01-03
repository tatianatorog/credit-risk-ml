# One hot encoding consists in replacing the categorical variable by a group of binary
# variables which take value 0 or 1, to indicate if a certain category is present in
# an observation. Each one of the binary variables are also known as dummy variables.

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def encode_categorical(df_train, df_val, df_test):
    # Identify binary and multi-category columns
    binary_cols = [
        col
        for col in df_train.columns
        if df_train[col].dtype == "object" and df_train[col].nunique() == 2
    ]
    multi_cat_cols = [
        col
        for col in df_train.columns
        if df_train[col].dtype == "object" and df_train[col].nunique() > 2
    ]

    # Binary encoding
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(df_train[binary_cols])

    df_train[binary_cols] = ordinal_encoder.transform(df_train[binary_cols])
    df_val[binary_cols] = ordinal_encoder.transform(df_val[binary_cols])
    df_test[binary_cols] = ordinal_encoder.transform(df_test[binary_cols])

    # One hot encoding
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_encoder.fit(df_train[multi_cat_cols])

    one_hot_cols_train = one_hot_encoder.transform(df_train[multi_cat_cols])
    one_hot_cols_val = one_hot_encoder.transform(df_val[multi_cat_cols])
    one_hot_cols_test = one_hot_encoder.transform(df_test[multi_cat_cols])

    # Convert sparse matrices to dense matrices
    # Transform all datasets using the fitted encoder
    train_onehot_df = pd.DataFrame(
        one_hot_cols_train,
        columns=one_hot_encoder.get_feature_names_out(multi_cat_cols).astype(str),
        index=df_train.index,
    )
    val_onehot_df = pd.DataFrame(
        one_hot_cols_val,
        columns=one_hot_encoder.get_feature_names_out(multi_cat_cols).astype(str),
        index=df_val.index,
    )
    test_onehot_df = pd.DataFrame(
        one_hot_cols_test,
        columns=one_hot_encoder.get_feature_names_out(multi_cat_cols).astype(str),
        index=df_test.index,
    )

    # Drop the original multi-category columns
    df_train.drop(multi_cat_cols, axis=1, inplace=True)
    df_val.drop(multi_cat_cols, axis=1, inplace=True)
    df_test.drop(multi_cat_cols, axis=1, inplace=True)

    # Concatenate the encoded columns to the respective datasets
    df_train = pd.concat([df_train, train_onehot_df], axis=1)
    df_val = pd.concat([df_val, val_onehot_df], axis=1)
    df_test = pd.concat([df_test, test_onehot_df], axis=1)

    return df_train, df_val, df_test


def impute_missing(df_train, df_val, df_test):
    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    df_train[:] = imp.fit_transform(df_train)
    df_val[:] = imp.transform(df_val)
    df_test[:] = imp.transform(df_test)

    return df_train, df_val, df_test


def scale_features(df_train, df_val, df_test):
    scaler = MinMaxScaler()
    df_train[:] = scaler.fit_transform(df_train)
    df_val[:] = scaler.transform(df_val)
    df_test[:] = scaler.transform(df_test)

    return df_train, df_val, df_test


def correct_anomalies(df):
    df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    return df


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct anomalies in DAYS_EMPLOYED column
    working_train_df = correct_anomalies(working_train_df)
    working_val_df = correct_anomalies(working_val_df)
    working_test_df = correct_anomalies(working_test_df)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    # Encoding Categorical Features
    working_train_df, working_val_df, working_test_df = encode_categorical(
        working_train_df, working_val_df, working_test_df
    )
    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    # Imputing Missing Values
    working_train_df, working_val_df, working_test_df = impute_missing(
        working_train_df, working_val_df, working_test_df
    )

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    # Feature Scaling
    working_train_df, working_val_df, working_test_df = scale_features(
        working_train_df, working_val_df, working_test_df
    )

    return (
        working_train_df.values,
        working_val_df.values,
        working_test_df.values,
    )
