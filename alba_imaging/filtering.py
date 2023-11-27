# %%
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
import pickle
import time
import pandas as pd
import numpy as np

# %%
# ---------------------------------------------------------------------------- #
#                               DEFINE FUNCTIONS                               #
# ---------------------------------------------------------------------------- #

def clean_and_sort(df, col_name, pidn_col):
    """
    Drop any cases that have an invalid or missing value for a given column.
    Re-sort the data by PIDN.
    """
    # Drop any cases that have an invalid or missing value for the dependent variable:
    print(f'{len(df[df[col_name] == ""]) + len(df[df[col_name].isna()])} cases have a missing or NaN value for {col_name}.')
    df = df.drop([row for row in df.index if pd.isna(df.loc[row, col_name])])
    df = df.drop([row for row in df.index if df.loc[row, col_name] == ""])
    # Sort the data by PIDN so the order is guaranteed to be the same moving forward:
    df = df.sort_values(by=[pidn_col])
    df.reset_index(drop=True, inplace=True)
    print(f'{len(df)} cases remain.')
    # Print the value counts for the dependent variable:
    print(f'Value counts for {col_name}:')
    print(df[col_name].value_counts())
    # Return the df:
    return df

def filter_df_criteria(df, dict, policy):
    """
    Filter a dataframe to only include cases that match a list of inclusion or exclusion criteria.
    The criteria are specified by a dictionary where the key is the column name and
    the value is a list of values to include or exclude.
    Example: Can specify a diagnosis column, and then provide a list of diagnoses to include.

    Parameters:
    df = dataframe to filter
    dict = dictionary of column names and corresponding lists of values to include or exclude
    policy = 'include' or 'exclude'
    """
    # For each column name in the dictionary, if the column exists in the df, filter the df.
    # If the policy is 'include', only include cases that match the values in the list.
    # If the policy is 'exclude', exclude cases that match the values in the list.
    for col in dict.keys():
        if col in df.columns:
            if policy == 'include':
                # Print the inclusion criteria:
                print(f'Including the following values for {col}:')
                print(dict[col])
                # Print how many cases are being excluded:
                print(f'{len(df)} cases before filtering for {col}.')
                df = df[df[col].isin(dict[col])]
                print(f'{len(df)} cases remain after filtering for {col}.')
            elif policy == 'exclude':
                # Print the exclusion criteria:
                print(f'Excluding the following values for {col}:')
                print(dict[col])
                # Print how many cases are being excluded:
                print(f'{len(df)} cases before filtering for {col}.')
                df = df[~df[col].isin(dict[col])]
                print(f'{len(df)} cases remain after filtering for {col}.')
        else:
            print(f'WARNING: {col} does not exist in the dataframe.')
    # Return the df:
    return df