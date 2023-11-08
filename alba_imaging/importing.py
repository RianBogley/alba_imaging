# %%
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
import pickle

# %%
# ---------------------------------------------------------------------------- #
#                               DEFINE FUNCTIONS                               #
# ---------------------------------------------------------------------------- #

def import_pickle(pkl_Filepath):
    """Import a pickle file and return the data as a dictionary."""
    with open(pkl_Filepath, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

def clean_and_sort(df, dep_var, pidn_col):
    """
    Drop any cases that have an invalid or missing value for the dependent variable.
    Re-sort the data by PIDN.
    """
    # Drop any cases that have an invalid or missing value for the dependent variable:
    print(f'{len(df[df[dep_var] == ""]) + len(df[df[dep_var].isna()])} cases have a missing or NaN value for {dep_var}.')
    df = df.drop([row for row in df.index if pd.isna(df.loc[row, dep_var])])
    df = df.drop([row for row in df.index if df.loc[row, dep_var] == ""])
    # Sort the data by PIDN so the order is guaranteed to be the same moving forward:
    df = df.sort_values(by=[pidn_col])
    df.reset_index(drop=True, inplace=True)
    print(f'{len(df)} cases remain.')
    # Print the value counts for the dependent variable:
    print(f'Value counts for {dep_var}:')
    print(df[dep_var].value_counts())
    # Return the df:
    return df
