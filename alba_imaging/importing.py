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

def import_pickle(pkl_Filepath):
    """Import a pickle file and return the data as a dictionary."""
    with open(pkl_Filepath, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data
