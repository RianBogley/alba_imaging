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
