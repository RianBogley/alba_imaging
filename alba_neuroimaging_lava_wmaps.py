# %% Neuroimaging Files Tools by Rian Bogley ##################################
###############################################################################
# %% PACKAGES #################################################################
# Import Packages
import os
import pandas as pd
import shutil
import glob
# Libraries
from nilearn.maskers import NiftiMasker
import numpy as np
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker
from nilearn.image import get_data
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from nilearn.decoding import DecoderRegressor
from nilearn import plotting
from nilearn.plotting import plot_stat_map, show
import nilearn.datasets
from joblib import Parallel, delayed
import pickle
from scipy import stats
###############################################################################
# %% GLOBAL VARIABLES #########################################################

###############################################################################
# %% FUNCTIONS ################################################################

# Copy Standard LAVA W-Maps from R: Drive to Specified Directory:
def copy_lava_wmaps(df, 
                    pidn_col_name='PIDN',
                    dcdate_col_name='DCDate',
                    wmaps_dir='L:/language/rbogley/LAVA_W-Maps/', 
                    r_drive='R:/projects/knect/images/wmaps/spmvbm12/'):
    """
    Copy Standard LAVA W-Maps from R: Drive to Specified Directory.
    Requires dataframe with PIDN column (and optional DCDate column for specific scans).
    Requires output directory.
    Requires connection to R: Drive W-Maps folder:
        Default is 'R:/projects/knect/images/wmaps/spmvbm12/'
    """
    # Check if connected to the R drive by testing R drive filepath:
    if os.path.exists(r_drive):
        print('R-Drive connection succesful.')
    else:
        print('R-Drive not connected. Please check your connection.')
        return

    # Import dataframe PIDN column and DCDate column (if exists):
    if pidn_col_name in df.columns and dcdate_col_name in df.columns:
        print('Valid PIDN and DCDate columns found.')
        df = df[[pidn_col_name, dcdate_col_name]]
        df = df.rename(columns={pidn_col_name: 'PIDN', dcdate_col_name: 'DCDate'})

    # TODO: ADD OPTION TO COPY ALL WMAPS IF NO DCDATE COLUMN EXISTS
    # BY FINDING ALL DC DATES FOR EACH CASE WITH PROVIDED PIDN AND COPYING ALL
    else:
        print('No valid PIDN and/or DCDate columns found. Please check your dataframe.')
        return
    
    # Print how many cases exist in the dataframe:
    print(f'{len(df)} cases found in dataframe.')
    # Print how many unique PIDN exist in the dataframe:
    print(f'{len(df["PIDN"].unique())} unique PIDN found in dataframe.')
    # Count how many cases have a missing or NaN DCDate:
    missing_dc_date = len(df[df['DCDate'] == '']) + len(df[df['DCDate'].isna()])
    print(f'{missing_dc_date} cases have a missing or NaN DCDates.')
    # Remove any rows with a blank or NaN DCDate:
    df = df[df['DCDate'] != '']
    df = df.dropna(subset=['DCDate'])
    
    print(f'Attempting to copy W-Maps for {len(df)} cases...')
    # Create a new column with each PIDN rounded down to the nearest 1000:
    df['PIDN_rounded'] = df['PIDN'].apply(round_down_pidn)

    # Create a new column with the original LAVA-generated wmap filepaths:
    df['wmap_source'] = ''
    for index, row in df.iterrows():
        df.at[index, 'wmap_source'] = get_lava_wmap_filename(row['PIDN'], row['DCDate'].strftime('%Y-%m-%d'), r_drive=r_drive)
    # Check if the original wmaps exist:
    for index, row in df.iterrows():
        # if wmap_source is blank, remove the row:
        if row['wmap_source'] == '':
            print(f"PIDN {row['PIDN']} DCDate {row['DCDate'].strftime('%Y-%m-%d')} does not exist in R Drive. Skipping case.")
            # df = df.drop(index)
    # Reset index:
    df = df.reset_index(drop=True)

    # Create a new column with the new wmap filepaths:
    df['wmap'] = ''
    for index, row in df.iterrows():
        df.at[index, 'wmap'] = new_wmap_filename(row['PIDN'], row['DCDate'].strftime('%Y-%m-%d'), wmaps_dir=wmaps_dir)
    # Check if the new wmaps already exist:
    for index, row in df.iterrows():
        if os.path.exists(row['wmap']):
            print(f"PIDN {row['PIDN']} DCDate {row['DCDate'].strftime('%Y-%m-%d')} already exists in output directory. Skipping case.")
            # df = df.drop(index)
    # Reset index:
    df = df.reset_index(drop=True)

    print(f'Copying the remaining {len(df)} W-Maps to output directory...')
    # Copy any remaining wmaps from the R Drive to the output directory:
    for index, row in df.iterrows():
        shutil.copy(row['wmap_source'], row['wmap'])
    # Return df as a new df:
    # Save df as a csv in output directory:
    df.to_csv(f'{wmaps_dir}/wmaps.csv', index=False)
    return df

# Round down PIDN to nearest 1000 value, for use in R Drive/I Drive filepaths:
def round_down_pidn(PIDN):
    """
    Round down PIDN to nearest 1000 value, for use in R Drive/I Drive filepaths.
    e.g. 12345 -> 12000
    e.g. 999 -> 0
    """
    return int(PIDN/1000)*1000

# Get Standard LAVA W-Map Filepath & Filename in R-Drive:
def get_lava_wmap_filename(PIDN, DCDate, r_drive):
    """
    Get Standard LAVA W-Map Filepath and Filename in R-Drive
    """
    # Create the parent wmap filepath in the R Drive using PIDN and DCDate:
    filepath = f"{r_drive}/{round_down_pidn(PIDN)}/{PIDN}/{DCDate.strftime('%Y-%m-%d')}/"
    # If the filepath exists, check for any subdirectories:
    if os.path.exists(filepath) and len(os.listdir(filepath)) > 0:
        filepath = f"{filepath}{os.listdir(filepath)[0]}/wmap.nii.gz"
        # If the wmap file exists, return it:
        if os.path.exists(filepath):
            return filepath

# Define New W-Map Filepath & Filename:
def new_wmap_filename(PIDN, DCDate, wmaps_dir):
    """
    Define New W-Map Filepath & Filename
    """
    # Return a joined filepath and filename:
    # New filename is PIDN_DCDATE.nii.gz
    filename = f'{PIDN}_{DCDate.strftime("%Y-%m-%d")}_wmap.nii.gz'
    return os.path.join(wmaps_dir, filename)

# Get New, Copied W-Map Filepaths & Filenames:
def get_new_wmap_filenames(df, 
                    pidn_col_name='PIDN',
                    dcdate_col_name='DCDate',
                    wmaps_dir='L:/language/rbogley/LAVA_W-Maps/'):
    """
    Get Copied W-Map Filepaths & Filenames
    """
    # Create a new column with the new wmap filepaths:
    df['w-map'] = ''
    df['invalid_w-map'] = ''

    # If PIDN and DCDate don't exist, return the df and an error:
    if pidn_col_name not in df.columns or dcdate_col_name not in df.columns:
        print('Valid PIDN and DCDate columns not found in dataframe as specified.')
        return df
    else:
        for index, row in df.iterrows():
            # Check if case has a valid DCDate (not blank or NaN):
            if row[dcdate_col_name] != '' and not pd.isna(row[dcdate_col_name]):
                # Get the new wmap filename:
                df.at[index, 'w-map'] = new_wmap_filename(row[pidn_col_name], row[dcdate_col_name], wmaps_dir=wmaps_dir)
        # Print how many cases have a potential w-map filepath:
        print(f'{len(df[df["w-map"] != ""])} cases have potential w-maps.')
        for index, row in df.iterrows():
            # Check if the new wmap file exists:
            if not os.path.exists(row['w-map']):
                # If not, move the filepath to the invalid_w-map column:
                df.at[index, 'invalid_w-map'] = row['w-map']
                # and remove the filepath from the w-map column:
                df.at[index, 'w-map'] = ''

    # Print how many cases have a w-map filepath:
    print(f'{len(df[df["w-map"] != ""])} cases have a valid w-map filepaths.')
    # Return the df:
    return df
