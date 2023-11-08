# %%
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
import os
import shutil
import pandas as pd
import datetime
import nibabel as nib
import numpy as np

# %%
# ---------------------------------------------------------------------------- #
#                               DEFINE FUNCTIONS                               #
# ---------------------------------------------------------------------------- #

# Create a Wynton job file to run Python scripts in a virtual environment:
def create_wynton_job_file(local_dir,
                           wynton_venv_dir,
                           python_mnt_filepath,
                           jobfile_name):
    """
    This script creates a shell script file for running a specified python file
    on Wynton, using a virtual environment. 
    Note: Virtual environment needs to be set-up separately.

    local_dir = Where to save the job file to then be transferred to Wynton.
    wynton_venv_dir = Path to the virtual environment created on Wynton.
    python_mnt_filepath = Path to the python file to be run on Wynton.
    jobfile_name = Name of the Wynton job file to be created.
    """

    job_file_script_text = f"""#!/bin/bash                        #-- Select interpreter
#                                  #-- Any line that starts with #$ is an instruction to SGE
#$ -S /bin/bash                    #-- the shell for the job
#$ -o $HOME                        #-- output directory (fill in)
#$ -e $HOME                        #-- error directory (fill in)
#$ -cwd                            #-- tell the job that it should start in your working directory
#$ -pe smp 12                      #-- Specify parallel environment and number of slots, required
#$ -r y                            #-- tell the system that if a job crashes, it should be restarted
##$ -j y                           #-- tell the system that the STDERR and STDOUT should be joined
#$ -l mem_free=8G                  #-- submits on nodes with enough free memory (required)
##$ -l arch=linux-x64              #-- SGE resources (CPU type)
##$ -l /wynton/scratch/=1G         #-- SGE resources (home and scratch disks)
#$ -l h_rt=336:00:00               #-- runtime limit (see above; this requests 24 hours) #normal is 336:00:00
##$ -t 1-10                        #-- remove first '#' to specify the number of
#$ -l eth_speed=10                 #-- tasks if desired (see Tips section)
#$ -l scratch=20G

#tasks=(0 1bac 2xyz 3ijk 4abc 5def 6ghi 7jkl 8mno 9pqr 1stu )
#input="${{tasks[$SGE_TASK_ID]}}"

#
# System info
date
hostname
hostrun=`/usr/bin/hostname`
qstat -j $JOB_ID                   # This is useful for debugging and usage purposes,
echo "host node: $hostrun"         # e.g. "did my job exceed its memory request?"
id
#
#

export MAC=/mnt/production/imaging-core/new_arch/
module load matlab/2018b
export MATLAB=/wynton/home/opt/matlab/2018b/
export OMP_NUM_THREADS=${{NSLOTS:-1}}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export TMPDIR=/mnt/production/tmp/
export HOSTNAME

cd {wynton_venv_dir}
. bin/activate

python3 {python_mnt_filepath}
"""
    with open(os.path.join(local_dir, jobfile_name), 'w') as f:
        f.write(job_file_script_text)

# Round down PIDN to nearest 1000 value, for use in R Drive/I Drive filepaths:
def round_down_pidn(PIDN):
    """
    Round down PIDN to nearest 1000 value.
    Useful for R Drive/I Drive imaging filepath structures.
    e.g. 12345 -> 12000
    e.g. 999 -> 0
    """
    # If PIDN is a string, convert to int:
    if type(PIDN) == str:
        PIDN = int(PIDN)
    return int(PIDN/1000)*1000

# Get standard LAVA W-Map filepath & filename in R-Drive:
def lava_wmap_filepath(PIDN, DCDate, rdrive_dir):
    """
    Get Standard LAVA W-Map Filepath and Filename in R-Drive
    """
    # DCDate is a not a string, convert to string:
    if type(DCDate) == datetime.date:
        DCDate = DCDate.strftime('%Y-%m-%d')
    elif type(DCDate) == pd.Timestamp:
        DCDate = DCDate.strftime('%Y-%m-%d')

    # Create the parent wmap filepath in the R Drive using PIDN and DCDate:
    # filepath = f"{rdrive_dir}/{round_down_pidn(PIDN)}/{PIDN}/{DCDate.strftime('%Y-%m-%d')}/"
    filepath = os.path.join(rdrive_dir, str(round_down_pidn(PIDN)), str(PIDN), DCDate)

    # If the filepath exists, check for any subdirectories:
    if os.path.exists(filepath) and len(os.listdir(filepath)) > 0:
        # Check if any subdirectories begin with "T1", if so, use that subdir,
        # if not, use the first subdir in the directory:
        if any([x.startswith('MP-LAS') for x in os.listdir(filepath)]):
            subdir = [x for x in os.listdir(filepath) if x.startswith('MP-LAS')][0]
        elif any([x.startswith('T1') for x in os.listdir(filepath)]):
            subdir = [x for x in os.listdir(filepath) if x.startswith('T1')][0]
        else:
            subdir = os.listdir(filepath)[0]
        filepath = os.path.join(filepath, subdir, 'wmap.nii.gz')
        filepath = os.path.normpath(filepath)
        # If the wmap file exists, return it:
        if os.path.exists(filepath):
            return filepath
        
# Define new W-Map filepath & filename:
def new_wmap_filepath(PIDN, DCDate, wmaps_dir):
    """
    Define New W-Map Filepath & Filename
    """
    # Return a joined filepath and filename:
    # New filename is PIDN_DCDATE.nii.gz
    # If DCDate is a date instead of a string, convert to string:
    if type(DCDate) == datetime.date:
        DCDate = DCDate.strftime('%Y-%m-%d')
    elif type(DCDate) == str:
        pass
    filename = f'{PIDN}_{DCDate}_wmap.nii.gz'
    filepath = os.path.join(wmaps_dir, filename)
    filepath = os.path.normpath(filepath)
    return filepath

# Add R-Drive LAVA W-Map filepaths to dataframe:
def get_lava_wmap_filepaths(df,
                            pidn_col='PIDN',
                            dcdate_col='DCDate',
                            wmaps_dir='R:/projects/knect/images/wmaps/spmvbm12/',
                            wmaps_dir_mgt='/shared/macdata/projects/knect/images/wmaps/spmvbm12/'):
    """
    Add LAVA W-Map Filepaths to DataFrame
    """
    # Create a new column with the original LAVA-generated wmap filepaths:
    df['wmap_lava'] = ''
    df['invalid_wmap_lava'] = ''

    # If PIDN and DCDate don't exist, return the df and an error:
    if pidn_col not in df.columns or dcdate_col not in df.columns:
        print('Valid PIDN and DCDate columns not found in dataframe as specified.')
    else:
        for index, row in df.iterrows():
            # Check if case has a valid DCDate (not blank or NaN):
            if row[dcdate_col] != '' and not pd.isna(row[dcdate_col]):
                # Get the new wmap filename:
                df.at[index, 'wmap_lava'] = lava_wmap_filepath(row[pidn_col], row[dcdate_col], rdrive_dir=wmaps_dir)
        for index, row in df.iterrows():
            # Check if wmap file exists on the R Drive:
            if row['wmap_lava'] == None or pd.isna(row['wmap_lava']) or row['wmap_lava'] == '' or not os.path.exists(row['wmap_lava']):
                # If so, move the filepath to the invalid_w-map column:
                df.at[index, 'invalid_wmap_lava'] = row['wmap_lava']
                # and remove the filepath from the w-map column, replace with NaN:
                df.at[index, 'wmap_lava'] = np.nan

    # Print how many cases have and do not have valid w-maps:
    print(f'{len(df[df["wmap_lava"] != ""])} cases have valid w-maps.')
    print(f'{len(df[df["wmap_lava"] == ""])} cases do not have valid w-maps.')
    
    # NOTE: TEST FOR MGT FILEPATH VERSION:
    # Make a copy of the wmap_lava column called wmap_lava_mgt and replace wmaps_dir part of each filepath with wmaps_dir_mgt:
    df['wmap_lava_mgt'] = df['wmap_lava'].copy()
    df['wmap_lava_mgt'] = df['wmap_lava_mgt'].str.replace(wmaps_dir, wmaps_dir_mgt)

    # Return the df:
    return df
            
# Add new W-Map filepaths & filenames to dataframe:
def get_new_wmap_filepaths(df, 
                    pidn_col='PIDN',
                    dcdate_col='DCDate',
                    wmaps_dir='L:/language/rbogley/LAVA_W-Maps/'):
    """
    Get Copied W-Map Filepaths & Filenames
    """
    # Create a new column with the new wmap filepaths:
    df['wmap'] = ''
    df['invalid_wmap'] = ''

    # If PIDN and DCDate don't exist, return the df and an error:
    if pidn_col not in df.columns or dcdate_col not in df.columns:
        print('Valid PIDN and DCDate columns not found in dataframe as specified.')
        return df
    else:
        for index, row in df.iterrows():
            # Check if case has a valid DCDate (not blank or NaN):
            if row[dcdate_col] != '' and not pd.isna(row[dcdate_col]):
                # Get the new wmap filename:
                df.at[index, 'wmap'] = new_wmap_filepath(row[pidn_col], row[dcdate_col], wmaps_dir=wmaps_dir)
        # Print how many cases have a potential W-Map filepath:
        print(f'{len(df[df["wmap"] != ""])} cases have potential w-maps.')
        for index, row in df.iterrows():
            # Check if the new wmap file exists:
            if not os.path.exists(row['wmap']):
                # If not, move the filepath to the invalid_w-map column:
                df.at[index, 'invalid_wmap'] = row['wmap']
                # and remove the filepath from the w-map column:
                df.at[index, 'wmap'] = ''

    # Print how many cases have a w-map filepath:
    print(f'{len(df[df["wmap"] != ""])} cases have a valid w-map filepath.')
    # Return the df:
    return df

# NOTE: CHECK THIS FUNCTION
def clear_invalid_wmaps(df, wmap_col='wmap_lava'):
    """
    Clear Invalid W-Maps from DataFrame
    """
    # Remove all rows from df that have no valid W-Map filepath (blank or NaN)
    print(f'Removing {len(df[df[wmap_col] == ""])} cases with no valid W-Map filepath...')
    df = df.drop([row for row in df.index if df.loc[row, wmap_col] == ""])
    df.reset_index(drop=True, inplace=True)
    print(f'{len(df)} cases remain.')
    return df

# Clean-up dataframe W-Map list:
def clean_df_wmaps(df, pidn_col, dcdate_col, wmaps_dir):
    """
    Add W-Map filepaths to the DataFrame and perform necessary cleaning.
    Returns the cleaned DataFrame.
    """
    # # Add W-Map filepaths to the df from the specified W-Maps directory:
    # df = get_new_wmap_filepaths(df=df, pidn_col=pidn_col, dcdate_col=dcdate_col, wmaps_dir=wmaps_dir)

    # Identify which cases have invalid W-Map filepaths:
    df_invalid_wmaps = df[[pidn_col, dcdate_col, 'invalid_wmap']]
    # Remove all rows in df_invalid_wmaps that do not have a unique value in the invalid_w-map column:
    df_invalid_wmaps = df_invalid_wmaps.drop([row for row in df_invalid_wmaps.index if df_invalid_wmaps.loc[row, 'invalid_wmap'] == ""])
    df_invalid_wmaps.reset_index(drop=True, inplace=True)
    print(f'{len(df_invalid_wmaps)} cases have invalid W-Map filepaths.')
    display(df_invalid_wmaps)

    # Remove all rows from df that have no valid W-Map filepath (blank or NaN)
    print(f'Removing {len(df[df["wmap"] == ""])} cases with no valid W-Map filepath...')
    df = df.drop([row for row in df.index if df.loc[row, 'wmap'] == ""])
    df.reset_index(drop=True, inplace=True)
    print(f'{len(df)} cases remain.')

    return df



# Copy standard LAVA W-Maps from R Drive to a specified directory:
def copy_lava_wmaps(df, 
                    pidn_col='PIDN',
                    dcdate_col='DCDate',
                    wmaps_dir='/mgt/language/master_imaging/LAVA_WMAPS/', 
                    rdrive_dir='/shared/macdata/projects/knect/images/wmaps/spmvbm12/'):
    """
    Copy standard Imaging Core-generated W-Maps from the R: Drive to a specified directory.
    Requires dataframe with PIDN column (and optional DCDate column for specific scans).
    Requires output directory to copy the W-Maps to.
    Requires connection to R: Drive W-Maps folder:
        Default is 'R:/projects/knect/images/wmaps/spmvbm12/'
    """
    # Check if connected to the R drive by testing R drive filepath:
    if os.path.exists(rdrive_dir):
        print('R-Drive connection succesful.')
    else:
        print('R-Drive not connected. Please check your connection.')
        return

    # Import dataframe PIDN column and DCDate column (if exists):
    if pidn_col in df.columns and dcdate_col in df.columns:
        print('Valid PIDN and DCDate columns found.')
        df = df[[pidn_col, dcdate_col]]
        df = df.rename(columns={pidn_col: 'PIDN', dcdate_col: 'DCDate'})

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
        df.at[index, 'wmap_source'] = get_lava_wmap_filepaths(row['PIDN'], row['DCDate'].strftime('%Y-%m-%d'), rdrive_dir=rdrive_dir)
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
        df.at[index, 'wmap'] = new_wmap_filepath(row['PIDN'], row['DCDate'].strftime('%Y-%m-%d'), wmaps_dir=wmaps_dir)
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






# Load in W-Maps using nibabel:
def load_wmaps(df, wmaps_col):
    """
    Load all the W-Maps using nibabel.
    """
    # Import all the W-Maps using nibabel
    wmap_files = df[wmaps_col].values
    print(f'Loading {len(wmap_files)} W-Maps...')
    # Using nibabel, load the wmap files in the list:
    wmaps = [nib.load(wmap) for wmap in wmap_files]
    # Return the wmaps:
    return wmaps

# Load in specified atlas using nilearn:
def load_atlas(atlas):
    """
    Load the atlas using nibabel.
    """
    # Load the atlas using nibabel:
    atlas_maps = atlas.maps
    atlas_labels = atlas.labels
    # Return the atlas:
    return atlas_maps, atlas_labels

# Resample W-Maps to dimensions of specified atlas:
def resample_wmaps(wmaps, atlas_maps):
    """
    Resample W-Maps to Atlas shape for Parcellation-Based Analysis.
    """
    # Resample W-Maps to Atlas shape for Parcellation-Based Analysis:
    print('Resampling W-Maps to Atlas shape for Parcellation-Based Analysis...')
    wmaps_parcel = [resample_to_img(wmap, atlas_maps) for wmap in wmaps]
    # Return the resampled W-Maps:
    return wmaps_parcel

# Fit W-Maps to the specified masker:
def fit_wmaps(wmaps, masker):
    """
    Fit the data to the specified masker,
    then for each W-Map in the fitted data,
    inverse transform each W-Map back to the original shape.
    """
    # Fit the train and test W-Maps to the masker:
    wmaps_fitted = masker.fit_transform(wmaps)
    # Print the shape of the fitted data:
    print(f'Fitted shape: {wmaps_fitted.shape}')
    # For each W-Map in the fitted train and test sets,
    # inverse transform each W-Map back to the original shape:
    wmaps_avg = [masker.inverse_transform(wmap) for wmap in wmaps_fitted]
    # Print the shape of the first case in the set:
    print(f'Inverse-transformed shape: {wmaps_avg[0].shape}')
    # Return the inverse-transformed W-Maps:
    return wmaps_avg

# Create pipeline_analysis.py ready CSV:
def create_pipeline_csv(df, output_dir):
    """
    """
    # Create a blank dataframe with the following columns:
    df_pipeline = pd.DataFrame(columns=['PIDN','Diagnosis','DCDate','Site','Path','ScannerID','SourceID'])
    # Find the matching columns in df
    df_pipeline['PIDN'] = df['PIDN']
    df_pipeline['DCDate'] = df['DCDate']
    df_pipeline['Path'] = df['ImgPath']
    df_pipeline['ScannerID'] = df['ScannerID']
    df_pipeline['SourceID'] = df['SourceID']

    # Set the remaining columns to defaults:
    df_pipeline['Diagnosis'] = 1
    df_pipeline['Site'] = 'MAC'

    # Remove Timestamps from DCDate, leaving only YYYY-MM-DD format and convert NaN to NaT:
    df_pipeline['DCDate']  = pd.to_datetime(df_pipeline['DCDate'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

    # Save the dataframe as a CSV file in the output directory:
    df_pipeline.to_csv(os.path.join(output_dir, 'pipeline_analysis.csv'), index=False)
    return df_pipeline
# %%
