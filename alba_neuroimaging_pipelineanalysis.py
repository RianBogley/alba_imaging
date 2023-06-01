# %% Neuroimaging Pipeline Analysis Setup Tools by Rian Bogley#################
###############################################################################
# %% PACKAGES #################################################################
# Import Packages
import os
import pandas as pd
###############################################################################
# %% GLOBAL VARIABLES #########################################################
###############################################################################
# %% FUNCTIONS ################################################################
# Create pipeline_analysis.py ready CSV:
def create_pipeline_csv(df_input, output_dir):
    """
    """
    # Create a blank dataframe with the following columns:
    df = pd.DataFrame(columns=['PIDN','Diagnosis','DCDate','Site','Path','ScannerID','SourceID'])
    # Find the matching columns in df_input
    df['PIDN'] = df_input['PIDN']
    df['DCDate'] = df_input['DCDate']
    df['Path'] = df_input['ImgPath']
    df['ScannerID'] = df_input['ScannerID']
    df['SourceID'] = df_input['SourceID']

    # Set the remaining columns to defaults:
    df['Diagnosis'] = 1
    df['Site'] = 'MAC'

    # Remove Timestamps from DCDate, leaving only YYYY-MM-DD format and convert NaN to NaT:
    df['DCDate']  = pd.to_datetime(df['DCDate'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

    # Save the dataframe as a CSV file in the output directory:
    df.to_csv(os.path.join(output_dir, 'pipeline_analysis.csv'), index=False)
    return df
###############################################################################