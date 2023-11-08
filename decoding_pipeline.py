# %%
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
# ---------------------------- STANDARD LIBRARIES ---------------------------- #
import glob
import math
import os
import pickle
import shutil
import sys
import time
import psutil

# ---------------------------- 3RD PARTY LIBRARIES --------------------------- #
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import nibabel as nib
import nilearn.datasets
import nilearn.decoding as decoding
import nilearn.image as image
from nilearn.image import resample_to_img, get_data
import nilearn.maskers as maskers
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.plotting import plot_stat_map, show
import nilearn.plotting as plotting
import numpy as np
import pandas as pd
import pingouin as pg
import plotly as py
import ptitprince as pt
import scipy.stats as stats
import seaborn as sns
import sklearn
from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, RidgeCV, LogisticRegressionCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, mean_squared_log_error, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.api import abline_plot, interaction_plot
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable

# ----------------------------- CUSTOM LIBRARIES ----------------------------- #
# from alba_imaging.imaging_core import create_wynton_job_file
from alba_imaging.modeling import decoder_python_job
from alba_imaging.imaging_core import lava_wmap_filepath, get_lava_wmap_filepaths, new_wmap_filepath, get_new_wmap_filepaths, clean_df_wmaps
# from alba_imaging.importing import clean_and_sort

# %%
# ----------------------------- SET UP FILEPATHS ----------------------------- #
# Make a folder on the L drive for the analysis.
# NOTE: This is where all job files will be saved and results will be copied.

# Specify the main filepath as it is read on both your local machine,
# and on the MGT node:
local_ldrive_dir = '/volumes/language/language/rbogley/wmaps_decoding/edevhx_project/'
mgt_ldrive_dir = '/shared/language/language/rbogley/wmaps_decoding/edevhx_project/'

# Specify the main CSV filename and ensure it is in the above main folder:
# csv_filename = 'LAVA_Merged_Data_first_neuropsychbedside_2023-11-03.csv'
csv_filename = 'LAVA_Merged_Data_first_neuropsychbedside_2023-11-07.csv'

# Now specify a path to a working directory on the MGT node that will be created:
mgt_working_dir = '/mgt/language/rbogley/Production/projects/wmaps_project/edevhx_project/'

# Now specify a path to the R-Drive where the MAC Imaging-Core generated W-Maps
# are found (as it is read on the MGT node):
wmaps_rdrive_dir = '/shared/macdata/projects/knect/images/wmaps/spmvbm12/'

# %%
# ----------------------------- DEFINE VARIABLES ----------------------------- #
# Define variables specific for your analysis from your dataset.
# dep_var_list = ['Calc','MMSETot','BNTCorr','BryTot','DigitFW','DigitBW','MTCorr','NumbLoc','ModRey','WRATTot'] # specify all dependent variable names (column names)
dep_var_list = ['ARHQ','Sum Spelling & Reading','Sum Math & Geometry','Stuttering?']

# covar_list = ['Gender','Educ','AgeAtDC_neuropsychbedside'] # specify all covariate names
# Specify PIDN and MRI DCDate column names:
pidn_col = 'PIDN'
dcdate_col = 'DCDate_t1'
wmaps_col = 'wmap_lava_mgt'

# %%
# --------------------------- DEFINE MODEL SETTINGS -------------------------- #
# Define parameters for the decoder:
estimator_type = 'ridge' # 'ridge' or 'svr'
atlas = 'harvard_oxford' # 'harvard_oxford' or 'aal'
analysis_type = 'voxel' # 'voxel' or 'parcel' - NOTE: May be able to remove this and just create jobs for both each time.
train_test_ratio = 0.8 # 0.6 = 60% train, 40% test
stratify = True # True or False
n_jobs = 8 # Number of jobs to run in parallel (do not exceed CPU core count)

# %%
# ---------------------------------------------------------------------------- #
#                                   RUN PREP                                   #
# ---------------------------------------------------------------------------- #
# ---------------------------- GENERATE JOB FILES ---------------------------- #

local_csv_filepath = os.path.join(local_ldrive_dir, csv_filename)
mgt_csv_filepath = os.path.join(mgt_ldrive_dir, csv_filename)

# Import the main csv file as a df:
df = pd.read_csv(local_csv_filepath, low_memory=False)

# %%
for dep_var in dep_var_list:
    print(f"Generating python job file and csv for {dep_var}...")
    df_temp = df.copy()

    ################################################
    # NOTE: TEMPORARILY TRIM DF:
    df_temp = df_temp[df_temp[dep_var].notna()]
    df_temp = df_temp[df_temp['wmap_lava'].notna()] #NOTE: MAKE WMAP LAVA A VARIABLE
    df_temp = df_temp.sample(n=40, random_state=2)
    ################################################

    decoder_python_job(csv=mgt_csv_filepath,
                        dep_var=dep_var,
                        wmaps_col=wmaps_col,
                        local_ldrive_dir=local_ldrive_dir,
                        mgt_ldrive_dir=mgt_ldrive_dir,
                        mgt_working_dir=mgt_working_dir,
                        analysis_type=analysis_type,
                        estimator_type=estimator_type,
                        atlas=atlas,
                        train_test_ratio=train_test_ratio,
                        n_jobs=n_jobs
                        )

    # NOTE: OLD FOR WYNTON RUNS, DO NOT USE YET:
    # decoder_wynton_job(csv=csv_filepath,
    #                     dep_var=dep_var,
    #                     ldrive_dir=ldrive_dir,
    #                     mgt_dir=mgt_dir,
    #                     wynton_venv_dir=wynton_venv_dir,
    #                     analysis_type='voxel',
    #                     estimator="ridge",
    #                     atlas="harvard_oxford")



# %%
