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
from nilearn.decoding import DecoderRegressor
import nilearn.image as image
from nilearn.image import resample_to_img, get_data
import nilearn.maskers as maskers
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.plotting import plot_stat_map, show
import nilearn.plotting as plotting
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import sklearn
from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, RidgeCV, LogisticRegressionCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, mean_squared_log_error, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ----------------------------- CUSTOM LIBRARIES ----------------------------- #
from alba_imaging.imaging_core import get_lava_wmap_filepaths, load_wmaps, load_atlas, resample_wmaps, fit_wmaps

# %%
# ---------------------------------------------------------------------------- #
#                               DEFINE FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def memory_usage_mb():
    # Return memory usage in MB:
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem

# This function creates a specific python job file to run the decoder on the 
# MGT node using a virtual environment.
def decoder_python_job(csv,
                       dep_var,
                       wmaps_col,
                       local_ldrive_dir,
                       mgt_ldrive_dir,
                       mgt_working_dir,
                       analysis_type='voxel',
                       estimator_type='ridge',
                       atlas='harvard_oxford',
                       train_test_ratio=0.6,
                       stratify=False,
                       n_jobs=8):
    """
    This function creates a python file for running the decoder on the MGT node.
    Inputs are:
    - csv = CSV containing at the very least:
        - A column with the filepaths to the wmap for each case
            (wmaps must be on the node or the servers, with filepaths setup
            correctly such that they can be accessed from the node, e.g.
            /shared/language/... or /mgt/language/...)
        - A column with the dependent variable of interest.
    - dep_var = The name of the dependent variable of interest
        (should be same name as col)
    - wmaps_col = The name of the column with the wmap filepaths
    - local_ldrive_dir = The location of the L-Drive directory as it is read on your local machine.
    - mgt_ldrive_dir = The location of the L-Drive directory as it is read on the MGT node.
    - mgt_working_dir = The location of the MGT directory where the output will be saved.
    - analysis_type = The type of analysis to be run (i.e. 'voxel' or 'parcel')
    - estimator_type = The type of estimator_type to be used (e.g. 'ridge' or 'svr')
    - atlas = The atlas to be used (e.g. 'harvard_oxford' or 'aal')
    - train_test_ratio = The ratio of training to test data (e.g. 0.6 = 60% train, 40% test)
    - stratify = Whether to stratify the data (True or False) False = normal sampling, True = stratified sampling
    """

    # Set up the run info:
    run_name = f'decoder_{dep_var}_{analysis_type}_{estimator_type}_{atlas}'

    python_filename = f'{run_name}.py'

    run_dir = os.path.join(mgt_working_dir, run_name)

    python_mgt_filepath = os.path.join(mgt_working_dir, python_filename)

    python_local_ldrive_filepath = os.path.join(local_ldrive_dir, python_filename)

    # Set up the contents of the python script to be run on the MGT node:
    python_script_text = f"""# Import modules:
import os
import pandas as pd
import nilearn
import sklearn
import nibabel
import numpy as np
import pickle
import time
import psutil 
import shutil

with sklearn.config_context(assume_finite=True):
    pass

with sklearn.config_context(working_memory=125):
    pass

from alba_imaging.modeling import decoder_split_data, decoder_prep, decoder_run, decoder_save, memory_usage_mb
from alba_imaging.imaging_core import load_wmaps, load_atlas, resample_wmaps, fit_wmaps


# Start the timer and memory measurement
start_time = time.time()
start_memory = memory_usage_mb()

# Make the main mgt_working_dir:
if not os.path.exists('{mgt_working_dir}'):
    os.mkdir('{mgt_working_dir}')

# Make the run subdir in the main mgt_working_dir:
if not os.path.exists('{run_dir}'):
    os.mkdir('{run_dir}')

# Import the csv as dataframe:
df=pd.read_csv('{csv}')

atlas_maps,atlas_labels,y_train,y_test,wmaps_train,wmaps_test,wmaps_train_avg,wmaps_test_avg,masker,estimator,estimator_type,dep_var = decoder_prep(df=df,dep_var='{dep_var}',wmaps_col='{wmaps_col}',estimator_type='{estimator_type}',atlas='{atlas}',train_test_ratio={train_test_ratio},stratify={stratify})
prep_time = time.time()

# Number of subjects in train and test sets:
n_train = len(wmaps_train)
n_test = len(wmaps_test)

decoder_results = decoder_run(dep_var=dep_var,
                                atlas_maps=atlas_maps,
                                atlas_labels=atlas_labels,
                                y_train=y_train,
                                y_test=y_test,
                                wmaps_train=wmaps_train,
                                wmaps_test=wmaps_test,
                                wmaps_train_avg=wmaps_train_avg,
                                wmaps_test_avg=wmaps_test_avg,
                                masker=masker,
                                estimator=estimator,
                                estimator_type=estimator_type,
                                scoring='neg_mean_squared_error',
                                screening_percentile=5,
                                n_jobs={n_jobs},
                                standardize=True)
run_time = time.time()
                                
decoder_save(decoder_results=decoder_results, main_dir='{run_dir}',
                dep_var='{dep_var}', analysis_type='{analysis_type}', estimator_type='{estimator_type}', atlas='{atlas}')

# End the timer and memory measurement
end_time = time.time()
end_memory = memory_usage_mb()

# Calculate elapsed time at each step and total memory used
start_to_prep_time = prep_time - start_time
prep_to_run_time = run_time - prep_time
run_to_end_time = end_time - run_time
total_time = end_time - start_time
memory_used = end_memory - start_memory

# Save the results of the times to a txt file:
with open('{run_dir}/{run_name}_run_info.txt', 'w') as f:
    # Print the info about the run:
    f.write(f'Run information:\\n')
    f.write(f'\\n')
    f.write(f'Run name: {run_name}\\n')
    f.write(f'CSV file: {csv}\\n')
    f.write(f'Dependent variable: {dep_var}\\n')
    f.write(f'W-Maps column: {wmaps_col}\\n')
    f.write(f'Analysis type: {analysis_type}\\n')
    f.write(f'Estimator type: {estimator_type}\\n')
    f.write(f'Atlas: {atlas}\\n')
    f.write(f'Stratify: {stratify}\\n')
    f.write(f'Train test ratio: {train_test_ratio}\\n')
    f.write(f'Number of subjects in train set: {'{n_train}'}\\n')
    f.write(f'Number of subjects in test set: {'{n_test}'}\\n')
    f.write(f'\\n')
    f.write(f'Time to run prep step: {'{start_to_prep_time}'} seconds.\\n')
    f.write(f'Time to run decoder: {'{prep_to_run_time}'} seconds.\\n')
    f.write(f'Time to save decoder results: {'{run_to_end_time}'} seconds.\\n')
    f.write(f'Total time to run decoder: {'{total_time}'} seconds.\\n')
    f.write(f'\\n')
    f.write(f'Memory used: {'{memory_used}'} MB.\\n')

# Check if the mgt L drive path exists, if so, copy the results folder to it and overwrite if it exists:
if os.path.exists('{mgt_ldrive_dir}') and os.path.exists('{run_dir}'):
    shutil.copytree('{run_dir}', '{mgt_ldrive_dir}/{run_name}', dirs_exist_ok=True)

"""
    # Print the location where files are being saved:
    print(f'Python script saved to: {python_local_ldrive_filepath}')

    with open(python_local_ldrive_filepath, 'w') as f:
        f.write(python_script_text)

    return run_name, python_filename, run_dir, python_local_ldrive_filepath

def decoder_split_data(wmaps, y, train_test_ratio=0.6, stratify=True):
    """
    Split the data into training and test sets.
    Specify whether to stratify depending on the dependent variable type.
    """
    print('Running decoder_split_data...')
    if stratify:
        print('Using stratified sampling to split data into training and test sets.')
        # Use stratified sampling to split the data into training and test sets:
        wmaps_train, wmaps_test, y_train, y_test = train_test_split(
            wmaps, y, train_size=train_test_ratio, random_state=0, stratify=y)
    else:
        print('Using normal sampling to split data into training and test sets.')
        # Use normal sampling to split the data into training and test sets:
        wmaps_train, wmaps_test, y_train, y_test = train_test_split(
            wmaps, y, train_size=train_test_ratio, random_state=0)

    # Print the shape, size, and number of cases in the training and test sets:
    print(f'wmaps_train shape: {wmaps_train[0].shape}')
    print(f'wmaps_test shape: {wmaps_test[0].shape}')
    print(f'{len(wmaps_train)} cases in wmaps_train.')
    print(f'{len(wmaps_test)} cases in wmaps_test.')

    # Return the split data:
    return wmaps_train, wmaps_test, y_train, y_test

def decoder_prep(df,
                dep_var,
                wmaps_col,
                atlas="harvard_oxford",
                estimator_type="ridge",
                train_test_ratio=0.6,
                stratify=False):
    """
    Prep the data for the decoder.
    """
    print('Running decoder_prep...')
    if estimator_type == "ridge":
        estimator = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    elif estimator_type == "svr":
        estimator = SVR(kernel='linear')


    if atlas == "harvard_oxford":
        atlas = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    elif atlas == "aal":
        atlas = nilearn.datasets.fetch_atlas_aal(version='SPM12')

    # ------------------------------ TRIM DATAFRAME ------------------------------ #
    df = df[df[dep_var].notna()]
    df = df[df[wmaps_col].notna()]
    
    # --------------------------------- DEFINE Y --------------------------------- #
    y = np.array(df[dep_var])
    
    # ------------------------------ LOAD THE W-MAPS ----------------------------- #
    # Load all the W-Maps using nibabel:
    wmaps = load_wmaps(df=df, wmaps_col=wmaps_col)

    # ------------------------------ LOAD THE ATLAS ------------------------------ #
    # Load the prespecified atlas using nibabel:
    atlas_maps, atlas_labels = load_atlas(atlas=atlas)

    # -------------------- SPLIT INTO TRAININGS AND TEST SETS -------------------- #
    # Assumes variable is continuous, and will use normal sampling (not stratified):
    wmaps_train, wmaps_test, y_train, y_test = decoder_split_data(wmaps=wmaps, y=y, train_test_ratio=train_test_ratio, stratify=stratify)
    
    #TODO: ADD LOGIC TO STRATIFY IF CATEGORICAL        

    # ------------------------------ SET UP MASKERS ------------------------------ #
    # Import NifTi masker for Voxel-Based Analysis:
    masker = NiftiMasker(standardize=True, smoothing_fwhm=2, memory='nilearn_cache')

    # ------------------------------ FIT THE W-MAPS ------------------------------ #
    # Fit the wmaps data:
    # Voxel-Based Analysis:
    wmaps_train_avg = fit_wmaps(wmaps=wmaps_train, masker=masker)
    wmaps_test_avg = fit_wmaps(wmaps=wmaps_test, masker=masker)

    return atlas_maps, atlas_labels, y_train, y_test, wmaps_train, wmaps_test, wmaps_train_avg, wmaps_test_avg, masker, estimator, estimator_type, dep_var

def decoder_run(dep_var,
                atlas_maps,
                atlas_labels,
                y_train,
                y_test,
                wmaps_train,
                wmaps_test,
                wmaps_train_avg,
                wmaps_test_avg,
                masker,
                estimator,
                estimator_type,
                scoring='neg_mean_squared_error',
                screening_percentile=5,
                n_jobs=8,
                standardize=True):
    """
    Run the decoder using the specified parameters and
    fit the data to the decoder.
    """
    print('Running decoder_run...')
    # Define the decoder:
    decoder = DecoderRegressor(estimator=estimator,
                                mask=masker,
                                screening_percentile=screening_percentile,
                                n_jobs=n_jobs,
                                standardize=standardize)
    print(f'Decoder parameters: {decoder.get_params()}')

    # Fit the decoder to the training data:
    decoder.fit(wmaps_train, y_train)

    # Sort the test data for better visualization:
    perm = np.argsort(y_test)[::-1]
    y_test_final= y_test[perm]
    wmaps_test_final = np.array(wmaps_test)[perm]

    # Predict the test data using the decoder:
    y_pred = decoder.predict(wmaps_test_final)

    # Calculate Mean Absolute Error:
    prediction_score = -np.mean(decoder.cv_scores_['beta'])

    # Get the weight image:
    weight_img = decoder.coef_img_['beta']
    
    decoder_results = {
        'dep_var': dep_var,
        'decoder': decoder,
        'masker': masker,
        'atlas_maps': atlas_maps,
        'atlas_labels': atlas_labels,
        'y_pred': y_pred,
        'y_test_final': y_test_final,
        'wmaps_test_final': wmaps_test_final,
        'prediction_score': prediction_score,
        'weight_img': weight_img,
    }

    # decoder_results = {
    # 'decoder': decoder,
    # 'masker': masker,
    # 'atlas_maps': atlas_maps,
    # 'atlas_labels': atlas_labels,
    # 'y_train': y_train,
    # 'y_test': y_test,
    # 'y_test_final': y_test_final,
    # 'y_pred': y_pred,
    # 'wmaps_train': wmaps_train,
    # 'wmaps_test': wmaps_test,
    # 'wmaps_train_avg': wmaps_train_avg,
    # 'wmaps_test_avg': wmaps_test_avg,
    # 'wmaps_test_final': wmaps_test_final,
    # 'prediction_score': prediction_score,
    # 'weight_img': weight_img
    # }

    # Return the results:
    return decoder_results

def decoder_save(decoder_results, main_dir, dep_var, analysis_type, estimator_type, atlas):
    """
    Save the decoder data.
    """
    print('Running decoder_save...')
    # Current Date:
    current_date = time.strftime("%m-%d-%Y")

    with open(f'{main_dir}/{dep_var}_{analysis_type}_{estimator_type}_{atlas}_{current_date}_decoder_results.pkl', 'wb') as f:
        pickle.dump(decoder_results, f)
    # Save the weight_img as a nifti file to the dir:
    nib.save(decoder_results['weight_img'], f'{main_dir}/{dep_var}_{analysis_type}_{estimator_type}_{atlas}_{current_date}_weight_img.nii.gz')

# %%




# def decoder_wynton_job(csv,
#                        dep_var,
#                        mgt_ldrive_dir='/Users/rbogley/Desktop/',
#                        mgt_working_dir='/mgt/language/rbogley/Production/projects/wmaps_project/',
#                        wynton_venv_dir="/wynton/home/tempini/rbogley/wmaps_predictor/",
#                        analysis_type='voxel',
#                        estimator='ridge',
#                        atlas='harvard_oxford',):
#     """
#     This function creates a python and shell script file for running the decoder on Wynton.
#     """

#     # Set up the output filepaths:
#     python_filename = f'decoder_{dep_var}_{analysis_type}_{estimator}_{atlas}.py'
#     job_filename = f'QB3_decoder_{dep_var}_{analysis_type}_{estimator}_{atlas}_job.sh'

#     python_mgt_filepath = os.path.join(mgt_working_dir, python_filename)
#     job_mgt_filepath = os.path.join(mgt_working_dir, job_filename)

#     python_local_ldrive_filepath = os.path.join(mgt_ldrive_dir, python_filename)
#     job_ldrive_filepath = os.path.join(mgt_ldrive_dir, job_filename)

#     # Set up the contents of the python script and Wynton job file:
#     python_script_text = f"""# Import modules:
# import pandas as pd
# from alba_imaging.modeling import decoder_split_data, decoder_prep, decoder_run, decoder_save

# df=pd.read_csv('{csv}')

# decoder_prep(estimator_type='{estimator_type}',df=df,dep_var='{dep_var}',atlas='{atlas}')

# decoder_run(wmaps_train=wmaps_train, wmaps_test=wmaps_test,
#             y_train=y_train, y_test=y_test, estimator='{estimator}', 
#             masker=masker, scoring='neg_mean_squared_error', screening_percentile=5, 
#             n_jobs=12, standardize=True)

# decoder_save(decoder=decoder, main_dir={mgt_working_dir},
#                 dep_var='{dep_var}', analysis_type='{analysis_type}')
#     """

#     with open(python_local_ldrive_filepath, 'w') as f:
#         f.write(python_script_text)

#     create_wynton_job_file(mgt_ldrive_dir=mgt_ldrive_dir,
#                             wynton_venv_dir=wynton_venv_dir,
#                             python_mgt_filepath=python_mgt_filepath,
#                             jobfile_name=job_filename)


    
# %%
