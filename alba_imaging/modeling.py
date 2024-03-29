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
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ----------------------------- CUSTOM LIBRARIES ----------------------------- #
from alba_imaging.imaging_core import get_lava_wmap_filepaths, load_wmaps, load_atlas, resample_wmaps, fit_wmaps
from alba_imaging.filtering import filter_df_criteria, clean_and_sort

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
                       y_col,
                       X_col,
                       local_ldrive_dir,
                       mgt_ldrive_dir,
                       analysis_type,
                       estimator_type,
                       atlas,
                       train_test_ratio,
                       stratify,
                       n_jobs,
                       exclusions_dict=None,
                       inclusions_dict=None):
    """
    This function creates a python file for running the decoder on the MGT node.
    Inputs are:
    - csv = CSV containing at the very least:
        - A column with the filepaths to the wmap for each case
            (wmaps must be on the node or the servers, with filepaths setup
            correctly such that they can be accessed from the node, e.g.
            /shared/language/... or /mgt/language/...)
        - A column with the dependent variable of interest.
    - y_col = The name of the column with the dependent variable of interest
        (should be same name as col)
    - X_col = The name of the column with the independent variable (i.e. wmap filepaths)
    - local_ldrive_dir = The location of the L-Drive directory as it is read on your local machine.
    - mgt_ldrive_dir = The location of the L-Drive directory as it is read on the MGT node.
    - mgt_working_dir = The location of the MGT directory where the output will be saved.
    - analysis_type = The type of analysis to be run (i.e. 'voxel' or 'parcel')
    - estimator_type = The type of estimator_type to be used (e.g. 'ridge' or 'svr')
    - atlas = The atlas to be used (e.g. 'harvard_oxford' or 'aal')
    - train_test_ratio = The ratio of training to test data (e.g. 0.6 = 60% train, 40% test)
    - stratify = Whether to stratify the data (True or False) False = normal sampling, True = stratified sampling
    - n_jobs = The number of jobs to run in parallel.
    - exclusions_dict = A dictionary of column names and corresponding lists of values to exclude.
    - inclusions_dict = A dictionary of column names and corresponding lists of values to include.
        Example: {'Diagnosis_Column': ['Normal', 'Control']}
    """

    # Set up the run info:
    run_name = f'decoder_{y_col}_{analysis_type}_{estimator_type}_ttr{train_test_ratio}_{atlas}'
    python_filename = f'{run_name}.py'
    mgt_run_dir = os.path.join(mgt_ldrive_dir, run_name)
    local_run_dir = os.path.join(local_ldrive_dir, run_name)
    mgt_run_python_filepath = os.path.join(mgt_run_dir, python_filename)
    local_run_python_filepath = os.path.join(local_run_dir, python_filename) 

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
from alba_imaging.filtering import clean_and_sort, filter_df_criteria
from alba_imaging.imaging_core import load_wmaps, load_atlas, resample_wmaps, fit_wmaps

# Start the timer and memory measurement
start_time = time.time()
start_memory = memory_usage_mb()

# Make the main mgt_working_dir:
if not os.path.exists('{mgt_run_dir}'):
    print(f'Creating run directory: {mgt_run_dir}')
    os.mkdir('{mgt_run_dir}')

# Import the csv as dataframe:
df=pd.read_csv('{csv}')

exclusions_dict = {exclusions_dict}
inclusions_dict = {inclusions_dict}

decoder_input = decoder_prep(df=df,
                                y_col='{y_col}',
                                X_col='{X_col}',
                                atlas='{atlas}',
                                estimator_type='{estimator_type}',
                                train_test_ratio={train_test_ratio},
                                stratify={stratify},
                                analysis_type='{analysis_type}',
                                exclusions_dict=exclusions_dict,
                                inclusions_dict=inclusions_dict)
prep_time = time.time()
print(decoder_input)

# Number of subjects in train and test sets:
n_train = len(decoder_input['X_train'])
n_test = len(decoder_input['X_test'])

decoder_results = decoder_run(y_col='{y_col}',
                                atlas_maps=decoder_input['atlas_maps'],
                                atlas_labels=decoder_input['atlas_labels'],
                                y_train=decoder_input['y_train'],
                                y_test=decoder_input['y_test'],
                                X_train=decoder_input['X_train'],
                                X_test=decoder_input['X_test'],
                                masker=decoder_input['masker'],
                                estimator=decoder_input['estimator'],
                                estimator_type=decoder_input['estimator_type'],
                                analysis_type='{analysis_type}',
                                scoring='neg_mean_squared_error',
                                screening_percentile=5,
                                n_jobs={n_jobs},
                                standardize=True)
run_time = time.time()
print(decoder_results)
                                
decoder_save(decoder_results=decoder_results, main_dir='{mgt_run_dir}',
                y_col='{y_col}', analysis_type='{analysis_type}', estimator_type='{estimator_type}', atlas='{atlas}')
            

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
with open('{mgt_run_dir}/{run_name}_run_info.txt', 'w') as f:
    # Print the info about the run:
    f.write(f'Run information:\\n')
    f.write(f'\\n')
    f.write(f'Run name: {run_name}\\n')
    f.write(f'CSV file: {csv}\\n')
    f.write(f'Dependent variable: {y_col}\\n')
    f.write(f'Indenpendent variable column: {X_col}\\n')
    f.write(f'Analysis type: {analysis_type}\\n')
    f.write(f'Estimator type: {estimator_type}\\n')
    f.write(f'Atlas: {atlas}\\n')
    f.write(f'Stratify: {stratify}\\n')
    f.write(f'Train test ratio: {train_test_ratio}\\n')
    f.write(f'Number of jobs: {n_jobs}\\n')
    f.write(f'\\n')
    if exclusions_dict and exclusions_dict != {{}}:
        f.write(f'Exclusion criteria:\\n')
        for col in exclusions_dict.keys():
            f.write(f'{'{col}'}: {'{exclusions_dict[col]}'}\\n')
    if inclusions_dict and inclusions_dict != {{}}:
        f.write(f'Inclusion criteria:\\n')
        for col in inclusions_dict.keys():
            f.write(f'{'{col}'}: {'{inclusions_dict[col]}'}\\n')

    f.write(f'\\n')    
    f.write(f'Number of subjects in train set: {'{n_train}'}\\n')
    f.write(f'Number of subjects in test set: {'{n_test}'}\\n')
    f.write(f'\\n')
    f.write(f'Time to run prep step: {'{start_to_prep_time}'} seconds.\\n')
    f.write(f'Time to run decoder: {'{prep_to_run_time}'} seconds.\\n')
    f.write(f'Time to save decoder results: {'{run_to_end_time}'} seconds.\\n')
    f.write(f'Total time to run decoder: {'{total_time}'} seconds.\\n')
    f.write(f'\\n')
    f.write(f'Memory used: {'{memory_used}'} MB.\\n')


"""
    # Check if the local_ldrive_filepath doesn't exist, if it doesn't, create it:
    if not os.path.exists(local_run_dir):
        print(f'Creating run directory: {local_run_dir}')
        os.mkdir(local_run_dir)
    
    # Check if the path exists, if so, save the python script to it:
    print(f'Python script saved to: {local_run_python_filepath}')
    with open(local_run_python_filepath, 'w') as f:
        f.write(python_script_text)

    return run_name, python_filename, local_run_dir, local_run_python_filepath

def decoder_split_data(wmaps, y, train_test_ratio=0.8, stratify=True):
    """
    Split the data into training and test sets.
    Specify whether to stratify depending on the dependent variable type.
    """
    print('Running decoder_split_data...')
    if stratify:
        print('Using stratified sampling to split data into training and test sets.')
        # Use stratified sampling to split the data into training and test sets:
        X_train, X_test, y_train, y_test = train_test_split(
            wmaps, y, train_size=train_test_ratio, random_state=0, stratify=y)
    else:
        print('Using normal sampling to split data into training and test sets.')
        # Use normal sampling to split the data into training and test sets:
        X_train, X_test, y_train, y_test = train_test_split(
            wmaps, y, train_size=train_test_ratio, random_state=0)

    # Print the shape, size, and number of cases in the training and test sets:
    print(f'X_train shape: {X_train[0].shape}')
    print(f'X_test shape: {X_test[0].shape}')
    print(f'{len(X_train)} cases in X_train.')
    print(f'{len(X_test)} cases in X_test.')

    # Return the split data:
    return X_train, X_test, y_train, y_test

def decoder_prep(df,
                y_col,
                X_col,
                atlas,
                estimator_type,
                train_test_ratio,
                stratify,
                analysis_type,
                exclusions_dict=None,
                inclusions_dict=None):
    """
    Prep the data for the decoder.
    y_col = The column name of the dependent variable of interest.
    X_col = The column name of the wmaps.
    atlas = The atlas to be used (e.g. 'harvard_oxford' or 'aal')
    estimator_type = The type of estimator_type to be used (e.g. 'ridge' or 'svr')
    train_test_ratio = The ratio of training to test data (e.g. 0.8 = 80% train, 20% test)
    stratify = Whether to stratify the data (True or False) False = normal sampling, True = stratified sampling
    analysis_type = The type of analysis to be run (i.e. 'voxel' or 'parcel')
    """
    print('Running decoder_prep...')
    if estimator_type == "ridge":
        estimator = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    elif estimator_type == "svr":
        estimator = SVR(kernel='linear')
    elif estimator_type == "logistic":
        estimator = LogisticRegressionCV(cv=5, random_state=0, max_iter=10000)

    if atlas == "harvard_oxford":
        atlas = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    elif atlas == "aal":
        atlas = nilearn.datasets.fetch_atlas_aal(version='SPM12')

    # ----------------------- CLEAN AND SORT THE DATAFRAME ----------------------- #
    # Drop any cases that have an invalid or missing value for X or y:
    df = clean_and_sort(df=df, col_name=y_col, pidn_col='PIDN')
    df = clean_and_sort(df=df, col_name=X_col, pidn_col='PIDN')

    # -------------------------- INCLUSIONS & EXCLUSIONS ------------------------- #
    # If any additional exclusion or inclusion criteria are specified, filter the df:
    # Otherwise, the data will only exclude cases with invalid values in the y & X.

    # EXCLUSIONS:
    if exclusions_dict and exclusions_dict != {}:
        # Clean and sort to get rid of any cases that have an invalid or missing value
        # for any of the exclusion criteria columns:
        for col in exclusions_dict.keys():
            df = clean_and_sort(df=df, col_name=col, pidn_col='PIDN')
        df = filter_df_criteria(df, exclusions_dict, policy='exclude')
    # INCLUSIONS:
    if inclusions_dict and inclusions_dict != {}:
        # Clean and sort to get rid of any cases that have an invalid or missing value
        # for any of the inclusion criteria columns:
        for col in inclusions_dict.keys():
            df = clean_and_sort(df=df, col_name=col, pidn_col='PIDN')
        df = filter_df_criteria(df, inclusions_dict, policy='include')

    # --------------------------------- DEFINE Y --------------------------------- #
    y = np.array(df[y_col])
    
    # ------------------------------ LOAD THE W-MAPS ----------------------------- #
    # Load all the W-Maps using nibabel:
    wmaps = load_wmaps(df=df, wmaps_col=X_col)

    # ------------------------------ LOAD THE ATLAS ------------------------------ #
    # Load the prespecified atlas using nibabel:
    atlas_maps, atlas_labels = load_atlas(atlas=atlas)

    # ----------------- RESAMPLE W-MAPS IF PARCEL-BASED ANALYSIS ----------------- #
    if analysis_type == "parcel":
        print(f'Parcel-based analysis selected. Resampling wmaps to {atlas} atlas maps.')
        wmaps = resample_wmaps(wmaps=wmaps, atlas_maps=atlas_maps)

    # -------------------- SPLIT INTO TRAININGS AND TEST SETS -------------------- #
    # Assumes variable is continuous, and will use normal sampling (not stratified):
    X_train, X_test, y_train, y_test = decoder_split_data(wmaps=wmaps, y=y, train_test_ratio=train_test_ratio, stratify=stratify)
    
    #TODO: ADD LOGIC TO STRATIFY IF CATEGORICAL        

    # ------------------------------ SET UP MASKERS ------------------------------ #
    if analysis_type == "voxel":
        print(f'Voxel-based analysis selected. Using NiftiMasker.')
        masker = NiftiMasker(standardize=True, smoothing_fwhm=2, memory='nilearn_cache')
        print(f'Masker parameters: {masker.get_params()}')
    elif analysis_type == "parcel":
        print(f'Parcel-based analysis selected. Using NiftiLabelsMasker.')
        masker = NiftiLabelsMasker(labels_img=atlas_maps, standardize=True, smoothing_fwhm=2, memory='nilearn_cache')
        print(f'Masker parameters: {masker.get_params()}')
        # Fit the wmaps to the Atlas:
        X_train = fit_wmaps(wmaps=X_train, masker=masker)
        X_test = fit_wmaps(wmaps=X_test, masker=masker)

    decoder_input = {
        'atlas_maps': atlas_maps,
        'atlas_labels': atlas_labels,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'masker': masker,
        'estimator': estimator,
        'estimator_type': estimator_type,
        'y_col': y_col,
        'analysis_type': analysis_type,
    }

    return decoder_input

def decoder_run(y_col,
                atlas_maps,
                atlas_labels,
                y_train,
                y_test,
                X_train,
                X_test,
                masker,
                estimator,
                estimator_type,
                analysis_type,
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
                                scoring=scoring,
                                screening_percentile=screening_percentile,
                                n_jobs=n_jobs,
                                standardize=standardize)
    print(f'Decoder parameters: {decoder.get_params()}')

    # Fit the decoder to the training data:
    decoder.fit(X_train, y_train)

    # Sort the test data for better visualization:
    perm = np.argsort(y_test)[::-1]
    y_test= y_test[perm]
    X_test = np.array(X_test)[perm]

    # Predict the test data using the decoder:
    y_pred = decoder.predict(X_test)

    # Calculate Mean Absolute Error:
    prediction_score = -np.mean(decoder.cv_scores_['beta'])

    # Get the weight image:
    weight_img = decoder.coef_img_['beta']
    
    decoder_results = {
        'y_col': y_col,
        'decoder': decoder,
        'masker': masker,
        'atlas_maps': atlas_maps,
        'atlas_labels': atlas_labels,
        'y_pred': y_pred,
        'y_test': y_test,
        'prediction_score': prediction_score,
        'weight_img': weight_img,
        'analysis_type': analysis_type,
    }

    # Return the results:
    return decoder_results

def decoder_save(decoder_results, main_dir, y_col, analysis_type, estimator_type, atlas):
    """
    Save the decoder data.
    """
    print('Running decoder_save...')
    # Current Date:
    current_date = time.strftime("%m-%d-%Y")

    with open(f'{main_dir}/{y_col}_{analysis_type}_{estimator_type}_{atlas}_{current_date}_decoder_results.pkl', 'wb') as f:
        pickle.dump(decoder_results, f)
    # Save the weight_img as a nifti file to the dir:
    nib.save(decoder_results['weight_img'], f'{main_dir}/{y_col}_{analysis_type}_{estimator_type}_{atlas}_{current_date}_weight_img.nii.gz')

# %%

def loocv_decoder_run(X,
                      y,
                      n_jobs=8,
                      standardize=True,
                      smoothing_fwhm=30,
                      screening_percentile=10):
    """
    Run the decoder using Leave-One-Out Cross-Validation (LOOCV).
    """
    print('Running loocv_decoder_run...')
    # Set up the masker and feature selection:
    nifti_masker = NiftiMasker(standardize=standardize, smoothing_fwhm=smoothing_fwhm, memory='nilearn_cache')
    variance_threshold = VarianceThreshold(threshold=.01)
    decoder = DecoderRegressor(estimator='ridge_regressor',
                               scoring='neg_mean_absolute_error',
                               screening_percentile=screening_percentile, n_jobs=n_jobs)
    # LOOCV setup
    loo = LeaveOneOut()
    scores = []

    for train_index, test_index in loo.split(X):
        # Split the data
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Apply masker and feature selection
        X_masked = nifti_masker.fit_transform(X_train)
        gm_maps_thresholded = variance_threshold.fit_transform(X_masked)
        mask = nifti_masker.inverse_transform(variance_threshold.get_support())
        # Update the mask of the decoder
        decoder.mask = mask
        # Fit and predict with the decoder
        decoder.fit(X_train, y_train)
        y_pred = decoder.predict(X_test)
        # Store the score
        scores.append(-decoder.cv_scores_['beta'])
        # Get the weight img:
        weight_img = decoder.coef_img_['beta']

    # Calculate average performance
    average_score = np.mean(scores)
    print(f'Average explained variance in LOOCV: {average_score}')
    return average_score, weight_img
