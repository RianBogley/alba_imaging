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
import scipy.stats as stats
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ----------------------------- CUSTOM LIBRARIES ----------------------------- #
from alba_imaging.plotting import html_brain_plot, view_img_plot, glass_brain_plot, mean_abs_err_plot, true_minus_pred_plot
from alba_imaging.importing import import_pickle

# %%
# ---------------------------------------------------------------------------- #
#                                  DEFINE DATA                                 #
# ---------------------------------------------------------------------------- #
# Specify working directory:
main_dir = '/shared/language/language/rbogley/wmaps_decoding/edevhx_project/'

# %%
# ---------------------------------------------------------------------------- #
#                                   PLOTTING                                   #
# ---------------------------------------------------------------------------- #
bg_img = nilearn.datasets.load_mni152_template(resolution=1)

# %%
# Find every pkl file in the main_dir and all its subdirs and add them to a list of pkl_files:
pkl_files = glob.glob(main_dir + '**/*.pkl', recursive=True)
for pkl in pkl_files:
    print(pkl)

# %%
# For each pickle file found, import the pickle file then plot and save the results:
for pickle in pkl_files:
    run_name = os.path.basename(pickle).split('.pkl')[0]
    run_dir = f'{os.path.dirname(pickle)}/'
    print(f'Run name: {run_name}')
    print(f'Run dir: {run_dir}')

    print(f'Importing: {run_name}')
    data = import_pickle(pickle)
    print(f'Finished importing: {pickle}')
    # Show all the variables in the pickle file:
    print(f'Variables in pickle: {pickle}')
    for key in data:
        print(key)
    # Save the weight_img as a nifti:
    nib.save(data['weight_img'], f'{run_dir}/{run_name}_weight_img.nii.gz')
    
    print(f'Plotting results for: {pickle}')
    # Plot HTML view of the results:
    html_brain = html_brain_plot(data['weight_img'], bg_img)
    html_brain.save_as_html(f'{run_dir}/{run_name}_html_brain.html')
    html_brain.save_as_html(f'{main_dir}/{run_name}_html_brain.html')
    # Plot View Img view of the results:
    view_img = view_img_plot(data['weight_img'], bg_img)
    # Plot Glass Brain view of the results:
    glass_brain = glass_brain_plot(data['weight_img'])
    glass_brain.savefig(f'{main_dir}/{run_name}_glass_brain.png', dpi=300)
    # Plot & save the Mean Absolute Error:
    mean_abs_err_plot(y_test_final=data['y_test_final'],y_pred=data['y_pred'],prediction_score=data['prediction_score'],main_dir=main_dir,run_name=run_name)
    # Plot & save the True - Predicted values:
    true_minus_pred_plot(y_test_final=data['y_test_final'],y_pred=data['y_pred'],main_dir=main_dir,run_name=run_name)
    print(f'Finished plotting results for: {pickle}')
    print(f'All figures saved to: {run_dir}')

# %%
