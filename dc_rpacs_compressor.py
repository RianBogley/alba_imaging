# %%
# ---------------------------------------------------------------------------- #
#              DYSLEXIA CENTER SCANS COMPRESSOR FOR RPACS UPLOADS              #
#                                BY: RIAN BOGLEY                               #
# ---------------------------------------------------------------------------- #
# This is a short script for copying the structural DICOMs of recently
# scanned/acquired Dyslexia Center cases stored in the L Drive participants
# directory to the Research Radiology PACS (rPACS) folder, then compressing the
# files into a tar-gz file to be uploaded to the rPACS server for the Dyslexia
# Center's neuroradiologists to review them for incidental findings.
# 
# NOTE: Prior to running this script, please check the following:
# 1. Make sure you are connected to the L drive.
# 2. Check that the main filepaths below are correct to your system.
# 3. Check that all recently acquired scans still have the dc, adys, or leegt
#   prefixes prior to copying. Having these prefixes still is how this script
#   identifies recently acquired scans vs old ones.
# 4. Check each case's scan notes (on REDCap) for any rejected scans. For any
#   scans that were repeated, add the string "reject_" as a prefix to the bad
#   scan's folder name to prevent it from being copied.
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
import tarfile
import os
import shutil
# ---------------------------------------------------------------------------- #
# %% ##########################################################################
# Set Variables:
# Original scans directory filepath:
scans_dir = 'L:/language/Dyslexia_project/Participants/'
# rPACS upload directory filepath:
rPACS_dir = 'L:/language/Dyslexia_project/ResearchRadiologyPACS/Not_Uploaded/'
# Scan prefixes to be copied:
scan_prefixes = ('dc', 'adys', 'leegt')
# Scan types to be copied:
scan_types = [
    't1_mp2rage_jose_UNI_Images',
    'T1_mprage_ND',
    't2_flair_sag_p3_ND',
    't2_space_sag_iso_p2_ND',
    'dti_2mm_m3p2_b2500_96dir_10b0s_TRACEW',
    'dti_2mm_m3p2_b2500_96dir_10b0s_ADC',
    ]

###############################################################################
# %% ##########################################################################
# Find all folders in scans_dir that begin with specified prefixes
# i.e. 'DC', 'ADYS', or 'LEEGT' (not case sensitive).
# Then, make a copy of just the folder name in a list:
new_scans = [f for f in os.listdir(scans_dir) 
if f.lower().startswith(scan_prefixes)]

# Check if list of found scans in new_scans already exist and are 
# compressed in rPACS_dir. If so, remove them from the list:
for root, dirs, files in os.walk(rPACS_dir):
    for file in files:
        if file.endswith('.tgz'):
            if file[:-4] in new_scans:
                print('Case: ' + file[:-4] + 
                ' is already compressed in rPACS_dir, skipping.')
                new_scans.remove(file[:-4])

# Print which remaining scans will be copied:
print('The following scans will be copied:')
for scan in new_scans:
    print(scan)
###############################################################################
# %% ##########################################################################
# For each case in the list, make a new folder in rPACS_dir:
for new_scan in new_scans:
    os.mkdir(rPACS_dir + new_scan)
    # Then, copy any specified scan subfolders from the original scans_dir
    # to the new folder in rPACS_dir unless it contains the string "reject":
    for root, dirs, files in os.walk(scans_dir + new_scan):
        for dir in dirs:
            if any(s in dir for s in scan_types):
                if 'reject' not in dir.lower():
                    shutil.copytree(os.path.join(root, dir), 
                    os.path.join(rPACS_dir + new_scan, dir))
    # Remove any potentially created NifTI files (ending in .nii or .nii.gz)
    # from the new folder to leave only the DICOM files for upload:
    for root, dirs, files in os.walk(rPACS_dir + new_scan):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                os.remove(os.path.join(root, file))
    # Compress the newly created folder into a tar.gz file (.tgz):
    with tarfile.open(rPACS_dir + new_scan + '.tgz', 'w:gz') as tar:
        tar.add(rPACS_dir + new_scan, arcname=new_scan)
    # Use shutil to remove the copied folder and all its contents:
    shutil.rmtree(rPACS_dir + new_scan)
# %% ##########################################################################
# If any tar.gz files in the rPACS_dir are over 80MB, print name and file size:
# Compressed files over 80MB will generally not upload properly to rPACS.
# If any are found, please check them for accidentally copied NifTI files,
# duplicated scans, or if neither of those are the case, remove the scan(s)
# with the lowest priority for radiology review. The order of priority of scans
# to delete is the same as the order they are listed in scan_types above 
# (with the scans at the top of the list being the most important to keep,
# and the scans at the bottom of the list being the least important to keep).
for root, dirs, files in os.walk(rPACS_dir):
    for file in files:
        if file.endswith('.tgz'):
            if os.path.getsize(rPACS_dir + file) > 80000000:
                print('File: ' + file + ' is over 80MB (',str(
                    os.path.getsize(rPACS_dir + file)/1000000
                    ) ,'MB), please check it.')
###############################################################################