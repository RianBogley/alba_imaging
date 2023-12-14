# %%
# ---------------------------------------------------------------------------- #
#                      DYSLEXIA CENTER DICOM RE-IDENTIFIER                     #
#                                BY: RIAN BOGLEY                               #
# ---------------------------------------------------------------------------- #
# This is a short script for re-identifying the DICOM metadata of a 
# Dyslexia Center participant's structural scans for use in a clinical referral.
# These re-identified images can then be uploaded to APEX/EPIC for reference
# by a clinician reviewing the research scans. Or, can be burned to a CD using
# the appropriate method (e.g. from a Siemens Scanner) such that the disc
# has DICOM reading software included.
# 
# This script will prompt you for the Participant's PIDN, First & Last Name, 
# and the MRI Date (MM-DD-YYYY), then will replace the DICOM metadata with the
# participant's real name, and it will replace the PatientID field with a 
# standard 1234567890 string (in lieu of an MRN), and it will clear the
# AccessionNumber field. Then it will output the new DICOM files to the
# Clinical Referrals folder in the L drive for use in the referral.
# 
# NOTE: Prior to running this script, please check the following:
# 1. MAKE SURE YOU RUN THIS SCRIPT LOCALLY ON A PROTECTED COMPUTER,
#   AND ON THE UCSF-WPA NETWORK.
# 2. Make sure you are connected to the L drive.
# 3. Check that the main filepaths below are correct to your system.
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
import pydicom
import os
import shutil

# THIS SCRIPT IS FOR RE-IDENTIFYING DYSLEXIA CENTER
# DICOM FILES FOR USE IN CLINICAL REFERRALS
# IT WILL CHANGE THE PATIENT NAME BACK TO THE PATIENT'S REAL NAME
# AND CLEAR THE PATIENT ID / ACCSESION NUMBER

# NOTE: DO NOT SAVE ANY PHI IN THIS SCRIPT!!!!!!!!

# %%
# Path to the folder containing the DICOM files
clinicalreferrals_dir = "/Volumes/language/language/Dyslexia_project/ClinicalReferrals/"
dyslexiascans_dir = "/Volumes/language/language/Dyslexia_project/Participants/"

# Ask for the patient's name:
print("IMPORTANT: PLEASE ENTER ALL THE FOLLOWING INFORMATION ACCURATELY!")
pidn = input("Enter the patient's PIDN: ")
first_name = input("Enter the patient's first name: ")
last_name = input("Enter the patient's last name: ")
dcdate = input("Enter the patient's date of scan (MM-DD-YYYY): ")

# Modify paths
patient_dir = os.path.join(clinicalreferrals_dir, pidn)
# %%
# Change DCDate into the "YYYYMMDD" format from the "MM-DD-YYYY" format:
dcdate = dcdate[6:10] + dcdate[0:2] + dcdate[3:5]

scans_dir = os.path.join(dyslexiascans_dir, pidn, dcdate)

# Check if the path exists, if not create a folder with the PIDN as the name in the dir:
if not os.path.exists(patient_dir):
    os.mkdir(patient_dir)
    
# Copy over any subdirectories that start with the following strings from scans_dir to patient_dir:
scan_types = [
    't1_mp2rage_jose_UNI_Images',
    'T1_mprage_ND',
    't2_flair_sag_p3_ND',
    't2_space_sag_iso_p2_ND',
    'dti_2mm_m3p2_b2500_96dir_10b0s_TRACEW',
    'dti_2mm_m3p2_b2500_96dir_10b0s_ADC',
    ]

# %%
# Save the subdirs of scans_dir that start with the strings in scan_types to a list:
subdirs = []
for dirpath, dirnames, filenames in os.walk(scans_dir):
    for dirname in [d for d in dirnames if d.startswith(tuple(scan_types))]:
        subdirs.append(os.path.join(dirpath, dirname))
        print(subdirs)

# %%
# Copy the subdirs to the patient_dir with any ".dcm" files in them too, but no other files:
for subdir in subdirs:
    # Copy the subdir to the patient_dir:
    new_dir = os.path.join(patient_dir, os.path.basename(subdir))
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    # Copy over any files that end in .dcm from the subdirectories in scans_dir to the new_dir in patient_dir:
    for filename in [f for f in os.listdir(subdir) if f.endswith(".dcm")]:
        old_file = os.path.join(subdir, filename)
        new_file = os.path.join(new_dir, filename)
        shutil.copyfile(old_file, new_file)
    # Check that all files were copied succesfully, if not, print the files that were not copied:
    if len(os.listdir(subdir)) != len(os.listdir(new_dir)):
        print("The following files were not copied succesfully:")
        print([f for f in os.listdir(subdir) if f not in os.listdir(new_dir)])

# Find all dicom files in the specified directory, in all subdirectories:
dicom_files = []
for dirpath, dirnames, filenames in os.walk(patient_dir):
    for filename in [f for f in filenames if f.endswith(".dcm")]:
        dicom_files.append(os.path.join(dirpath, filename))
        print(dicom_files)

for subdir in os.listdir(patient_dir):
    print(f"There are {len(os.listdir(os.path.join(patient_dir, subdir)))} dicom files in {subdir}")

# Edit the metadata of the dicom files to change the patient
# name and PatientID
for dicom_file in dicom_files:
    ds = pydicom.dcmread(dicom_file)
    ds.PatientName = last_name + '^' + first_name
    # Assign a random PatientID in lieu of a real MRN:
    ds.PatientID = '1234567890'
    ds.AccessionNumber = ''
    # ds.StudyDate = dcdate
    ds.save_as(dicom_file)

for dicom_file in dicom_files:
    ds = pydicom.dcmread(dicom_file)
    print(ds.PatientName, ds.PatientID, ds.AccessionNumber, ds.StudyDate)

# %%
# Clear PIDN, First_name, and Last_name:
pidn = ''
first_name = ''
last_name = ''
dcdate = ''

# %%
