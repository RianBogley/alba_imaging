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
clinicalreferrals_dir = "L:/language/Dyslexia_project/ClinicalReferrals/"
dyslexiascans_dir = "L:/language/Dyslexia_project/Participants/"

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
# Find any subdirectories in scans_dir that start with the strings in scan_types and copy them to patient_dir:
for scan_type in scan_types:
    for dirpath, dirnames, filenames in os.walk(scans_dir):
        for dirname in [d for d in dirnames if d.startswith(scan_type)]:
            shutil.copytree(os.path.join(dirpath, dirname), os.path.join(patient_dir, dirname))

# Remove any files in the newly copied subdirectories that end in .nii or .gz:
for dirpath, dirnames, filenames in os.walk(patient_dir):
    for filename in [f for f in filenames if f.endswith('.nii') or f.endswith('.gz')]:
        os.remove(os.path.join(dirpath, filename))

# Find all dicom files in the specified directory, in all subdirectories:
dicom_files = []
for dirpath, dirnames, filenames in os.walk(patient_dir):
    for filename in [f for f in filenames if f.endswith(".dcm")]:
        dicom_files.append(os.path.join(dirpath, filename))

dicom_files

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

# Clear PIDN, First_name, and Last_name:
pidn = ''
first_name = ''
last_name = ''
dcdate = ''