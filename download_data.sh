#!/bin/bash
# Downloads the 5 ExtraSensory files needed for CS60055 Challenge 1.
# Small files first. Big raw files ONE AT A TIME (the website asks this).
set -e
mkdir -p data && cd data

# ---- small files (download today) ----
wget -c http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip
wget -c http://extrasensory.ucsd.edu/data/cv5Folds.zip
wget -c http://extrasensory.ucsd.edu/data/additional_data_files/ExtraSensory.per_uuid_original_labels.zip

# ---- big raw files (run overnight, sequential) ----
wget -c http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip
wget -c http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip

# ---- unzip into clean folders ----
unzip -n ExtraSensory.per_uuid_features_labels.zip -d labels
unzip -n cv5Folds.zip                              -d cv_folds
unzip -n ExtraSensory.per_uuid_original_labels.zip -d original
unzip -n ExtraSensory.raw_measurements.raw_acc.zip   -d raw_acc
unzip -n ExtraSensory.raw_measurements.proc_gyro.zip -d raw_gyro

echo "DONE. Folder layout is inside ./data/"
