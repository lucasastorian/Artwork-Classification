import json
import numpy as np

import os
import subprocess
import zipfile

WORKING_DIR = os.getcwd()
TRAIN_FILE = 'artworks.zip'

def download_file_from_gcs(source, destination):
    """Download files from GCS to WORKING_DIR/.

    Args:
        source: GCS path to the training data
        destination: GCS path to the validation data.
    Returns:
        The local data paths where the data is downloaded.
    """

    local_file_names = [destination]
    print("Local File Names: ", local_file_names)
    gcs_input_paths = [source]
    print("GCS Input Paths: ", gcs_input_paths)

    raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)
                                  for local_file_name in local_file_names]
    print("Raw Local Files Data Paths: ", raw_local_files_data_paths)

    for i, gcs_input_path in enumerate(gcs_input_paths):
        if gcs_input_path:
            subprocess.check_call(['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

    return raw_local_files_data_paths

def load_data(path):
    """Loads the IMDB Dataset in npz format.

    Args:
        path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')

    Returns:
        A tuple of numpy arrays: '(_train, y_train), (x_test, y_test)'.

    Raises:
        ValueError: In case path is not defined.
    """

    if not path:
        raise ValueError('No training file defined')
    if path.startswith('gs://'):
        download_file_from_gcs(path, destination=TRAIN_FILE)
        path = TRAIN_FILE

    with zipfile.ZipFile(TRAIN_FILE, 'r') as zip_ref:
        zip_ref.extractall('.')
    print('Listing Directory Contents')
    print(os.listdir('.'))