'''
    This script downloads the data used in the cvpr paper
    Make sure to run this script from the top-level folder of the repository
    i.e.,
        python scripts/download_data.py
'''

#### Standard Library Imports
import zipfile
import os
import sys
sys.path.append('.')
sys.path.append('..')

#### Library imports
import gdown

#### Local imports
from research_utils.io_ops import load_json

if __name__=='__main__':

    ## 
    io_dirpaths = load_json('io_dirpaths.json')
    data_base_dir = io_dirpaths['data_base_dirpath']
    print(data_base_dir)

    ## Output dir where to store the dataset (leaving default should work)
    os.makedirs(data_base_dir, exist_ok=True)
    
    ## dataset folder ID
    dataset_dir = data_base_dir

    ## Get the url 
    # gdrive_dataset_folder_url = "https://drive.google.com/drive/folders/1vJsZxGgXxh1s-wLIpSIcFg0hPx8pDJea"
    gdrive_dataset_folder_url = "https://drive.google.com/drive/folders/1afCoNUpSS1VjGQVh-GhPTQHtOD0LoAMq"
    
    
    zip_fpaths = gdown.download_folder(url=gdrive_dataset_folder_url, output=dataset_dir, quiet=False)

