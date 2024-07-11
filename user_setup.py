# This file is used to setup the project. It is executed when the project is imported.
# This file should be used to download all large files (e.g., model weights) and store them to disk.
# In this file, you can also check if the environment works as expected.
# If something goes wrong, you can exit the script with a non-zero exit code.
# This will help you detect issues early on.
#
# Below, you can find some sample code:

import subprocess
import sys
import zipfile
import os
from os.path import join

def download_large_files():
    french = 'https://drive.google.com/file/d/1dxLuwhrN-vfZzB66-ntMq0_AVK2rMUy0/view?usp=drive_link'
    german = 'https://drive.google.com/file/d/1lvSffqiCtMghcCrFqw3VpWDlZWyR9hxl/view?usp=drive_link'

    cwd = join(os.getcwd(), 'dataset')

    gdown.download(french, cwd, quiet=False)
    gdown.download(german, cwd, quiet=False)

    with zipfile.ZipFile(french, 'r') as zip_ref:
        zip_ref.extractall(join(cwd,'french'))
    with zipfile.ZipFile(german, 'r') as zip_ref:
        zip_ref.extractall(join(cwd,'german'))

    return True

def check_environment():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect"])

    return True


if __name__ == "__main__":
    print("Perform your setup here.")
    
    if not check_environment():
        print("Environment check failed.")
        exit(1)
        
    if not download_large_files():
        print("Downloading large files failed.")
        exit(1)
        
    exit(0)