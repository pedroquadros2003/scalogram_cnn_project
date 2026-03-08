import os

FOLDER_PATH = "/home/Workspace/GeneratedScalogramsGrayMoreOverlap"

import logging
logger = logging.getLogger(__name__)

def list_files(folder_path):
    # Get all files
    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    return files


if __name__ == "__main__":

    list_files(folder_path=FOLDER_PATH)