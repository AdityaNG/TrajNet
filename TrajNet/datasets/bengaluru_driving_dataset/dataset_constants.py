import glob
import os
from pathlib import Path

str(Path(__file__).parent.absolute())

# ROOT_DATASET_DIR = "dataset"
ROOT_DATASET_DIR = os.path.join(
    str(Path(__file__).parent.parent.absolute()), "dataset"
)
ROOT_DATASET_DIR = os.path.join(os.path.expanduser("~/Datasets"), "dataset")
print(ROOT_DATASET_DIR)
DATASET_DIR = os.path.join(ROOT_DATASET_DIR, "android")
DATASET_LIST = sorted(glob.glob(DATASET_DIR + "/*"))
if len(DATASET_LIST) == 0:
    DATASET_LIST = ["dataset/android/"]
PANDA_DIR = os.path.join(ROOT_DATASET_DIR, "panda_logs")
PANDA_LIST = sorted(glob.glob(PANDA_DIR + "/*.csv"))
if len(PANDA_LIST) == 0:
    PANDA_LIST = ["dataset/panda_logs/"]

PANDA_CACHE_DIR = ".panda_cache"
TRAJECTORY_CACHE_DIR = ".trajectory_cache"

WRITE_BUFFER_SIZE = 10

DVR_CAPTURE_FORMAT = "rtsp://{}:{}@{}:{}/cam/realmonitor?channel={}&subtype=0"
DVR_NUM_CAMS = 8
DVR_DIR = os.path.join(ROOT_DATASET_DIR, "dvr_logs")
DVR_SCALE_FACTOR = 0.2
