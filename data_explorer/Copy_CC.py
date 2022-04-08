import os
import json
from os import path
from tqdm import tqdm

from shutil import copy

#Define Image Path
CC_Train_Image_Path = "/data/home/lyuchen/viltdata/large_experiments/cmd/cc/training"
CC_Val_Image_Path = "/data/home/lyuchen/viltdata/large_experiments/cmd/cc/validation"


#Target Folder
target_folder = "/data/home/zmykevin/project/CC_image"

with open("/data/home/zmykevin/vinvl_data/CC/cc_clip_test.json", "r") as f:
        clip_test = json.load(f)
for k,v in tqdm(clip_test.items()):
    images = []
    texts = []
    
    current_image_path = os.path.join(CC_Train_Image_Path, str(v['cc_id'])) if path.exists(os.path.join(CC_Train_Image_Path, str(v['cc_id']))) else os.path.join(CC_Val_Image_Path, str(v['cc_id']))
    assert path.exists(current_image_path)
    
    #copy the image to target folder
    copy(current_image_path, target_folder)