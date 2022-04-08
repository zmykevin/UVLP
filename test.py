import numpy as np
import json

#Import Caption Objects
vinvl_data_path="/data/home/zmykevin/vinvl_data/CC/cc_objects_captions.json"
with open(vinvl_data_path, "r") as f:
    vinvl_data = json.load(f)

data_path = "/fsx/zmykevin/data/mmf_data/datasets/cc/defaults/annotations/val_vinvl.npy"
data = np.load(data_path, allow_pickle=True)
print(data[0])



for i, v in vinvl_data.items():
#     # print(i)
#     # print(v)

    if i == str(data[0]["image_id"]):
        print(v)
        print(i)
        break
if i == len(vinvl_data):
    print("not found")