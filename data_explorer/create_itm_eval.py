import numpy as np
import json
from tqdm import tqdm
import os

def get_answer(d):
    return float(d.get('answer'))

if __name__ == "__main__":
	prediction_path = "/home/zmykevin/fb_intern/exp/mmf_exp/itm_conceptual_captions_visual_bert_35004637/reports/itm_conceptual_captions_run_test_2022-01-29T00:09:15.json"
	with open(prediction_path, "r") as f:
		prediction = json.load(f)
	
	data_path = "/home/zmykevin/fb_intern/data/mmf_data/datasets/cc/defaults/annotations"
	train_data_1 = np.load(os.path.join(data_path, "train_vinvl_sbert_cc_nps_0.npy"), allow_pickle=True)
	#Form detection
	train_dict = {x["image_id"]:x["captions"] for x in train_data_1}
	itm_picked_dict = {}
	for i in range(0,len(prediction), 5):
		image_id = prediction[i]["filename"]
		current_batch = np.array([x["answer"] for x in prediction[i:i+5]])
		current_caps = train_dict[image_id] 
		#sort current_batch
		sorted_orders = np.argsort(current_batch)
		picked_cap = current_caps[sorted_orders[-1]]


		itm_picked_dict[image_id] = picked_cap
		
	#Save the Generated Data
	with open("/home/zmykevin/fb_intern/data/mingyang_data/CC/itm_picked_cap.json", "w") as f:
		json.dump(itm_picked_dict,f)
	# with open("/home/zmykevin/fb_intern/data/mingyang_data/CC/cc_clip_test.json", "r") as f:
	# 	clip_test = json.load(f)

	# # #Load the Corresponding Data from Training Annotation
	# data_path = "/home/zmykevin/fb_intern/data/mmf_data/datasets/cc/defaults/annotations"
	# train_data_1 = np.load(os.path.join(data_path, "train_vinvl_sbert_cc_nps_0.npy"), allow_pickle=True)
	
	# selected_data_1 = []
	# selected_data_2 = []
	# selected_data_3 = []
	# selected_data_4 = []
	# selected_data_5 = []
	# for x in train_data_1:
	# 	image_id = str(x["image_id"])
	# 	if clip_test.get(image_id, None) is not None:
	# 		selected_data_1.append({"image_id": image_id, "captions": [x["captions"][0]]})
	# 		selected_data_2.append({"image_id": image_id, "captions": [x["captions"][1]]})
	# 		selected_data_3.append({"image_id": image_id, "captions": [x["captions"][2]]})
	# 		selected_data_4.append({"image_id": image_id, "captions": [x["captions"][3]]})
	# 		selected_data_5.append({"image_id": image_id, "captions": [x["captions"][4]]})
	# 		# for cap in x["captions"]:
	# 		# 	y = {}
	# 		# 	y["image_id"] = x["image_id"] 
	# 		# 	y["captions"] = [cap]
	# 		# 	selected_data.append(y)
	# 	# print(x["image_id"])
	# 	# print(x["captions"])
	# #save this  data
	# print(len(selected_data_1))
	# with open("/home/zmykevin/fb_intern/data/mmf_data/datasets/cc/defaults/annotations/val_itm_eval_0.npy", "wb") as f:
	# 	np.save(f, selected_data_1)
    