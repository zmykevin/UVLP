import json
import os

data_path = "/home/zmykevin/fb_intern/data/mingyang_data/CC/clip_scores"
clip_itm_picked_name = "cc_clip_score_itm_picked.json"

clip_retrieved_all = []
for i in range(5):
	with open(os.path.join(data_path, "cc_clip_score_sentencebert_sorted_{}.json".format(i)), "r") as f:
		current_clip_retrieved_data = json.load(f)
		clip_retrieved_all.append(current_clip_retrieved_data)
clip_score_retrieved = {}
for k in clip_retrieved_all[0].keys():
	clip_score_retrieved[k] = [clip_retrieved_all[j][k] for j in range(5)]

with open(os.path.join(data_path, "cc_clip_score_itm_picked.json"), "r") as f:
	clip_itm_picked = json.load(f)

ave_mean = 0
larger_than_mean = 0
over_median = 0
for k in clip_itm_picked.keys():
	clip_score_retrieved[k].sort()
	mean_score = sum(clip_score_retrieved[k])/5
	if clip_itm_picked[k] > mean_score:
		larger_than_mean += 1
	if clip_itm_picked[k] >= clip_score_retrieved[k][2]:
		over_median += 1
	ave_mean += mean_score


ave_mean = ave_mean/len(clip_itm_picked)
print(ave_mean)
print(larger_than_mean)
print(larger_than_mean/len(clip_itm_picked))
print(over_median)
# clip_retrieved_0_name = "cc_clip_score_sentencebert_sorted.json"
# clip_retrieved_1_name = "cc_clip_score_sentencebert_sorted_1.json"
# clip_retrieved_2_name = "cc_clip_score_sentencebert_sorted_2.json"
# clip_retrieved_3_name = "cc_clip_score_sentencebert_sorted_1.json"
# data_list = []
# for x in range(5):
# 	if x == 0:
# 		current_data = json.load(open(os.path.join(data_path, "{}.json".format(clip_score_file_name)), "r"))
# 	else:
# 		current_data = json.load(open(os.path.join(data_path, "{}_{}.json".format(clip_score_file_name, x)), "r"))
# 	data_list.append(current_data)

# not_top1 = 0
# max_score_list = []
# mean_score_list = []
# for k, v in data_list[0].items():
# 	score_list = [d[k] for d in data_list if d.get(k, None) is not None]
# 	top1_score = score_list[0]
# 	score_list.sort(reverse=True)
# 	max_score = score_list[0]
# 	mean_score = sum(score_list)/len(score_list)
# 	if max_score != top1_score:
# 		not_top1 += 1
# 	max_score_list.append(max_score)
# 	mean_score_list.append(mean_score)

# print(not_top1)
# print(sum(max_score_list)/len(max_score_list))
# print(sum(mean_score_list)/len(mean_score_list))


