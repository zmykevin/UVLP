import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

data_path = "/fsx/zmykevin/data/mmf_data/datasets/cc/defaults/annotations"
#data_index_range = [3,6,7,8,9]
data_index_range = [0,1,2,3,6,7,8,9]
for id_ in data_index_range:

	data_annotation = np.load(os.path.join(data_path, "train_vinvl_bookcorpus_retrieved_sorted_all_nps_{}.npy".format(id_)), allow_pickle=True)
	# for i, x in enumerate(data_annotation):
	# 	print(x)
	# 	if i > 5:
	# 		break
	# break

	new_data_annotation = []
	for sample_data in tqdm(data_annotation):
		#sample_data = data_annotation[i]

		objects = sample_data["objects"]
		objects_norep = list(set(objects))
		ob_vecs = model.encode(objects_norep)

		#create a new data
		new_data = sample_data.copy()
		new_data["noun_phrases"] = []
		new_data["objects_norep"] = objects_norep

		noun_phrases = []
		for noun_phrase in sample_data["noun_phrases"]:
			noun_phrases += noun_phrase
		#print(noun_phrases)
		if not noun_phrases:
			continue
		np_vecs = model.encode(noun_phrases)
		if len(noun_phrases) == 1:
			np.expand_dims(np_vecs,axis=0)
		ob_vecs = model.encode(objects_norep)
		np_cosine_sim = cosine_similarity(np_vecs, ob_vecs)
		#print(np_cosine_sim.shape)
		#reassign
		index = 0
		for noun_phrase in sample_data["noun_phrases"]:
			np_sim_list = []
			for word in noun_phrase:
				np_sim_list.append({word: np_cosine_sim[index]})
				index += 1
			new_data["noun_phrases"].append(np_sim_list)

		new_data_annotation.append(new_data)
		#break

	print("finish generating id: {}".format(id_))
	with open(os.path.join(data_path, "train_vinvl_bookcorpus_retrieved_sorted_all_nps_{}.npy".format(id_)), "wb") as f:
			np.save(f, new_data_annotation)






	
