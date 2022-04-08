import numpy as np
import json
import os
import os.path as op
import logging
from tqdm import tqdm
import faiss

import csv
from datasets import load_dataset

def save_csv(visualization, output_path):
    csv_columns = ['original_id', "original_obj", "original_caption", "retrieved_1", "retrieved_2", "retrieved_3"]
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in visualization:
            writer.writerow(data)
    
def save_json(closest_captions, output_path):
    with open(output_path, "w") as f:
        json.dump(closest_captions, f)


if __name__ == "__main__":
	with open("/data/home/zmykevin/vinvl_data/CC/cc_objects_captions.json", "r") as f:
		cc_objects_captions = json.load(f)

	#load the bookcorpus
	#bookcorpus = load_dataset("bookcorpus")["train"]
	bookcorpus = []
	with open("/data/home/zmykevin/vinvl_data/book_corpus/bc1g.doc", "r", encoding="utf-8") as f:
	    all_text = f.read().lower()
	    bookcorpus.extend([l.strip('\n').strip('\r').strip('\n') for l in all_text.split("\n")])


	cc_book_corpus_captions = {}

	id_list = []
	# for k, v in cc_objects_captions.items():
	# 	id_list.append(k)
	# 	cc_book_corpus_captions[k] = {"caption":v["caption"], "objects_no_rep":v["objects_no_rep"]}


	#current_index = int(id_list[-1]) +1
	current_index = 0
	for i, x in enumerate(bookcorpus):
		id_list.append(str(current_index))
		cc_book_corpus_captions[str(current_index)] = {"caption": x, "objects_no_rep": None}
		current_index += 1
		# if i == cut_off_threshold:
		# 	break

	#dump the data to f
	# with open("/data/home/zmykevin/vinvl_data/CC/bookcorpus_sentences.json", "w") as f:
	# 	json.dump(cc_book_corpus_captions, f)
    
	# id_list = []
	# object_list = []
	# caption_list = []
	# for k, v in cc_objects_captions.items():
	#     id_list.append(k)
	#     object_list.append(v['objects_no_rep'])
	#     caption_list.append(v['caption'])

	# print("finish loading the captions")

	#Load the npy file
	caption_embedding = np.load("/data/home/zmykevin/vinvl_data/CC/BC_sentence_BertEmbedding_sorted.npy")
	# #caption_cc_embedding = np.load("/data/home/zmykevin/vinvl_data/CC/CC_caption_BertEmbedding_sorted.npy")
	# caption_bookcorpus_embedding = np.load("/data/home/zmykevin/vinvl_data/book_corpus/bookcorpus_caption_BertEmbedding.npy")
	object_embedding = np.load("/data/home/zmykevin/vinvl_data/CC/CC_objects_BertEmbedding_sorted.npy")

	# #concatenate the caption_emebdding
	# #caption_embedding = np.concatenate((caption_cc_embedding, caption_bookcorpus_embedding), axis=0)
	# caption_embedding = caption_bookcorpus_embedding
	print('finish loading the embeddings')
    
	# # print(len(id_list))
	# # print(caption_embedding.shape[0]) 
	# assert len(id_list)==caption_embedding.shape[0]

    #prepare index
	d = caption_embedding.shape[1]
	#index = faiss.IndexFlatL2(d)
	res = faiss.StandardGpuResources()
	index_flat = faiss.IndexFlatL2(d)
	gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
	#make it into gpu index

	gpu_index_flat.add(caption_embedding)

	# # #Find all the closest repository
	output_path = "/data/home/zmykevin/vinvl_data/CC"
	visualization = []
	closest_captions = {}
	error_ids = []
	batch_size = 500
	save_checkpoint = 1000000
	k = 100

	checkpoint_index = 0
	for i in tqdm(range(0, object_embedding.shape[0], batch_size)):
	    valid_batch_size = batch_size if i+batch_size < object_embedding.shape[0] else object_embedding.shape[0]-i
	    D,I = gpu_index_flat.search(object_embedding[i:i+valid_batch_size], k)
	    #Get the Index Matrix
	    
	    for j in range(valid_batch_size):
	        try:
		        original_id = id_list[j + i]
		        right_side = [id_list[x] for x in I[j]]

		        current_output = {}
		        current_output['original_id'] = original_id
		        current_output['original_obj'] = cc_book_corpus_captions[original_id]['objects_no_rep']
		        current_output['original_caption'] = cc_book_corpus_captions[original_id]['caption']
		        current_output['retrieved_1'] = cc_book_corpus_captions[right_side[0]]['caption']
		        current_output['retrieved_2'] = cc_book_corpus_captions[right_side[1]]['caption']
		        current_output['retrieved_3'] = cc_book_corpus_captions[right_side[2]]['caption']

		        #store the other info into json
		        closest_captions[original_id] = right_side
		        visualization.append(current_output)
	        
            #break
            # current_output['original_obj'] = cc_objects_captions[original_id]['objects_no_rep']
            # current_output['original_caption'] = cc_objects_captions[original_id]['caption']
            # current_output['retrieved_1'] = cc_objects_captions[right_side[0]]['caption']
            # current_output['retrieved_2'] = cc_objects_captions[right_side[1]]['caption']
            # current_output['retrieved_3'] = cc_objects_captions[right_side[2]]['caption']
	        except:
	            print("The bug occurs at {}".format(i+j))
	            print("The error id is: {}".format(original_id))
	            error_ids.append(original_id)
	    
	    if int((i+valid_batch_size)/save_checkpoint) > checkpoint_index:
	        print("save the checkpoint for: {}".format(i+valid_batch_size))
	        save_json(closest_captions, os.path.join(output_path, "captions_retrieved_bertembedding_bookcorpus_sorted_top100_backup.json")) 
	        save_csv(visualization, os.path.join(output_path, "captions_retrieved_bertembedding_bookcorpus_sorted_top100_backup.csv"))
	        save_json(error_ids, os.path.join(output_path, "captions_retrieved_bertembedding_bookcorpus_sorted_top100_error_ids_backup.json"))
	        checkpoint_index +=1

	#save the json file
	print("finish retrieving and start to save the json and checkpoints. ")
	save_json(closest_captions, os.path.join(output_path, "captions_retrieved_bertembedding_bookcorpus_sorted_top100.json"))   
	#save the csv file
	save_csv(visualization, os.path.join(output_path, "captions_retrieved_bertembedding_bookcorpus_sorted_top100.csv"))
	save_json(error_ids, os.path.join(output_path, "captions_retrieved_bertembedding_sorted_error_ids_top100.json"))
