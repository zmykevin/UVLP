import numpy as np
import json
import os
import os.path as op
import logging
from tqdm import tqdm
from numba import jit
import sister


from datasets import load_dataset

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#bert_embedding = sister.BertEmbedding(lang="en")

with open("/data/home/zmykevin/vinvl_data/CC/cc_objects_captions_sorted.json", "r") as f:
    cc_objects_captions = json.load(f)

#Load the Book Corpus
# bookcorpus = load_dataset("bookcorpus")
bookcorpus = []
with open("/data/home/zmykevin/vinvl_data/book_corpus/bc1g.doc", "r", encoding="utf-8") as f:
    all_text = f.read().lower()
    bookcorpus.extend([l.strip('\n').strip('\r').strip('\n') for l in all_text.split("\n")])
print(len(bookcorpus))

caption_list = bookcorpus
print("finish loading the captions")

# id_list = []
# object_list = []
# caption_list = []
# for k, v in cc_objects_captions.items():
#     id_list.append(k)
#     object_list.append(v['objects_no_rep'])
#     caption_list.append(v['caption'])

# sample_number = 2500000
# for i,v in tqdm(enumerate(bookcorpus["train"])):
#     caption_list.append(v['text'])
#     if i == sample_number:
#         break


batch_size = 100
save_checkpoint = 500000
#output_path = "/data/home/zmykevin/vinvl_data/book_corpus"
output_path = "/data/home/zmykevin/vinvl_data/CC"

#current_embedding = np.load(os.path.join(output_path, "CC_caption_BertEmbedding_sorted_backup.npy"))
# current_obj_embedding = np.load(os.path.join(output_path, "CC_objects_BertEmbedding_sorted_backup.npy"))
#print("Load the Previous Checkpoints")

current_embedding = None
#current_obj_embedding = None
#current_embedding_list = [current_embedding]
# current_obj_embedding_list = [current_obj_embedding]
current_embedding_list = []
# current_obj_embedding_list = []
#start_position = current_embedding.shape[0]
start_position = 0
checkpoint_index = 0
for i in tqdm(range(start_position, len(caption_list), batch_size)):
    valid_batch_size = batch_size if i+batch_size < len(caption_list) else len(caption_list)-i
    cap = caption_list[i:i+valid_batch_size]
    cap_vec = model.encode(cap)

    # obj = object_list[i:i+valid_batch_size]
    # obj_vec = model.encode(obj)
    current_embedding_list.append(cap_vec)
    # current_obj_embedding_list.append(obj_vec)
        
    if int((i+valid_batch_size)/save_checkpoint) > checkpoint_index:
        print("Save checkpoint at: {} steps".format(i))
        current_save_embedding = np.concatenate(current_embedding_list, axis=0)
        # current_save_obj_embedding = np.concatenate(current_obj_embedding_list, axis=0)
        # with open(os.path.join(output_path, "bookcorpus_caption_BertEmbedding_backup.npy"), "wb") as f:
        #     np.save(f, current_save_embedding)
        with open(os.path.join(output_path, "BC_sentence_BertEmbedding_backup.npy"), "wb") as f:
            np.save(f, current_save_embedding)

        # with open(os.path.join(output_path, "CC_objects_BertEmbedding_sorted_backup.npy"), "wb") as f:
        #     np.save(f, current_save_obj_embedding)
        current_embedding_list.clear()
        # current_obj_embedding_list.clear()
        current_embedding_list.append(current_save_embedding)
        # current_obj_embedding_list.append(current_save_obj_embedding)
        checkpoint_index +=1
        #break
        
current_save_embedding = np.concatenate(current_embedding_list, axis=0)
# with open(os.path.join(output_path, "bookcorpus_caption_BertEmbedding.npy"), "wb") as f:
#     np.save(f, current_save_embedding)

with open(os.path.join(output_path, "BC_sentence_BertEmbedding_sorted.npy"), "wb") as f:
    np.save(f, current_save_embedding)

# current_save_obj_embedding = np.concatenate(current_obj_embedding_list, axis=0)
# with open(os.path.join(output_path, "CC_objects_BertEmbedding_sorted.npy"), "wb") as f:
#     np.save(f, current_save_obj_embedding)