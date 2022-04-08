#!/usr/bin/env python
# coding: utf-8


import numpy as np
import json
import os
import os.path as op
import logging
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

import time
import csv

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

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

def load_csv(csv_path):
    loaded_data = []
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                header_info = row
            else:
                current_data_point = {}
                for j, x in enumerate(row):
                    current_data_point[header_info[j]] = x
                loaded_data.append(current_data_point)
    return loaded_data


with open("/data/home/zmykevin/vinvl_data/CC/cc_objects_captions_sorted.json", "r") as f:
    cc_objects_captions = json.load(f)

print("finish loading the captions")

id_list = []
object_list = []
caption_list = []
for k, v in cc_objects_captions.items():
    id_list.append(k)
    object_list.append(v['objects_no_rep'])
    caption_list.append(v['caption'])


all_list = object_list + caption_list


# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(all_list)

print("compute tfidf_matrix")

# print(tfidf_matrix.shape)


object_tfidf = sparse.csr_matrix(tfidf_matrix[:3116254])
caption_tfidf = sparse.csr_matrix(tfidf_matrix[3116254:])


#Start to comute the cosine similarity
batch_size = 1000
save_checkpoint = 100000
closest_captions = {}
visualization = []
#
#Define output path
output_path = "/data/home/zmykevin/vinvl_data/CC"
#closest_captions = json.load(open(os.path.join(output_path, "captions_retrieved.json"),"r"))
#visualization = load_csv(os.path.join(output_path, "captions_retrieved.csv"))
#starting_index = len(visualization)
starting_index = 0
error_ids = []

checkpoint_index = 0

for i in tqdm(range(starting_index,caption_tfidf.shape[0], batch_size)):
    valid_batch_size = batch_size if i+batch_size < caption_tfidf.shape[0] else caption_tfidf.shape[0]-i
    cosine_sim = awesome_cossim_top(object_tfidf[i:i+valid_batch_size], caption_tfidf.T, 10, 0)
    non_zeros = cosine_sim.nonzero()
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    for index in range(valid_batch_size):
        try:
            original_id = id_list[index + i]
            right_side = [id_list[x] for x in sparsecols[index*10:index*10+10]]
            current_output = {}
            current_output['original_id'] = original_id
            current_output['original_obj'] = cc_objects_captions[original_id]['objects_no_rep']
            current_output['original_caption'] = cc_objects_captions[original_id]['caption']
            current_output['retrieved_1'] = cc_objects_captions[right_side[0]]['caption']
            current_output['retrieved_2'] = cc_objects_captions[right_side[1]]['caption']
            current_output['retrieved_3'] = cc_objects_captions[right_side[2]]['caption']
            #store the other info into json
            closest_captions[original_id] =right_side
            visualization.append(current_output)
        except:
            print("The bug occurs at {}".format(i+index))
            print("The error id is: {}".format(original_id))
            error_ids.append(original_id)

    if int((i+valid_batch_size)/save_checkpoint) > checkpoint_index:
        print("save the checkpoint for: {}".format(i+valid_batch_size))
        save_json(closest_captions, os.path.join(output_path, "captions_retrieved_sorted_backup.json")) 
        save_csv(visualization, os.path.join(output_path, "captions_retrieved_sorted_backup.csv"))
        save_json(error_ids, os.path.join(output_path, "captions_retrieved_sorted_error_ids.json"))
        checkpoint_index +=1


#save the json file
print("finish retrieving and start to save the json and checkpoints. ")
save_json(closest_captions, os.path.join(output_path, "captions_retrieved_sorted.json"))   
#save the csv file
save_csv(visualization, os.path.join(output_path, "captions_retrieved_sorted.csv"))
save_json(error_ids, os.path.join(output_path, "captions_retrieved_sorted_error_ids.json"))








