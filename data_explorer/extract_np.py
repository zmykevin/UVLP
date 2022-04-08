import spacy
import os
import numpy as np
nlp = spacy.load('en_core_web_sm')
from tqdm import tqdm

def extract_np(text):
    nps = []
    conference_doc = nlp(text)
    for chunk in conference_doc.noun_chunks:
    	#print(str(chunk))
    	nps.append(str(chunk))
        #print (chunk)
    return nps


if __name__ == "__main__":
	#Import the training annotation dataset
	data_path = "/fsx/zmykevin/data/mmf_data/datasets/cc/defaults/annotations"
	#data_annotation = np.load(os.path.join(data_path, "train_vinvl_sbert_retrieved_sorted_all_0.npy"), allow_pickle=True)

	#Iterate through the data annotation
	for id_ in [0,1,2,3,6,7,8,9]:
		data_annotation = np.load(os.path.join(data_path, "train_vinvl_bookcorpus_retrieved_sorted_all_{}.npy".format(id_)), allow_pickle=True)


		new_data_annotation = []
		for d in tqdm(data_annotation):
		    #iterate
		    new_d = d.copy()
		    new_d["noun_phrases"] = []
		    for caption in d['captions']:
		        nps = extract_np(caption)
		        new_d["noun_phrases"].append(nps)
		    new_data_annotation.append(new_d)
		    #break
		#print(new_data_annotation)
		with open(os.path.join(data_path, "train_vinvl_bookcorpus_retrieved_sorted_all_nps_{}.npy".format(id_)), "wb") as f:
			np.save(f, new_data_annotation)
		#break