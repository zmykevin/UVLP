import numpy as np 
import os

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

# def extract_vision_vector():

total_batch=42
sample_num = 50000
#hidden vectors directory
#nmi_dir = "/fsx/zmykevin/experiments/nmi/mu_vla"
# nmi_dir = "/fsx/zmykevin/experiments/nmi/ucla"
# nmi_dir = "/fsx/zmykevin/experiments/nmi/random"
nmi_dir = "/fsx/zmykevin/experiments/nmi/paired"
all_files = os.listdir(nmi_dir)

hidden_prefix = "hidden_"
image_val_prefix = "image_val_"
text_val_prefix = "text_val_"

text_vecs = []
vision_vecs = []
for i in tqdm(range(total_batch)):
	hidden_file = os.path.join(nmi_dir,hidden_prefix+str(i+1)+".npy")
	image_val_file = os.path.join(nmi_dir,image_val_prefix+str(i+1)+".npy")
	text_val_file = os.path.join(nmi_dir, text_val_prefix+str(i+1)+".npy")

	#load the three file
	hidden=np.load(hidden_file)
	image_val=np.load(image_val_file)
	text_val=np.load(text_val_file)

	# print(hidden.shape)
	# print(len(image_val))
	# print(len(text_val))
    
	#extract the vision features
	text_hidden = hidden[:,:60,:]
	vision_hidden = hidden[:,-100:,:]
	#print(text_hidden.shape)
	#print(vision_hidden.shape)
	for t_x, val_t in zip(text_hidden,text_val):
		valid_t_x = t_x[:val_t]
		text_vecs.append(valid_t_x)
		
	for v_x, val_v in zip(vision_hidden,image_val):
		valid_v_x = v_x[:val_v]
		vision_vecs.append(valid_v_x)

#Finish load the vecs
print("Finish load the vecs")
vision_vecs = np.concatenate(vision_vecs,axis=0)[:sample_num]
text_vecs = np.concatenate(text_vecs,axis=0)[:sample_num]

all_vecs = np.concatenate([vision_vecs,text_vecs],axis=0)
print(all_vecs.shape)
#compute k-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(all_vecs)
print("Finish Compute KMeans")
kmeans_labels = kmeans.labels_
print(kmeans_labels[:1000])
gt_labels = np.concatenate([np.zeros(sample_num,dtype=np.int),np.ones(sample_num,dtype=np.int)])
#print(gt_labels.shape)
print(gt_labels[:1000])
#compute nmi
print("nmi score is: ")
print(normalized_mutual_info_score(gt_labels, kmeans_labels))
