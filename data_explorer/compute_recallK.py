import numpy as np
import json

def i2t(sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i(sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)



#Load the combine the test data split
#Load and combine the test data split
result_path_0 = "/fsx/zmykevin/experiments/sweep_jobs/visualbert_no_pretrain_vinvl_itm_test_zs0..ngpu1/itm_flickr30k_visual_bert_334064/reports/itm_flickr30k_run_test_2021-11-16T14:04:57.json"
result_path_1 = "/fsx/zmykevin/experiments/sweep_jobs/visualbert_no_pretrain_vinvl_itm_test_zs1..ngpu1/itm_flickr30k_visual_bert_43079355/reports/itm_flickr30k_run_test_2021-11-16T13:17:51.json"
result_path_2 = "/fsx/zmykevin/experiments/sweep_jobs/visualbert_no_pretrain_vinvl_itm_test_zs2..ngpu1/itm_flickr30k_visual_bert_46542493/reports/itm_flickr30k_run_test_2021-11-16T13:18:17.json"
result_path_3 = "/fsx/zmykevin/experiments/sweep_jobs/visualbert_no_pretrain_vinvl_itm_test_zs3..ngpu1/itm_flickr30k_visual_bert_22239873/reports/itm_flickr30k_run_test_2021-11-16T13:17:29.json"
result_path_4 = "/fsx/zmykevin/experiments/sweep_jobs/visualbert_no_pretrain_vinvl_itm_test_zs4..ngpu1/itm_flickr30k_visual_bert_22303047/reports/itm_flickr30k_run_test_2021-11-16T13:16:32.json"

with open(result_path_0, "r") as f:
    prediction_raw_0 = json.load(f)

with open(result_path_1, "r") as f:
    prediction_raw_1 = json.load(f)
    
with open(result_path_2, "r") as f:
    prediction_raw_2 = json.load(f)
    
with open(result_path_3, "r") as f:
    prediction_raw_3 = json.load(f)
    
with open(result_path_4, "r") as f:
    prediction_raw_4 = json.load(f)
    
print(len(prediction_raw_0))
print(len(prediction_raw_1))
print(len(prediction_raw_2))
print(len(prediction_raw_3))
print(len(prediction_raw_4))


#Create the combined five predictions and convert the raw prediction to prediction
prediction_raw = prediction_raw_0 + prediction_raw_1 + prediction_raw_2 + prediction_raw_3 + prediction_raw_4

prediction = np.zeros((1000,5000))

break_point = 5000
col_idx = 0
for i, x in enumerate(prediction_raw):
    #print(x)
    row_idx = int(i/break_point)
    #print(row_idx)
    if i%break_point == 0:
        col_idx = 0
    prediction[row_idx,col_idx] = x['answer']
    col_idx += 1

print("image to text retrieval accuracy: ")
print(i2t(prediction))
print("text to image retrieval accuracy: ")
print(t2i(prediction))
