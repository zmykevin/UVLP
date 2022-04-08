import json
import numpy as np
import os
from tqdm import tqdm
import os.path as op
import logging

class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=False):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file. 
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            generate_lineidx_file(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            logging.info('{}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.info('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

annotation = "/data/home/zmykevin/vinvl_data/CC/model_0060000/annotations/0/dataset_cc.json"
annotation_data = json.load(open(annotation, "r"))
annotation_split = {x['imgid']:x['split'] for x in annotation_data['images']}
# # annotation_item = {x['imgid']:x for x in annotation_data['images']}


		
# print(len(validation_list))
start_index = 8
id_range = [start_index] #WHere validation is located
valid_ids = {}
count = 0
for id_ in id_range: 
    print("Create the features for chunk {}".format(id_))
    #Lets load the VinVL Features
    coco_vinvl_path = "/data/home/zmykevin/vinvl_data/CC/model_0060000/{}".format(id_)
    #coco_vinvl_feature_tsv = TSVFile(os.path.join(coco_vinvl_path, "features.tsv"))
    coco_vinvl_prediction_tsv = TSVFile(os.path.join(coco_vinvl_path, "predictions.tsv"))

    num_rows = coco_vinvl_prediction_tsv.num_rows()
    for i in tqdm(range(num_rows)):
        #assert coco_vinvl_prediction_tsv.seek(i)[0] == coco_vinvl_feature_tsv.seek(i)[0]
        current_prediction = coco_vinvl_prediction_tsv.seek(i)
        #current_feature = coco_vinvl_feature_tsv.seek(i)
        img_id = int(current_prediction[0])#check the original img_id
        
        assert annotation_split.get(img_id, None) is not None
        if annotation_split.get(img_id, None) == "train":
            valid_ids[img_id] = True

validation_list = []
for x in tqdm(annotation_data['images']):
  #print(x)
  if x ['split'] == "train" and valid_ids.get(x['imgid'], False):
      validation_list.append({'image_id': x['imgid'], 'captions': [x['sentences'][0]['raw']]})

print(len(validation_list))

#dump the validation_list
output_path = "/fsx/zmykevin/data/mmf_data/datasets/cc/defaults/annotations/train_vinvl_{}.npy".format(start_index)
with open(output_path, "wb") as f:
    np.save(f, validation_list)
print("Save the annotation in : {}".format(output_path))