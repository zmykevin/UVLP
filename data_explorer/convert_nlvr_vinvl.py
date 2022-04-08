#!/usr/bin/env python
# coding: utf-8

import numpy as np
import lmdb
import os
import pickle
import torch
from iopath.common.file_io import PathManager as pm
import json
from tqdm import tqdm
import os.path as op
import base64
import logging

PathManager = pm()

#ORIGINAL_COCO_PATH = "/fsx/zmykevin/data/mmf_data/datasets/coco/defaults/features/test2015.lmdb"
#ORIGINAL_COCO_PATH = "/fsx/zmykevin/data/mmf_data/datasets/flickr30k/defaults/features/detectron.lmdb"
MAX_SIZE = 1333
MIN_SIZE = 800

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
            

class LMDBCreater:
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
    def create(self, features, infos):
        env = lmdb.open(self.lmdb_path, map_size=1099511627776)
        id_list = []
        with env.begin(write=True) as txn:
            for feature, info in zip(features, infos):
                item = {}
                #item["feature_path"] = info['feature_path']
                #item['image_id'] = info['image_id']
                # key = info['image_id'].encode()
                key = info['feature_path'].encode()
                id_list.append(key)
                #Create a structured array
                info["features"] = feature
                reader = np.array([info])

                item["features"] = reader      
                item["image_height"] = reader.item().get("image_height")
                item["image_width"] = reader.item().get("image_width")
                item["num_boxes"] = reader.item().get("num_boxes")
                item["objects"] = reader.item().get("objects")
                item["bbox"] = reader.item().get("bbox")
                item["cls_prob"] = reader.item().get("cls_prob")
                
                txn.put(key, pickle.dumps(item))
            txn.put(b"keys", pickle.dumps(id_list))
                
class PaddedFasterRCNNFeatureReader:
    def __init__(self, max_loc):
        self.max_loc = max_loc
        self.first = True
        self.take_item = False

    def _load(self, image_feat_path):
        image_info = {}
        image_info["features"] = load_feat(image_feat_path)

        info_path = "{}_info.npy".format(image_feat_path.split(".npy")[0])
        if PathManager.exists(info_path):
            image_info.update(load_feat(info_path).item())

        return image_info

    def read(self, image_feat_path):
        image_info = self._load(image_feat_path)
        if self.first:
            self.first = False
            if (
                image_info["features"].size == 1
                and "image_feat" in image_info["features"].item()
            ):
                self.take_item = True

        image_feature = image_info["features"]
        #print(image_feature)
        if self.take_item:
            item = image_info["features"].item()
            if "image_text" in item:
                image_info["image_text"] = item["image_text"]
                image_info["is_ocr"] = item["image_bbox_source"]
                image_feature = item["image_feat"]

            if "info" in item:
                if "image_text" in item["info"]:
                    image_info.update(item["info"])
                image_feature = item["feature"]

        # Handle case of features with class probs
        if (
            image_info["features"].size == 1
            and "features" in image_info["features"].item()
        ):
            item = image_info["features"].item()
            image_feature = item["features"]
            image_info["image_height"] = item["image_height"]
            image_info["image_width"] = item["image_width"]

            # Resize these to self.max_loc
            image_loc, _ = image_feature.shape
            image_info["cls_prob"] = np.zeros(
                (self.max_loc, item["cls_prob"].shape[1]), dtype=np.float32
            )
            image_info["cls_prob"][0:image_loc,] = item["cls_prob"][: self.max_loc, :]
            image_info["bbox"] = np.zeros(
                (self.max_loc, item["bbox"].shape[1]), dtype=np.float32
            )
            image_info["bbox"][0:image_loc,] = item["bbox"][: self.max_loc, :]
            image_info["num_boxes"] = item["num_boxes"]

        # Handle the case of ResNet152 features
        if len(image_feature.shape) > 2:
            shape = image_feature.shape
            image_feature = image_feature.reshape(-1, shape[-1])

        image_loc, image_dim = image_feature.shape
        tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat[0:image_loc,] = image_feature[: self.max_loc, :]  # noqa
        image_feature = torch.from_numpy(tmp_image_feat)

        del image_info["features"]
        image_info["max_features"] = torch.tensor(image_loc, dtype=torch.long)
        return image_feature, image_info


class LMDBFeatureReader(PaddedFasterRCNNFeatureReader):
    def __init__(self, max_loc, base_path):
        super().__init__(max_loc)
        self.db_path = base_path

        if not PathManager.exists(self.db_path):
            raise RuntimeError(
                "{} path specified for LMDB features doesn't exists.".format(
                    self.db_path
                )
            )
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False, buffers=True) as txn:
            self.image_ids = pickle.loads(txn.get(b"keys"))
            self.image_id_indices = {
                self.image_ids[i]: i for i in range(0, len(self.image_ids))
            }

    def _load(self, image_file_path):
        #print("env is: {}".format(self.env))
        if self.env is None:
            #print("initialize db")
            self._init_db()

        split = os.path.relpath(image_file_path, self.db_path).split(".npy")[0]

        try:
            image_id = int(split.split("_")[-1])
            # Try fetching to see if it actually exists otherwise fall back to
            # default
            img_id_idx = self.image_id_indices[str(image_id).encode()]
        except (ValueError, KeyError):
            # The image id is complex or involves folder, use it directly
            image_id = str(split).encode()
            img_id_idx = self.image_id_indices[image_id]

        with self.env.begin(write=False, buffers=True) as txn:
            image_info = pickle.loads(txn.get(self.image_ids[img_id_idx]))

        return image_info


def normalize_bbox(bbox, im_shape):
    im_height = im_shape[0]
    im_width = im_shape[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    # Scale based on minimum size
    im_scale = MIN_SIZE / im_size_min

    # Prevent the biggest axis from being more than max_size
    # If bigger, scale it down
    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = MAX_SIZE / im_size_max

    normalized_bbox = [x/im_scale for x in bbox]
    return normalized_bbox

#Lets load the VinVL Features
#coco_vinvl_path = "/data/home/zmykevin/vinvl_data/Flickr30K"
coco_vinvl_path = "/home/zmykevin/fb_intern/data/vinvl_data/nlvr2/nlvr2_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000"
coco_vinvl_feature_tsv = TSVFile(os.path.join(coco_vinvl_path, "features.tsv"))
coco_vinvl_prediction_tsv = TSVFile(os.path.join(coco_vinvl_path, "predictions.tsv"))
# coco_vinvl_id2index = json.load(open(os.path.join(coco_vinvl_path, "imageid2idx.json"), "r"))



# max_feat = 30
# FEAT_READER = LMDBFeatureReader(max_feat, ORIGINAL_COCO_PATH)



feat_list = []
info_list = []
for i in tqdm(range(coco_vinvl_feature_tsv.num_rows())):
    assert coco_vinvl_prediction_tsv.seek(i)[0] == coco_vinvl_feature_tsv.seek(i)[0]
    #img_id = coco_vinvl_feature_tsv.seek(i)[0].zfill(12)
    img_id = coco_vinvl_feature_tsv.seek(i)[0]
    # img_file_path = os.path.join(ORIGINAL_COCO_PATH, "flickr30k_{}.npy".format(img_id))
    # try:
    #     sample_feats, sample_info = FEAT_READER.read(img_file_path)
    # except:
    #     continue
    
    #replace feats with current information
    vinvl_num_boxes = int(coco_vinvl_feature_tsv.seek(i)[1])
    vinvl_feature = np.frombuffer(base64.b64decode(coco_vinvl_feature_tsv.seek(i)[2]), np.float32
                ).reshape((vinvl_num_boxes, -1))[:,:2048]
    vinvl_prediction = json.loads(coco_vinvl_prediction_tsv.seek(i)[1])
    
    #update the new num_box, bbox, objects and cls_prob can be reduced to the new num_boxes
    sample_feats_changed = vinvl_feature
    #sample_info_changed = sample_info.copy()
    sample_info_changed = {}
    sample_info_changed["image_height"] = vinvl_prediction["image_h"]
    sample_info_changed["image_width"] = vinvl_prediction["image_w"]
    #sample_info_changed['num_boxes'] = vinvl_num_boxes if vinvl_num_boxes < 100 else 100
    sample_info_changed['num_boxes'] = vinvl_num_boxes

    #update the bbox information
    updated_bbox = np.array([normalize_bbox(obj['rect'],[sample_info_changed["image_height"],sample_info_changed["image_width"]]) for obj in vinvl_prediction['objects']])

    sample_info_changed['bbox'] = updated_bbox

    #update the dimension for objects, and cls_prob
    sample_info_changed['objects'] = np.zeros(sample_info_changed['num_boxes'], np.int32)
    sample_info_changed['cls_prob'] = np.zeros((sample_info_changed['num_boxes'],1601))
    
    if sample_info_changed.get("feature_path", None) is None:
        sample_info_changed["feature_path"] = img_id
    #append it to feat_list and info_list
    feat_list.append(sample_feats_changed)
    info_list.append(sample_info_changed)
    

#lmdb_path = "/fsx/zmykevin/data/mmf_data/datasets/coco/defaults/features/test2015_vinvl_nopadding.lmdb"
lmdb_path = "/fsx/zmykevin/data/mmf_data/datasets/nlvr2/defaults/features/train_vinvl.lmdb"
lmdb_creater = LMDBCreater(lmdb_path)
lmdb_creater.create(feat_list, info_list)






