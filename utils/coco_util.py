#%%
from pathlib import Path
import numpy as np
import scipy.io
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np

import json

from model.model_utils import *
from model.yolo3 import Yolo3

from copy import copy

class CocoUtil :
    def __init__(self, abs_path : Path, config : dict, datatype : str):

        self.config = config

        self.abs_path = abs_path

        self.class_list = json.loads((abs_path/"categories.txt").open("r").read())

        self.img_id_list = []
        self.image_list = []
        self.ann_list = []

        self.datatype = datatype

        if datatype == "train" :
            for v in tqdm((abs_path/"coco_bbox_modified").iterdir()) :
            # for v in  :
                self.img_id_list.append(v.stem)
                self.image_list.append(abs_path/"train2017"/(v.stem + ".jpg"))
                self.ann_list.append(json.loads((abs_path/"coco_bbox_modified"/(v.stem + ".txt")).open("r").read()))
        elif datatype == "test" :
            for v in tqdm((abs_path/"coco_bbox_mod_val").iterdir()) :
            # for v in  :
                self.img_id_list.append(v.stem)
                self.image_list.append(abs_path/"val2017"/(v.stem + ".jpg"))
                self.ann_list.append(json.loads((abs_path/"coco_bbox_mod_val"/(v.stem + ".txt")).open("r").read()))


        self.data_list = self.ann_list

    def extract_image(self, image_index) :

        i = image_index
        img_raw = Image.open(str(self.abs_path/self.image_list[i]))
        preprocessed = self.preprocess_image(img_raw)
        # if list(preprocessed.shape) != (self.config["image_shape"]) :
            # print("picture is wrong!")
            # return None
        # else :
        return preprocessed
        

    def extract_bbox (self, image_index) :

        i = image_index

        ann = self.ann_list[i]
        bboxes = []
        classes = []
        height = 0
        width = 0
        for v in ann :
            bboxes.append(v["bbox"])
            classes.append(v["category_id"]-1)
            width = v["width"]
            height = v["height"]

        if width == 0 or height == 0 : return bboxes, classes
        
        bboxes = np.array(bboxes, dtype=np.int)
        classes = np.array(classes, dtype=np.int)

        bboxes[:, 2:4] += bboxes[:, :2]

        return bboxes, classes


    def preprocess_image (self, image) :

        # image = image.resize((self.config["image_shape"][0], self.config["image_shape"][1]))
        image = np.asarray(image).copy().astype(np.float32)/256

        # if self.datatype == "train" :
        #     image += np.random.normal(0, 0.02, image.shape)

        return image

    def draw_sample (self, idid) :

        img, bb, classes = self.get_data(idid)
        img = (img*256).astype(np.uint8)

        imgimg = Image.fromarray(img)
        imgdraw = ImageDraw.Draw(imgimg)
        for v in bb :
            imgdraw.rectangle([*v])
        
        return img.shape, bb, imgimg

    def get_data (self, image_index) :

        image = self.extract_image(image_index)
        bboxes, classes = self.extract_bbox(image_index)

        if bboxes is None or image is None or len(bboxes) == 0 or len(image.shape) < 3 :
            return None, None, None
            
        image, bboxes = self.pad_to_square(image, bboxes)
        if self.config["is_training"] :
            image, bboxes = self.flip_image(image, bboxes)

        return image, bboxes, classes
    
    def flip_image(self, image, bboxes) :

        rint = np.random.randint(0, 3)
        if rint == 0 :
            image = np.flip(image, axis=0)
            tmp = self.config["image_shape"][1] - bboxes[:, 1]
            tmp2 = self.config["image_shape"][1] - bboxes[:, 3]
            bboxes[:, 1] = tmp2
            bboxes[:, 3] = tmp
        elif rint == 1 :
            image = np.flip(image, axis=1)
            tmp = self.config["image_shape"][0] - bboxes[:, 0]
            tmp2 = self.config["image_shape"][0] - bboxes[:, 2]
            bboxes[:, 0] = tmp2
            bboxes[:, 2] = tmp

        return image, bboxes

    def pad_to_square (self, image, bboxes) :

        newimg = np.zeros((max(image.shape), max(image.shape), 3))
        new_offset = (int(newimg.shape[0]/2 - image.shape[0]/2), int(newimg.shape[1]/2 - image.shape[1]/2))
        newimg[new_offset[0]:new_offset[0]+image.shape[0], new_offset[1]:new_offset[1]+image.shape[1], :] = image
        newimg = cv2.resize(newimg, (self.config["image_shape"][0], self.config["image_shape"][1]))

        a = self.config["image_shape"][0]/(image.shape[0] + new_offset[0]*2)
        b = self.config["image_shape"][1]/(image.shape[1] + new_offset[1]*2)
        new_bboxes = []
        for v in bboxes :
            new_bboxes.append(
                [
                    (v[0] + new_offset[1])*b,
                    (v[1] + new_offset[0])*a,
                    (v[2] + new_offset[1])*b,
                    (v[3] + new_offset[0])*a
                ]
            )
        new_bboxes = np.array(new_bboxes, dtype=np.int)

        return newimg, new_bboxes

# #################################


# %%

# import yaml
# from pathlib import Path

# config = yaml.load("".join(Path("configs/train/train_coco.yaml").open("r").readlines()), Loader=yaml.FullLoader)
# cutil = CocoUtil(Path("/datasets/coco"), config=config, datatype="train")


# #%%

# a, b, c = cutil.draw_sample(9)
# print(a)
# print(b)
# print(c)




# # %%

# c

