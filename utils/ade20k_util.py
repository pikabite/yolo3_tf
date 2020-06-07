#%%
import csv
from pathlib import Path
import numpy as np
import scipy.io
from tqdm import tqdm
from PIL import Image, ImageDraw

from model.model_utils import *
from model.yolo3 import Yolo3

class Ade20kUtil :
    def __init__(self, abs_path : Path, config : dict):

        self.config = config
        # class_list = 
        mat = scipy.io.loadmat(str(Path("datas", "ade20k", "index_ade20k.mat")))
        self.abs_path = abs_path

        self.class_list = []
        tmpi = 0
        for cla in [v for v in csv.DictReader(Path("datas", "ade20k", "objectInfo150.csv").open("r").readlines())] :
            if cla["Stuff"] == "0" :
                self.class_list.append({"id":tmpi, "name":cla["Name"]})
                tmpi += 1
        
        self.image_list = []
        self.seggt_list = []
        self.ann_list = []
        for i in tqdm(range(mat["index"][0][0][1][0].shape[0])) :
            self.image_list.append(mat["index"][0][0][1][0][i][0] + "/" + mat["index"][0][0][0][0][i][0])
            self.seggt_list.append(mat["index"][0][0][1][0][i][0] + "/" + mat["index"][0][0][0][0][i][0].replace(".jpg", "_seg.png"))
            self.ann_list.append(mat["index"][0][0][1][0][i][0] + "/" + mat["index"][0][0][0][0][i][0].replace(".jpg", "_atr.txt"))

        self.data_list = self.ann_list

    def extract_image(self, image_index) :

        i = image_index
        img_raw = Image.open(str(self.abs_path/self.image_list[i]))
        preprocessed = self.preprocess_image(img_raw)
        if list(preprocessed.shape) != (self.config["image_shape"]) :
            print("picture is wrong!")
            return None
        else :
            return preprocessed
        

    def extract_bbox (self, image_index) :

        i = image_index

        img_seggt = Image.open(str(self.abs_path/self.seggt_list[i])).resize((self.config["image_shape"][0], self.config["image_shape"][1]))
        img_seggt = np.asarray(img_seggt).transpose((2, 0, 1))

        obj_list = []
        for v in [v.replace(" ", "").replace(",", ";").split("#") for v in (self.abs_path/self.ann_list[image_index]).open("r").readlines()] :
            for c in self.class_list :
                if v[1] == "0" and v[3] in c["name"] :
                    obj_list.append({
                        "id" : int(v[0]),
                        "class_id" : c["id"],
                        "name" : c["name"]
                    })
                    break

        # [[ymin xmin ymax xmax], ...]
        bboxes = []
        classes = []
        b_index = np.unique(img_seggt[2])
        
        for ii, v in enumerate(obj_list) :
            # print(v["id"])
            # print(b_index)
            if v["id"] >= len(b_index) :
                # print(str(v["id"]) + " is not in b_index")
                return None, None
            tmp = img_seggt[2] == b_index[v["id"]]
            tmptmp = np.where(tmp)
            bbox = [np.min(tmptmp[1]), np.min(tmptmp[0]), np.max(tmptmp[1]), np.max(tmptmp[0])]
            bbox = [
                np.min([bbox[0], bbox[2]]),
                np.min([bbox[1], bbox[3]]),
                np.max([bbox[0], bbox[2]]),
                np.max([bbox[1], bbox[3]])
            ]
            bboxes.append(bbox)

            bbox = np.array(bbox)
            # print(bbox[2:4] - bbox[:2])

            classes.append(v["class_id"])
            
            # img_raw = Image.open(str(self.abs_path/self.image_list[i])).resize((self.config["image_shape"][0], self.config["image_shape"][1]))
            # ImageDraw.Draw(img_raw).rectangle([tuple(bbox[:2]), tuple(bbox[2:4])], outline="red")
        
        # img_raw.save(f"./tmp_img/{self.seggt_list[i].split('/')[-1]}")

        return bboxes, classes


    def preprocess_image (self, image) :

        image = image.resize((self.config["image_shape"][0], self.config["image_shape"][1]))
        image = np.asarray(image).copy().astype(np.float32)

        image += np.random.normal(0, 1, image.shape)

        return image


# #################################

# import yaml

# config = yaml.load("".join(Path("configs/train/train_test.yaml").open("r").readlines()), Loader=yaml.FullLoader)
# au = Ade20kUtil(Path("/home/kdy/Downloads/tmp26"), config=config)

# au.extract_image(10)
# au.extract_bbox(10)

# model = Yolo3(config=config)

# bbox_gt = np.array([
#     [319,  29, 414, 184],
#     [122,  16, 353, 232],
#     [308, 185, 414, 252],
#     [293, 234, 414, 332]
# ], dtype=np.float32)
# class_gt = np.array([
#     [0, 0, 0, 1, 0],
#     [0, 0, 0, 1, 0],
#     [0, 0, 0, 1, 0],
#     [0, 0, 0, 1, 0]
# ], dtype=np.float32)

# au.data_to_gt(model, bbox_gt, class_gt)


# #%%

# def convert_bbox_to_predict_test2 (bbox, prior) :
#     # bbox : [ymin, xmin, ymax, xmax]
#     # prior : [ymin, xmin, ymax, xmax]
#     e = 1e-9
#     pyx, phw = np.split(prior, 2)
#     byx, bhw = np.split(bbox, 2)

#     pyx = pyx.astype(np.float32)
#     byx = byx.astype(np.float32)
#     phw = phw.astype(np.float32)
#     bhw = bhw.astype(np.float32)

#     phw -= pyx
#     bhw -= byx

#     pcenter = pyx + phw/2
#     bcenter = byx + bhw/2

#     apm = 10.0

#     if phw[0] == phw[1] :
#         stride = (phw[0] + phw[1])/(2*apm)
#     else :
#         stride = (phw[0] + phw[1])/(2.5*apm)

#     c = (pcenter - stride*0.5)/stride

#     bcenter = bcenter/stride
#     print(bcenter)
#     print(c)
#     # print(bcenter-c)
#     # print(1-bcenter+c)
#     tyx = np.log((bcenter-c)/(1-bcenter+c))
#     thw = np.log(bhw/phw)

#     return np.concatenate([tyx, thw], axis=0)



# iou, index = allocate_bbox_on_anchor(model.total_priors, bbox_gt[2])

# print(bbox_gt[2])
# print(model.total_priors[index])

# convert_bbox_to_predict_test2(bbox_gt[2], model.total_priors[index])


# # %%


# %%
