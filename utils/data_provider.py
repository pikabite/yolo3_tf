#%%
import numpy as np

import csv
from pathlib import Path

from PIL import Image, ImageDraw

class DataGenerator () :

    def __init__(self, data_util, config):
        # self.ade20k_util = Ade20kUtil(Path("/home/kdy/Downloads/tmp26"))
        
        self.config = config
        self.data_util = data_util

        self.datas = self.data_util.data_list

        self.batch_size = self.config["batch_size"]
        self.shuffle = self.config["shuffle"]

        self.on_epoch_end()

    def len (self):
        return int(np.floor(len(self.datas) / self.batch_size))

    def get_item (self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # data_tmp = [v for v in indexes]

        X, y = self.data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.datas))
        if self.shuffle == True :
            np.random.shuffle(self.indexes)

    def data_generation(self, data_tmp):

        # [batch_size, image_size, image_size, 3]
        x = []
        y = []
        for v in data_tmp:
            image, bboxes, classes = self.data_util.get_data(v)
            if bboxes is None or image is None or len(bboxes) == 0 :
                continue
            
            # self.save_img_tmp(image, bboxes)

            x.append(image)
            y.append(np.concatenate([np.array(bboxes), np.expand_dims(classes, axis=1)], axis=1))

        return x, y

    def save_img_tmp (self, image, bboxes) :
        tmpimg = Image.fromarray((image*255).astype(np.uint8))
        for v in bboxes :
            ImageDraw.Draw(tmpimg).rectangle([v[0], v[1], v[2], v[3]])
        tmpimg.save("./tmp_folder/" + str(np.random.randint(0, 200)) + ".png")



#%%

