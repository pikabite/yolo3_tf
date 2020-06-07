#%%
from pathlib import Path
import os, sys, argparse
import yaml

from model.yolo3 import Yolo3
from utils.data_provider import DataGenerator
from utils.ade20k_util import Ade20kUtil
from utils.coco_util import CocoUtil

import numpy as np

import tensorflow as tf

from tqdm import tqdm

from model.model_utils import *

import time
from datetime import datetime

import cv2

def pred_to_bbox (bbox, priors, stride) :

    if bbox.shape[0] <= 0 :
        return np.array([[0, 0, 0, 0]])
    
    # bbox = np.squeeze(bbox)
    # priors = np.squeeze(priors)
    # print(bbox.shape)
    # print(priors.shape)
    
    sigmoided = 1 / (1 + np.exp(-bbox[:, :2]))
    cxy = (sigmoided + (priors[:, :2]/2+priors[:, 2:4]/2)//stride)*stride

    exponended = np.exp(bbox[:, 2:4])
    wh = exponended * (priors[:, 2:4] - priors[:, :2])

    return np.squeeze(np.concatenate([cxy - wh/2, cxy + wh/2], axis=1))


#%%

if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    
    config = yaml.load("".join(Path(args.config).open("r").readlines()), Loader=yaml.FullLoader)
    # config = yaml.load("".join(Path("configs/train/train_test.yaml").open("r").readlines()), Loader=yaml.FullLoader)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in config["gpu_indices"]])

    model = Yolo3(config=config)
    # print(model.bbox_out)

    if config["dataset"] == "coco" :
        dg = DataGenerator(CocoUtil(Path(config["dataset_path"]), config=config, datatype="test"), config=config)
    elif config["dataset"] == "ade20k" :
        dg = DataGenerator(Ade20kUtil(Path(config["dataset_path"]), config=config), config=config)

    # dg = DataGenerator(Ade20kUtil(Path("/home/kdy/Downloads/tmp26"), config=config), config=config)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    model.sess = tf.Session(config=tfconfig)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    present_epoch = config["present_epoch"]
    print(f"test start")
    print(f"epoch : {str(present_epoch)}")

    save_path = (Path(config["weight_store_path"])/(config["dataset"]+"_"+str(present_epoch)))
    saver.restore(model.sess, str(save_path/"model.ckpt"))
    # model.sess.run(tf.local_variables_initializer())

    # global_vars = tf.global_variables()
    # is_not_initialized = model.sess.run([tf.is_variable_initialized(var) for var in global_vars])
    # not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    # if not_initialized_vars is not None:
    #     model.sess.run(tf.variables_initializer(not_initialized_vars))

    # if not Path(config["log_dir"]).exists() : Path(config["log_dir"]).mkdir(parents=True)
    # log_file = Path(config["log_dir"])/(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+".txt")
    # log_file.open("a+").write("epoch;iteration;total_loss;bbox_loss;objness_loss;class_loss\n")

    images, gts = dg.get_item(0)

    image_inputs = np.array(images)
    bbox_input = np.array([v[:, :4] for v in gts])
    # class_input = np.array([np.eye(config["class_number"])[np.squeeze(v[:, 4:5])] for v in gts])
    class_input = np.squeeze([v[:, 4:5] for v in gts])

    tensors = [
        model.bbox_out_set[0], model.objness_out_set[0], model.class_out_set[0],
        model.bbox_out_set[1], model.objness_out_set[1], model.class_out_set[1],
        model.bbox_out_set[2], model.objness_out_set[2], model.class_out_set[2]
    ]
    feed = {
        model.image_inputs : image_inputs,
        model.is_training : False
    }
    outs = model.sess.run(tensors, feed_dict=feed)
    b_bout, b_oout, b_cout, m_bout, m_oout, m_cout, s_bout, s_oout, s_cout = outs

    conf_thr = 0.5
    # conf_thr = np.max(s_oout)

    
    b_oout = 1/(1+np.exp(-b_oout))
    m_oout = 1/(1+np.exp(-m_oout))
    s_oout = 1/(1+np.exp(-s_oout))

    # print(b_bout.shape)
    # print((b_oout > iou_thr).shape)
    print(np.sum((b_oout >= conf_thr)*1))
    print(np.sum((m_oout >= conf_thr)*1))
    print(np.sum((s_oout >= conf_thr)*1))

    b_tmptmp = list(np.where(b_oout >= np.max(b_oout)))
    m_tmptmp = list(np.where(m_oout >= np.max(m_oout)))
    s_tmptmp = list(np.where(s_oout >= np.max(s_oout)))
    b_tmptmp[1], b_tmptmp[2] = b_tmptmp[2], b_tmptmp[1]
    m_tmptmp[1], m_tmptmp[2] = m_tmptmp[2], m_tmptmp[1]
    s_tmptmp[1], s_tmptmp[2] = s_tmptmp[2], s_tmptmp[1]
    bbox = pred_to_bbox(b_bout[b_tmptmp[:4]], model.priors_set[0][b_tmptmp[1:4]], 32)
    mbox = pred_to_bbox(m_bout[m_tmptmp[:4]], model.priors_set[1][m_tmptmp[1:4]], 16)
    sbox = pred_to_bbox(s_bout[s_tmptmp[:4]], model.priors_set[2][s_tmptmp[1:4]], 8)

    # tmp = np.concatenate([
    #     bbox,
    #     mbox,
    #     sbox
    # ], axis=0)

    # print(bbox)
    # print(mbox)
    # print(sbox)

    # print(bbox_input)

    print("----------------------")

    tmpimg = image_inputs[0]*255

    b_tmptmp = np.where(b_oout >= conf_thr)
    for i in range(len(b_tmptmp[0])) :
        tmp = b_bout[b_tmptmp[0][i], b_tmptmp[1][i], b_tmptmp[2][i], b_tmptmp[3][i]]
        tmp2 = model.priors_set[0][b_tmptmp[1][i], b_tmptmp[2][i], b_tmptmp[3][i]]
        bbox = pred_to_bbox(np.array([tmp]), np.array([tmp2]), 32)
        # print(bbox)
        cv2.rectangle(tmpimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
    for i in range(len(m_tmptmp[0])) :
        tmp = m_bout[m_tmptmp[0][i], m_tmptmp[1][i], m_tmptmp[2][i], m_tmptmp[3][i]]
        tmp2 = model.priors_set[1][m_tmptmp[1][i], m_tmptmp[2][i], m_tmptmp[3][i]]
        bbox = pred_to_bbox(np.array([tmp]), np.array([tmp2]), 16)
        # print(bbox)
        cv2.rectangle(tmpimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
    for i in range(len(s_tmptmp[0])) :
        tmp = s_bout[s_tmptmp[0][i], s_tmptmp[1][i], s_tmptmp[2][i], s_tmptmp[3][i]]
        tmp2 = model.priors_set[2][s_tmptmp[1][i], s_tmptmp[2][i], s_tmptmp[3][i]]
        bbox = pred_to_bbox(np.array([tmp]), np.array([tmp2]), 16)
        # print(bbox)
        cv2.rectangle(tmpimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)

    cv2.imwrite("./tmp/pred_output.png", tmpimg)


    # s_oout = np.squeeze(s_oout)
    # print(np.sum((s_oout >= 0.5)*1))
    # indexes = np.where(s_oout == np.max(s_oout))
    # indexes = np.squeeze(indexes)
    # s_max_bout = s_bout[0, indexes[0], indexes[1], indexes[2]]
    # sigmoided = 1 / (1 + np.exp(-s_max_bout[:2]))
    # prior = model.priors_set[2][indexes[0], indexes[1], indexes[2]]
    # xy = (sigmoided + (prior[:2]/2+prior[2:4]/2)//8)*8

    # print(xy)
    # print(bbox_input)

    
    # model.priors[maximum_objs_index][:2]
    
    print("Done")
