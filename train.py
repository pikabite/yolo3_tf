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
        dg = DataGenerator(CocoUtil(Path(config["dataset_path"]), config=config, datatype="train"), config=config)
        dg_v = DataGenerator(CocoUtil(Path(config["dataset_path"]), config=config, datatype="test"), config=config)
    elif config["dataset"] == "ade20k" :
        dg = DataGenerator(Ade20kUtil(Path(config["dataset_path"]), config=config), config=config)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    model.sess = tf.Session(config=tfconfig)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=20)

    global_vars = tf.global_variables()
    is_not_initialized = model.sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if not_initialized_vars is not None:
        model.sess.run(tf.variables_initializer(not_initialized_vars))

    epoch = config["epoch"]
    present_epoch = config["present_epoch"]
    print(f"training start")
    print(f"total epoch : {str(epoch)}")

    if not Path(config["log_dir"]).exists() : Path(config["log_dir"]).mkdir(parents=True)
    log_file = Path(config["log_dir"])/(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+".txt")
    log_file.open("a+").write("epoch;iteration;total_loss;bbox_xy_loss;bbox_wh_loss;objness_loss;class_loss\n")

    if present_epoch > 1 :
        save_path = (Path(config["weight_store_path"])/(config["dataset"]+"_"+str(present_epoch)))
        saver.restore(model.sess, str(save_path/"model.ckpt"))
        print("model restore")
        

    for e in range(present_epoch, epoch+1) :
        lr = config["learning_rate"] * (config["lr_decaying_rate"]**(e/config["lr_decaying_interval"]))

        loss_epoch_total = []
        tqdm_iterator = tqdm(range(dg.len()))

        for i in tqdm_iterator :
            # print(i)

            images, gts = dg.get_item(i)

            if len(images) == 0 or len(gts) == 0 :
                continue

            image_inputs = np.array(images)
            bbox_input = np.array([v[:, :4] for v in gts])
            class_input = np.squeeze([v[:, 4:5] for v in gts])

            if len(class_input) < 1 :
                continue
            vv = False

            for ii in range(len(bbox_input)) :
                vv = vv + np.any(bbox_input[ii][:, 2:4] - bbox_input[ii][:, :2] <= 0)
            if vv : continue

            b_bbox_inputs = []
            m_bbox_inputs = []
            s_bbox_inputs = []
            b_class_inputs = []
            m_class_inputs = []
            s_class_inputs = []
            b_objness_inputs = []
            m_objness_inputs = []
            s_objness_inputs = []
            b_noobjness_inputs = []
            m_noobjness_inputs = []
            s_noobjness_inputs = []
            for ii in range(len(gts)) :
                bbox_input_tmp, class_input_tmp, objness_input_tmp, noobj_input_tmp = data_to_gt(model.priors_set, bbox_input[ii], class_input[ii], config["class_number"])
                b_bbox_inputs.append(bbox_input_tmp[0])
                m_bbox_inputs.append(bbox_input_tmp[1])
                s_bbox_inputs.append(bbox_input_tmp[2])
                b_class_inputs.append(class_input_tmp[0])
                m_class_inputs.append(class_input_tmp[1])
                s_class_inputs.append(class_input_tmp[2])
                b_objness_inputs.append(objness_input_tmp[0])
                m_objness_inputs.append(objness_input_tmp[1])
                s_objness_inputs.append(objness_input_tmp[2])
                b_noobjness_inputs.append(noobj_input_tmp[0])
                m_noobjness_inputs.append(noobj_input_tmp[1])
                s_noobjness_inputs.append(noobj_input_tmp[2])

                # save single image and its objness scoremap
                # save_files_from_final_output(image_inputs[ii], objness_input_tmp, noobj_input_tmp)

            feed = {
                model.image_inputs : image_inputs,
                model.gt_placeholders[0]["bbox_computed_gt"] : b_bbox_inputs,
                model.gt_placeholders[0]["objectness_gt"] : b_objness_inputs,
                model.gt_placeholders[0]["noobjectness_gt"] : b_noobjness_inputs,
                model.gt_placeholders[0]["class_computed_gt"] : b_class_inputs,
                model.gt_placeholders[1]["bbox_computed_gt"] : m_bbox_inputs,
                model.gt_placeholders[1]["objectness_gt"] : m_objness_inputs,
                model.gt_placeholders[1]["noobjectness_gt"] : m_noobjness_inputs,
                model.gt_placeholders[1]["class_computed_gt"] : m_class_inputs,
                model.gt_placeholders[2]["bbox_computed_gt"] : s_bbox_inputs,
                model.gt_placeholders[2]["objectness_gt"] : s_objness_inputs,
                model.gt_placeholders[2]["noobjectness_gt"] : s_noobjness_inputs,
                model.gt_placeholders[2]["class_computed_gt"] : s_class_inputs,
                model.decaying_lr : lr,
                model.is_training : True
            }

            tensors = [model.train_op, model.final_loss, model.losses["bbox_xy_loss"], model.losses["bbox_wh_loss"], model.losses["objness_loss"], model.losses["class_loss"]]
            _, loss_tmp, bbox_xy_loss, bbox_wh_loss, objness_loss, class_loss = model.sess.run(tensors, feed_dict=feed)

            loss_epoch_total.append(loss_tmp)
            tqdm_iterator.set_description(f"Loss : {str(np.mean(loss_epoch_total))}, bbox_xy_loss : {str(bbox_xy_loss)}, bbox_wh_loss : {str(bbox_wh_loss)}, obj_loss : {str(objness_loss)}, class_loss : {str(class_loss)}")

            log_file.open("a+").write(f"{str(e)};{str(i)};{str(loss_tmp)};{str(bbox_xy_loss)};{str(bbox_wh_loss)};{str(objness_loss)};{str(class_loss)}\n")

        print(f"{str(e)} epoch is over!! avg loss {str(np.mean(loss_epoch_total))}")
        log_file.open("a+").write(f"{str(e)} epoch is over!! avg loss {str(np.mean(loss_epoch_total))}\n")

        loss_epoch_total = []

        if e%config["validation_interval"] == 0 :

            valid_count = 1
            for i in range(valid_count) :
                images_v, gts_v = dg_v.get_item(i)
                validate_data(model, images_v, gts_v)

        if e%config["saving_interval"] == 0 and e > 1 and e != config["present_epoch"] :
            save_path = (Path(config["weight_store_path"])/(config["dataset"]+"_"+str(e)))
            if not save_path.exists() : save_path.mkdir(parents=True)
            print(save_path/"model.ckpt")
            saver.save(model.sess, str(save_path/"model.ckpt"))
            print("SAVE!")
            pass
        dg.on_epoch_end()



#%%
