#%%

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
from model.model_utils import *

# %%

class Yolo3 :
    
    def __init__(self, config):

        self.config = config
        self.class_number = config["class_number"]
        self.batch_size = config["batch_size"]

        self.sm = 1

        tf.reset_default_graph()

        self.build_model()
        self.anchor_bbox_priors()
        if self.config["is_training"] :
            self.compute_loss_all()

    def darknet_block (self, netnet, filters, repeat) :

        for i in range(repeat) :
            shortcut = netnet
            netnet = slim.conv2d(netnet, filters, 1)
            netnet = slim.conv2d(netnet, filters*2, 3)
            netnet = shortcut + netnet
        return netnet

    def obj_conv_block (self, netnet, filter1, filter2, repeat=3) :

        for i in range(repeat) :
            netnet = slim.conv2d(netnet, filter1, 1)
            netnet = slim.conv2d(netnet, filter2, 3)
        return netnet

    def build_model (self) :

        shape = self.config["image_shape"]
        self.image_inputs = tf.placeholder(tf.float32, [None, shape[0], shape[1], shape[2]], "input_image")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.build_backbone(is_training=self.is_training)
        self.build_rpns(is_training=self.is_training)

    def upsample (self, netnet) :
        new_height, new_width = int(netnet.shape[1]*2), int(netnet.shape[2]*2)
        netnet = tf.image.resize_nearest_neighbor(netnet, (new_height, new_width))
        return netnet

    def build_backbone (self, is_training, end_points_collection="darknet53") :

        with slim.arg_scope(
            [slim.conv2d], 
            padding="SAME",
            activation_fn=tf.nn.relu6,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training},
            outputs_collections=end_points_collection,
        ) :
            
            # self.image_inputs = tf.transpose(self.image_inputs, (0, 2, 1, 3))

            netnet = slim.conv2d(self.image_inputs, 32, 3)
            netnet = slim.conv2d(netnet, 64, 3, stride=2)
            self.darknet_block(netnet, 32, 1)
            netnet = slim.conv2d(netnet, 128, 3, stride=2)
            self.darknet_block(netnet, 64, 2)
            netnet = slim.conv2d(netnet, 256, 3, stride=2)
            self.darknet_block(netnet, 128, 8)
            self.route1 = netnet
            netnet = slim.conv2d(netnet, 512, 3, stride=2)
            self.route2 = netnet
            netnet = slim.conv2d(netnet, 1024, 3, stride=2)
            self.route3 = netnet

    def build_rpns (self, is_training, end_points_collection="darknet53_rpn") :

        with slim.arg_scope(

            [slim.conv2d],
            padding="SAME",
            activation_fn=tf.nn.relu6,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training},
            outputs_collections=end_points_collection,
        ) :

            big_features = slim.conv2d(self.route3, 256, 1)
            self.big_anchor = tf.identity(slim.conv2d(big_features, int(3*(5+self.class_number)), 1, normalizer_fn=None, activation_fn=None))
            self.big_bbox_out = tf.stack(tf.split(self.big_anchor[:, :, :, :12], 3, axis=3), axis=3)
            self.big_objectness_out = tf.stack(tf.split(self.big_anchor[:, :, :, 12:15], 3, axis=3), axis=3)
            self.big_class_out = tf.stack(tf.split(self.big_anchor[:, :, :, 15:], 3, axis=3), axis=3)

            for_middle_from_big = slim.conv2d(self.route3, 256, 1)

            big_anchor_upsampled = self.upsample(for_middle_from_big)

            middle_bridge = tf.concat([big_anchor_upsampled, self.route2], axis=3)
            middle_bridge = self.obj_conv_block(middle_bridge, 256, 512)
            middle_features = slim.conv2d(middle_bridge, 256, 1)
            self.middle_anchor = tf.identity(slim.conv2d(middle_features, int(3*(5+self.class_number)), 1, normalizer_fn=None, activation_fn=None))
            self.middle_bbox_out = tf.stack(tf.split(self.middle_anchor[:, :, :, :12], 3, axis=3), axis=3)
            self.middle_objectness_out = tf.stack(tf.split(self.middle_anchor[:, :, :, 12:15], 3, axis=3), axis=3)
            self.middle_class_out = tf.stack(tf.split(self.middle_anchor[:, :, :, 15:], 3, axis=3), axis=3)

            for_small_from_middle = self.upsample(slim.conv2d(middle_bridge, 128, 1))

            concated_small = tf.concat([self.route1, for_small_from_middle], axis=3)
            aft_s_block = self.obj_conv_block(concated_small, 128, 256)
            small_features = slim.conv2d(aft_s_block, 256, 1)
            self.small_anchor = tf.identity(slim.conv2d(small_features, int(3*(5+self.class_number)), 1, normalizer_fn=None, activation_fn=None))
            self.small_bbox_out = tf.stack(tf.split(self.small_anchor[:, :, :, :12], 3, axis=3), axis=3)
            self.small_objectness_out = tf.stack(tf.split(self.small_anchor[:, :, :, 12:15], 3, axis=3), axis=3)
            self.small_class_out = tf.stack(tf.split(self.small_anchor[:, :, :, 15:], 3, axis=3), axis=3)
            
            self.bbox_out_set = [
                self.big_bbox_out,
                self.middle_bbox_out,
                self.small_bbox_out
            ]
            self.objness_out_set = [
                self.big_objectness_out,
                self.middle_objectness_out,
                self.small_objectness_out
            ]
            self.class_out_set = [
                self.big_class_out,
                self.middle_class_out,
                self.small_class_out
            ]

            # print(self.total_bbox_out)
            # print(self.total_objectness_out)
            # print(self.total_class_out)

    def anchor_bbox_priors (self) :

        iamge_size = (self.config["image_shape"][0], self.config["image_shape"][1])
        self.priors_set = self.create_priors(iamge_size)

    def compute_loss (self, bbox_out_tensor, objness_out_tensor, class_out_tensor, priors, stride) :

        shapes = objness_out_tensor.shape
        objectness_gt = tf.placeholder(tf.float32, [None, shapes[1], shapes[2], 3, 1])
        bbox_computed_gt = tf.placeholder(tf.float32, [None, shapes[1], shapes[2], 3, 4])
        noobjectness_gt = tf.placeholder(tf.float32, [None, shapes[1], shapes[2], 3, 1])
        class_computed_gt = tf.placeholder(tf.float32, [None, shapes[1], shapes[2], 3, self.class_number])
        # noobjecness_gt = tf.ones_like(objectness_gt) - objectness_gt

        obj_mask = objectness_gt
        noobj_mask = noobjectness_gt


        lambda_bbox = self.config["lambda_bbox"]
        lambda_obj = self.config["lambda_obj"]
        lambda_noobj = self.config["lambda_noobj"]

        # priors.shape == (13, 13, 3, 4)...
        # stride == 32, 16, 8
        cc = ((priors[:, :, :, 2:4] + priors[:, :, :, :2])/2)//stride
        xy_bboxout = bbox_out_tensor[:, :, :, :, :2]
        # xy_bboxout = tf.nn.sigmoid(xy_bboxout) + cc
        xy_bboxout = tf.nn.sigmoid(xy_bboxout)

        wh_bboxout = bbox_out_tensor[:, :, :, :, 2:4]
        wh_bboxout = tf.clip_by_value(wh_bboxout, -6, 6)
        # wh_bboxout = tf.exp(wh_bboxout) * (priors[:, :, :, 2:4] - priors[:, :, :, :2])
        wh_bboxout = tf.exp(wh_bboxout)

        xy_out_modified = xy_bboxout
        wh_out_modified = wh_bboxout

        bbox_loss_xy = lambda_bbox * tf.reduce_sum(obj_mask * tf.reduce_sum(tf.square(xy_out_modified - bbox_computed_gt[:, :, :, :, :2]), axis=4, keepdims=True), axis=[1, 2, 3, 4])
        bbox_loss_wh = lambda_bbox * tf.reduce_sum(obj_mask * tf.reduce_sum(tf.square(tf.sqrt(wh_out_modified) - tf.sqrt(bbox_computed_gt[:, :, :, :, 2:4])), axis=4, keepdims=True), axis=[1, 2, 3, 4])
        bbox_loss = bbox_loss_xy + bbox_loss_wh


        # obj_se = tf.square(objness_out_tensor - objectness_gt)
        # obj_loss1 = lambda_obj * tf.reduce_sum(objectness_gt * obj_se, axis=[1, 2, 3, 4])
        # obj_loss2 = lambda_noobj * tf.reduce_sum(noobjecness_gt * obj_se, axis=[1, 2, 3, 4])

        tmp = tf.square(tf.nn.sigmoid(objness_out_tensor) - objectness_gt)
        # tmp = tf.nn.sigmoid_cross_entropy_with_logits(labels=objectness_gt, logits=objness_out_tensor)
        obj_loss1 = lambda_obj * tf.reduce_sum(obj_mask * tmp, axis=[1, 2, 3, 4])
        obj_loss2 = lambda_noobj * tf.reduce_sum(noobj_mask * tmp, axis=[1, 2, 3, 4])
        objness_loss = obj_loss1 + obj_loss2

        # class_out_tensor = tf.nn.sigmoid(class_out_tensor)
        class_loss = tf.reduce_sum(obj_mask * tf.reduce_sum(tf.square(class_computed_gt - class_out_tensor), axis=4, keepdims=True), axis=[1, 2, 3, 4])
        # class_loss = tf.reduce_sum(obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=class_computed_gt, logits=class_out_tensor), axis=[1, 2, 3, 4])

        total_losses = bbox_loss + objness_loss + class_loss
        # total_losses = objness_loss

        bbox_xy_loss = tf.squeeze(tf.reduce_mean(bbox_loss_xy, axis=0))
        bbox_wh_loss = tf.squeeze(tf.reduce_mean(bbox_loss_wh, axis=0))
        bbox_loss = tf.squeeze(tf.reduce_mean(bbox_loss, axis=0))
        objness_loss = tf.squeeze(tf.reduce_mean(objness_loss, axis=0))
        class_loss = tf.squeeze(tf.reduce_mean(class_loss, axis=0))

        final_loss = tf.squeeze(tf.reduce_mean(total_losses, axis=0))

        return (
            [
                objectness_gt,
                noobjectness_gt,
                bbox_computed_gt,
                class_computed_gt
            ],
            [
                bbox_xy_loss,
                bbox_wh_loss,
                objness_loss,
                class_loss
            ],
            final_loss
        )

    def compute_loss_all (self) :
        self.gt_placeholders = []
        self.losses = {"bbox_xy_loss":[], "bbox_wh_loss":[], "objness_loss":[], "class_loss":[]}
        losses_tmp = []
        final_losses = []
        for i in range(3) :
            output = self.compute_loss(self.bbox_out_set[i], self.objness_out_set[i], self.class_out_set[i], self.priors_set[i], stride=32*(0.5**i))
            gt_holders, losses, final_loss = output
            self.gt_placeholders.append({
                "objectness_gt" : gt_holders[0],
                "noobjectness_gt" : gt_holders[1],
                "bbox_computed_gt" : gt_holders[2],
                "class_computed_gt" : gt_holders[3]
            })
            losses_tmp.append(losses)
            final_losses.append(losses)
        tmp = tf.reduce_sum(tf.transpose(losses_tmp), axis=1)
        self.losses["bbox_xy_loss"] = tmp[0]
        self.losses["bbox_wh_loss"] = tmp[1]
        self.losses["objness_loss"] = tmp[2]
        self.losses["class_loss"] = tmp[3]
        self.final_loss = tf.reduce_sum(final_losses)
        self.compute_optimizer()

    def compute_optimizer (self) :

        self.decaying_lr = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.decaying_lr, beta1=0.9, beta2=0.999,
                                    epsilon=1e-8, use_locking=False, name='Adam')
        # self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.decaying_lr)
        # self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.decaying_lr, momentum=0.03)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops) :
            self.train_op = self.optimizer.minimize(self.final_loss, colocate_gradients_with_ops=True)
            # train_op = slim.learning.create_train_op(final_loss, self.optimizer)


    def create_priors (self, image_size) :
        # [[ymin, xmin, ymax, xmax]...]

        clusters = self.config["clusters"]

        priorsets = []

        for i in range(3) :
            final_feature_size = 13*self.sm*(2**i)
            stride = int(image_size[0]/final_feature_size)
            cxy = np.stack(np.mgrid[0:final_feature_size, 0:final_feature_size]*stride, axis=0).transpose((1, 2, 0))
            # cxy = np.stack(np.mgrid[0:final_feature_size, 0:final_feature_size]*stride, axis=0).transpose((2, 1, 0))

            centerxy = cxy + stride/2

            priors_set = []
            # for c in clusters[i*3:(i+1)*3] :
            for c in clusters[(2-i)*3:(3-i)*3] :
                # print(c)
                priors_set.append(
                    np.concatenate([
                        centerxy - np.ones_like(cxy) * c/2,
                        centerxy + np.ones_like(cxy) * c/2,
                    ], axis=2)
                )
            priors = np.stack(priors_set, axis=2).astype(np.float32)
            # priors = np.reshape(priors, (-1, 4))
            # print(priors.shape)
            priorsets.append(priors)

        # priorsets = np.concatenate(priorsets, axis=0)

        return priorsets

