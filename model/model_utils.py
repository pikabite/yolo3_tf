#%%
import numpy as np

import tensorflow as tf

from scipy import signal
import cv2

# %%

def calculate_ious (bboxes, bbox_gt) :
    # [[ymin, xmin, ymax, xmax]...]
    
    y1 = np.maximum(bbox_gt[0], bboxes[:, 0])
    y2 = np.minimum(bbox_gt[2], bboxes[:, 2])
    x1 = np.maximum(bbox_gt[1], bboxes[:, 1])
    x2 = np.minimum(bbox_gt[3], bboxes[:, 3])
    ii = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    gt_volume = (bbox_gt[0]-bbox_gt[2])*(bbox_gt[1]-bbox_gt[3])
    bboxes_volume = (bboxes[:, 0]-bboxes[:, 2])*(bboxes[:, 1]-bboxes[:, 3])
    
    uu = bboxes_volume + gt_volume - ii

    uu += (uu==0.)*1e-9

    return ii/uu



#%%

def anchor_bbox_priors (bboxes_tensor, image_size) :
    # [[ymin, xmin, ymax, xmax]...]
    stride = int(image_size[0]/bboxes_tensor.shape[1])
    cxy = np.stack(np.mgrid[0:bboxes_tensor.shape[1], 0:bboxes_tensor.shape[2]]*stride, axis=0).transpose((1, 2, 0))
    # cxy = np.repeat(np.expand_dims(cxy, 0), bboxes_tensor.shape[0], axis=0)

    priors = np.stack([
        np.concatenate([cxy, np.ones_like(cxy) * stride], axis=2),
        np.concatenate([cxy, np.ones_like(cxy) * [0.5, 2.0] * stride], axis=2),
        np.concatenate([cxy, np.ones_like(cxy) * [2.0, 0.5] * stride], axis=2)
    ], axis=2)
    # print(priors.shape)
    # print("--------------------")

    priors = priors.transpose((3, 2, 0, 1))
    c1s = priors[0] + stride/2
    c2s = priors[1] + stride/2

    priors = np.stack([
        c1s - priors[2]/2*8,
        c2s - priors[3]/2*8,
        c1s + priors[2]/2*8,
        c2s + priors[3]/2*8,
    ], axis=0)
    priors = np.clip(priors.transpose((3, 2, 1, 0)), 0, image_size[0])

    return priors

# a = np.reshape(anchor_bbox_priors(np.random.rand(2, 13, 13, 12), (416, 416)), (-1, 4))[100]
# a
# a[2:4] - a[:2]


#%%

def allocate_bbox_on_anchor (priors, bbox) :
    # [[ymin, xmin, ymax, xmax], [ymin, xmin, ymax, xmax] ... ] : n, 3, 4
    # priors.shape => (10647, 4)
    bboxc = np.array([bbox[0]+bbox[2], bbox[1]+bbox[3]])/2

    offset_b = (bboxc//32).astype(np.int)[0]*13*3 + (bboxc//32).astype(np.int)[1]*3
    offset_m = (bboxc//(32/2)).astype(np.int)[0]*13*2*3 + (bboxc//(32/2)).astype(np.int)[1]*3 + 13*13*3
    offset_s = (bboxc//(32/4)).astype(np.int)[0]*13*4*3 + (bboxc//(32/4)).astype(np.int)[1]*3 + 13*13*3 + 13*13*4*3

    offsets = [offset_b, offset_m, offset_s]

    priors_at_pixel = np.concatenate([
        priors[offset_b:offset_b+3],
        priors[offset_m:offset_m+3],
        priors[offset_s:offset_s+3]
    ])
    # print(priors_at_pixel)

    ious = calculate_ious(priors_at_pixel, bbox)
    index = np.argmax(ious)

    namurge = index%3
    mok = index//3
    index2 = offsets[mok] + namurge

    return np.max(ious), index2

# iou, idid = allocate_bbox_on_anchor(np.array(model.priors), np.array([100, 102, 130, 130]))
# a, b, c = data_to_gt(model.priors, np.array([[10, 102, 20, 110], [100, 102, 140, 130]]), [1, 0], 2)

#%%

# %%

def convert_predict_to_bbox (predicted, prior) :
    # predicted : [tx, ty, tw, th]
    # prior : [ymin, xmin, ymax, xmax]
    e = 1e-9
    apm = 2.0
    pxy, pwh = tf.split(prior, 2)
    bxy, bwh = tf.split(predicted, 2)
    if pwh[0] == pwh[1] :
        stride = (pwh[0] + pwh[1])/(2*apm)
    else :
        stride = (pwh[0] + pwh[1])/(2.5*apm)
    
    pwh -= pxy
    pwh /= stride
    pcenter = pxy + pwh/2
    c = (pcenter - stride*0.5)/stride
    
    xy = 1/(e + 1+tf.exp(-bxy)) + c
    wh = tf.exp(bwh)*stride
    xy *= stride
    xy -= wh/2
    wh += xy
    return tf.concat([xy, wh], axis=0)


def convert_bbox_to_predict (bbox, prior) :
    # bbox : [ymin, xmin, ymax, xmax]
    # prior : [ymin, xmin, ymax, xmax]
    e = 1e-9
    pxy, pwh = np.split(prior, 2)
    bxy, bwh = np.split(bbox, 2)

    pxy = pxy.astype(np.float32)
    bxy = bxy.astype(np.float32)
    pwh = pwh.astype(np.float32)
    bwh = bwh.astype(np.float32)

    pwh -= pxy
    bwh -= bxy

    pcenter = pxy + pwh/2
    bcenter = bxy + bwh/2

    stride = 32

    # print(prior)
    c = (pcenter - stride*0.5)/stride

    bcenter = bcenter/stride

    # to avoid big loss (when a center position is in the boundary of the grid cell)
    txy = np.clip(np.log((bcenter-c+e)/(1-bcenter+c+e)), -4, 4)
    twh = np.log((bwh+e)/(pwh+e))

    return np.concatenate([txy, twh], axis=0)
    

def convert_bbox_to_predict (bbox, prior, stride) :
    # bbox : [ymin, xmin, ymax, xmax]
    # prior : [ymin, xmin, ymax, xmax]
    e = 1e-9
    pxy, pwh = np.split(prior, 2)
    bxy, bwh = np.split(bbox, 2)

    pxy = pxy.astype(np.float32)
    bxy = bxy.astype(np.float32)
    pwh = pwh.astype(np.float32)
    bwh = bwh.astype(np.float32)

    pwh -= pxy
    bwh -= bxy

    pcenter = pxy + pwh/2
    bcenter = bxy + bwh/2

    c = (pcenter - stride*0.5)

    bcenter = (bcenter - c)/stride
    # print(bcenter)

    # bwh /= pwh

    # txy = bcenter
    # twh = bwh
    txy = bcenter + (c//stride)
    twh = bwh
    # txy = np.log(1/(1+np.exp(-bcenter)+e))
    # twh = np.log(bwh+e)

    # print(txy, twh)

    return np.concatenate([txy, twh], axis=0)



#%%

def data_to_gt (priors, bboxes, classes, class_number) :

    b_bbox_input = np.zeros((13, 13, 3, 4))
    b_objness_input = np.zeros((13, 13, 3, 1))
    b_noobjness_input = np.ones((13, 13, 3, 1))
    b_class_input = np.zeros((13, 13, 3, class_number))

    m_bbox_input = np.zeros((26, 26, 3, 4))
    m_objness_input = np.zeros((26, 26, 3, 1))
    m_noobjness_input = np.ones((26, 26, 3, 1))
    m_class_input = np.zeros((26, 26, 3, class_number))

    s_bbox_input = np.zeros((52, 52, 3, 4))
    s_objness_input = np.zeros((52, 52, 3, 1))
    s_noobjness_input = np.ones((52, 52, 3, 1))
    s_class_input = np.zeros((52, 52, 3, class_number))

    strides = [32, 16, 8]
    ii = 0
    for bbox in bboxes :
        i = 0
        pixel_priors = []
        xywh = []
        cc_set = []
        for prior in priors :
            stride = strides[i]
            bbox = np.array(bbox)
            bboxc = (bbox[:2] + bbox[2:4])/2
            cc = (bboxc//stride).astype(np.int)
            # width and height cross
            cc[0], cc[1] = cc[1], cc[0]
            pixel_priors.append(prior[cc[0], cc[1], :, :] + np.concatenate([bboxc - cc*stride, bboxc - cc*stride]))
            xy = bboxc/stride
            wh = bbox[2:4] - bbox[:2]
            xywh.append(np.concatenate([xy, wh]))
            cc_set.append(cc)
            i += 1
        # print(np.concatenate(pixel_priors, axis=0).shape)
        iou = calculate_ious(np.concatenate(pixel_priors, axis=0), bbox)

        # iou2 = np.reshape(calculate_ious(np.reshape(priors[2], (-1, 4)), bbox), (52, 52, 3))
        # print(np.where(iou2 >= 0.5))
        # print(cc_set[2])

        prior_index = np.argmax(iou)
        
        ignore_thr = 0.5

        # print(prior_index)
        if prior_index < 3 :
            i1, i2, i3 = cc_set[0][0], cc_set[0][1], prior_index%3
            
            xywh[0] = np.concatenate([
                np.array(xywh[0]%1)[:2],
                xywh[0][2:4]/(pixel_priors[0][i3][2:4] - pixel_priors[0][i3][:2])
            ])

            ious = np.reshape(calculate_ious(np.reshape(priors[0], (-1, 4)), bbox), (13, 13, 3, 1))
            # width and height cross
            ious = np.transpose(ious, axes=(1, 0, 2, 3))

            b_bbox_input[i1, i2, i3] = xywh[0]
            b_objness_input[i1, i2, i3] = 1
            b_noobjness_input[i1, i2, i3] = 0
            b_noobjness_input += (ious >= ignore_thr)*-1
            b_class_input[i1, i2, i3, classes[ii]] = 1
        elif prior_index < 6 :
            i1, i2, i3 = cc_set[1][0], cc_set[1][1], prior_index%3
            
            xywh[1] = np.concatenate([
                np.array(xywh[1]%1)[:2],
                xywh[1][2:4]/(pixel_priors[1][i3][2:4] - pixel_priors[1][i3][:2])
            ])

            ious = np.reshape(calculate_ious(np.reshape(priors[1], (-1, 4)), bbox), (26, 26, 3, 1))
            # width and height cross
            ious = np.transpose(ious, axes=(1, 0, 2, 3))

            m_bbox_input[i1, i2, i3] = xywh[1]
            m_objness_input[i1, i2, i3] = 1
            m_noobjness_input[i1, i2, i3] = 0
            m_noobjness_input += (ious >= ignore_thr)*-1
            m_class_input[i1, i2, i3, classes[ii]] = 1
        else :
            # print(cc_set[2])
            i1, i2, i3 = cc_set[2][0], cc_set[2][1], prior_index%3

            xywh[2] = np.concatenate([
                np.array(xywh[2]%1)[:2],
                xywh[2][2:4]/(pixel_priors[2][i3][2:4] - pixel_priors[2][i3][:2])
            ])

            ious = np.reshape(calculate_ious(np.reshape(priors[2], (-1, 4)), bbox), (52, 52, 3, 1))
            # width and height cross
            ious = np.transpose(ious, axes=(1, 0, 2, 3))

            # print("=============================")
            # print(i1, i2, i3)
            # print(bbox)
            # print(pixel_priors[2])
            # print(xywh)
            # print("=============================")

            s_bbox_input[i1, i2, i3] = xywh[2]
            s_objness_input[i1, i2, i3] = 1
            s_noobjness_input[i1, i2, i3] = 0
            s_noobjness_input += (ious >= ignore_thr)*-1
            s_class_input[i1, i2, i3, classes[ii]] = 1

        ii += 1

    return [
        b_bbox_input,
        m_bbox_input,
        s_bbox_input
    ], [
        b_class_input,
        m_class_input,
        s_class_input
    ], [
        b_objness_input,
        m_objness_input,
        s_objness_input
    ], [
        b_noobjness_input,
        m_noobjness_input,
        s_noobjness_input
    ]


# a, b, c = data_to_gt(model.priors_set, np.array([[100, 102, 130, 130], [100, 102, 140, 130]]), [1, 0], 2)

# print(a[380])
# print(c[380])
# print(b[380])



def validate_data (model, images_v, gts_v) :

    # print(gts_v)
    bbox_input_v = np.array([v[:, :4] for v in gts_v])
    class_input_v = np.squeeze([v[:, 4:5] for v in gts_v])

    compute_output(model, images_v)

    bboxbbox = bbox_input_v[0]
    print((bboxbbox[:, :2] + bboxbbox[:, 2:4])/2)


#%%

def compute_output (model, image_inputs) :


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

    b_oout = 1/(1+np.exp(-b_oout))
    m_oout = 1/(1+np.exp(-m_oout))
    s_oout = 1/(1+np.exp(-s_oout))

    pred_bboxes = []

    b_indexes = np.squeeze(np.where(b_oout[0] >= 0.5))
    m_indexes = np.squeeze(np.where(m_oout[0] >= 0.5))
    s_indexes = np.squeeze(np.where(s_oout[0] >= 0.5))

    print(b_indexes)
    for v in b_indexes[0] :
        pass


    print(np.sum((b_oout[0] >= 0.5)*1))
    print(np.sum((m_oout[0] >= 0.5)*1))
    print(np.sum((s_oout[0] >= 0.5)*1))
    for i in range(b_oout.shape[0]) :
        try :
            indexes = np.where(b_oout[i] == np.max(b_oout[i]))
            indexes = np.squeeze(indexes)
            # xy coordinate cross
            indexes[0], indexes[1] = indexes[1], indexes[0]
            b_max_bout = b_bout[i, indexes[0], indexes[1], indexes[2]]
            sigmoided = 1 / (1 + np.exp(-b_max_bout[:2]))
            prior = model.priors_set[0][indexes[0], indexes[1], indexes[2]]
            xy = (sigmoided + (prior[:2]/2+prior[2:4]/2)//32)*32
            print(xy)
        except Exception as ee :
            print("error")
            pass
        break
    for i in range(m_oout.shape[0]) :
        try :
            indexes = np.where(m_oout[i] == np.max(m_oout[i]))
            indexes = np.squeeze(indexes)
            # xy coordinate cross
            indexes[0], indexes[1] = indexes[1], indexes[0]
            m_max_bout = m_bout[i, indexes[0], indexes[1], indexes[2]]
            sigmoided = 1 / (1 + np.exp(-m_max_bout[:2]))
            prior = model.priors_set[1][indexes[0], indexes[1], indexes[2]]
            xy = (sigmoided + (prior[:2]/2+prior[2:4]/2)//16)*16
            print(xy)
        except Exception as ee :
            print("error")
            pass
        break
    for i in range(s_oout.shape[0]) :
        try :
            indexes = np.where(s_oout[i] == np.max(s_oout[i]))
            indexes = np.squeeze(indexes)
            # xy coordinate cross
            indexes[0], indexes[1] = indexes[1], indexes[0]
            s_max_bout = s_bout[i, indexes[0], indexes[1], indexes[2]]
            sigmoided = 1 / (1 + np.exp(-s_max_bout[:2]))
            prior = model.priors_set[2][indexes[0], indexes[1], indexes[2]]
            xy = (sigmoided + (prior[:2]/2+prior[2:4]/2)//8)*8
            print(xy)
        except Exception as ee :
            print("error")
            pass
        break


#%%

def calculate_ap (output_data, gt_data) :
    pass



#%%

def save_files_from_final_output (image_input, objness_input_tmp, noobj_input_tmp) :

    tmptmp1 = image_input*255

    tmptmp2 = np.zeros((416, 416))
    tmptmp3 = np.zeros((416, 416))
    for v in objness_input_tmp : 
        v = np.squeeze(np.sum(v, axis=2))*255
        v = np.clip(signal.convolve2d(v, np.ones((3,3))), 0, 255)
        tmptmp2 += cv2.resize(np.squeeze(v), (416, 416), interpolation=cv2.INTER_LINEAR) 
    for vv in noobj_input_tmp :
        vv = 1 - vv
        vv = np.squeeze(np.sum(vv, axis=2))*255
        vv = np.clip(signal.convolve2d(vv, np.ones((3,3))), 0, 255)
        tmptmp3 += cv2.resize(np.squeeze(vv), (416, 416), interpolation=cv2.INTER_LINEAR) 
    tmptmp3 = 255 - np.clip(tmptmp3, 0, 255)

    tmpint = str(np.random.randint(0, 100))

    cv2.imwrite("./tmp_folder/" + tmpint + "_img.png", tmptmp1)
    cv2.imwrite("./tmp_folder/" + tmpint + "_obj.png", tmptmp2)
    cv2.imwrite("./tmp_folder/" + tmpint + "_noobj.png", tmptmp3)
