# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import argparse
import time
from tqdm import tqdm

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def non_max_suppression(boxes, confidences, iou_threshold):
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2
    
    # if probabilities are provided, sort on them instead
    if confidences is not None:
        idxs = confidences

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                                np.where(overlap > iou_threshold)[0])))
    # return the indices of the picked bounding boxes
    return pick


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', required=True, type=str, help='inference model path')
    parser.add_argument('--classes', '-c', required=True, type=str, help='class list file (same as training)')
    parser.add_argument('--test', '-t', required=True, type=str, help='testing file list')
    parser.add_argument('--output', '-o', required=True, type=str, help='saved path')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='nms iou threshold')
    parser.add_argument('--conf_threshold', default=0.25, type=float, help='confidence threshold')
    args = parser.parse_args()
    return args

def read_conf(testing_file, class_file):
    # test list
    with open (testing_file, "r") as f:
        test_list = f.read().strip().split()
    # class list
    labels_to_names = {}
    with open(class_file, "r") as f:
        data = f.read().strip().split("\n")
    for item in data:
        val, key = item.split(",")
        labels_to_names[int(key)] = val
    return test_list, labels_to_names


if __name__ == "__main__":
    args = get_args()
    # make directory for saving files
    os.makedirs(args.output, exist_ok=True)
    # read config file
    test_list, labels_to_names = read_conf(args.test, args.classes)
    # load model
    model = models.load_model(args.model, backbone_name='resnet50')

    # inference start
    for fname in tqdm(test_list):
        image = read_image_bgr(fname)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # process image
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        # correct for image scale
        boxes /= scale
        # NMS for each classes
        for i in range(len(labels_to_names)):
            idxs = (labels == i)
            box_i = boxes[idxs]
            score_i = scores[idxs]
            label_i = labels[idxs]
            anchors_nms_idx = non_max_suppression(box_i, score_i, args.iou_threshold)

            box_i = box_i[anchors_nms_idx]
            score_i = score_i[anchors_nms_idx]
            label_i = label_i[anchors_nms_idx]
            # draw
            for box, score, label in zip(box_i, score_i, label_i):
                # scores are sorted so we can break
                if score < args.conf_threshold:
                    break
                color = label_color(label)
                b = box.astype(int)
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)
        new_fname = os.path.basename(fname)
        abs_new_path = os.path.join(args.output, new_fname)
        cv2.imwrite(abs_new_path, draw)
