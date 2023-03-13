import numpy as np
import cv2
import os
import pandas as pd
import json
import sys

sys.path.append('..')

SPECIAL_JSON_FILES = ['X10-239/X10-239-2.json']

IMG_EXTENSION = ".png"

LIST_TYPE = ['jp', 'latin', 'jp-text', 'latin-text', 'text']

SHAPE = ['rect']

CORRECT_TYPE_MAP = {'laitn': 'latin', 'latiin': 'latin', 'lati': 'latin',
                    'jpv': 'jp', 'lain': 'latin', 'textv': 'text',
                    'jp text': 'jp-text', 'jp.text': 'jp-text',
                    'latein-text': 'latin-text', 'lain-text': 'latin-text',
                    'js': 'jp', 'laitn-text': 'latin-text',
                    'latin-tex': 'latin-text', 'lathin': 'latin',
                    'text-latin': 'latin-text', 'laatin': 'latin', 'latn': 'latin', 'jp-texy': 'jp-text',
                    'jo-text': 'jp-text', 'jptext': 'jp-text', 'laten': 'latin', 'jp-txt': 'jp-text',
                    'latim': 'latin', 'ｌatin': 'latin', 'jp-texxt': 'jp-text', 'latin-': 'latin', 'p': 'jp',
                    'tex': 'text', 'texxt': 'text', 'larin': 'latin', 'latinn': 'latin'}

LABEL_NAMES = set(['Label', ' Label', 'Label ', 'Lavel', 'l', 'label', '  Label'])
TYPE_NAMES = set(['Type', 'Type ', 'type'])


def union_boxes(np_arr):
    print('Input union: \n{}'.format(np_arr))
    x1 = np.min(np_arr[:, 0])
    y1 = np.min(np_arr[:, 1])
    x2 = np.max(np_arr[:, 2])
    y2 = np.max(np_arr[:, 3])
    print('Merge: {}'.format([x1, y1, x2, y2]))
    return [x1, y1, x2, y2]


def convert_polygons2rects(np_polygon):
    list_rects = []
    for b in range(np_polygon.shape[0]):
        x1 = np.min(np_polygon[b, :, 0])
        y1 = np.min(np_polygon[b, :, 1])
        x2 = np.max(np_polygon[b, :, 0])
        y2 = np.max(np_polygon[b, :, 1])
        list_rects.append([x1, y1, x2, y2])
    return np.array(list_rects)


def convert_list_imgs_np_array(list_imgs, std_height=64, rgb=False, background_value=0, min_width=40, use_binary=False,
                               use_normalize=False):
    list_std_imgs = []
    max_width = min_width
    for i in range(len(list_imgs)):
        s = 1.0 * std_height / list_imgs[i].shape[0]
        if np.min(list_imgs[i].shape) > 0:
            std_img = cv2.resize(list_imgs[i], (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
            # print('list_imgs[i] shape: {}, std_img: {}'.format(list_imgs[i].shape, std_img.shape))
            if not rgb:
                if len(std_img.shape) == 3 and std_img.shape[2] == 3:
                    # print('std_img shape: {}'.format(std_img.shape))
                    std_img = cv2.cvtColor(std_img, cv2.COLOR_RGB2GRAY)
            elif rgb:
                if len(std_img.shape) == 2:
                    std_img = cv2.cvtColor(std_img, cv2.COLOR_GRAY2RGB)
                elif len(std_img.shape) == 3:
                    std_img = np.squeeze(std_img, -1)
                    if std_img.shape[2] == 1:
                        std_img = cv2.cvtColor(std_img, cv2.COLOR_GRAY2RGB)
                else:
                    raise ValueError('std_img must have 2 or 3 dim, but got {}'.format(len(std_img)))

            if use_binary:
                fil_size = std_height // 4
                if fil_size % 2 == 0:
                    fil_size += 1
                fil_size = max(fil_size, 3)
                std_img = cv2.GaussianBlur(std_img, (fil_size, fil_size), 0)
                std_img = cv2.threshold(std_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                std_img = (std_img / 255).astype(np.uint8)
            elif use_normalize:
                std_img = std_img / 255.0

            if std_img.shape[1] > max_width:
                max_width = std_img.shape[1]
        else:
            std_img = None

        list_std_imgs.append(std_img)

    n_b = len(list_imgs)
    n_h = std_height
    n_w = max_width
    n_c = 1

    batch_imgs = np.ones((n_b, n_h, n_w, n_c), dtype=np.float) * background_value

    for i_b in range(n_b):
        if list_std_imgs[i_b] is None: continue
        h, w = list_std_imgs[i_b].shape[:2]
        batch_imgs[i_b, :h, :w, :] = np.expand_dims(list_std_imgs[i_b], axis=-1)
    return batch_imgs


def apply_sort(ls, sorted_idx):
    if len(ls) != len(sorted_idx):
        raise ValueError(
            'The num of items must equal to the num of indices ({} != {})'.format(len(ls), len(sorted_idx)))

    ls = [ls[ind] for ind in sorted_idx]
    return ls


def cvt_charset_to_ord(file_fullname, out_file_fullname):
    with open(file_fullname, 'rt', encoding='utf-8') as f:
        lines = f.readlines()

    with open(out_file_fullname, 'w') as fout:
        for l in lines:
            l = l.strip()
            if l == '':
                l = ' '
            s = str(ord(l))
            print('char: ({}) -> ({})'.format(l, s))
            fout.write("%s\n" % s)

def get_min_max_xy(cors):
    """
    Get min, max of x, y cordinates
    :param cors: all corners of shape.
    :return: (int, int, int, int)
    Example:
    >>> get_min_max_xy([[1866, 237], [2003, 237], [2003, 269], [1866, 269]])
    (1866, 237, 2003, 269)
    """
    ys = [cor[1] for cor in cors]
    xs = [cor[0] for cor in cors]

    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    return (minx, miny, maxx, maxy)

def calculateIntersection(a0, a1, b0, b1):
    """
    Calculate intersection between two pair of cordinates
    Args:
        -------------
        a0, a1 (a0 < a1): min and max cordinates of line 1
        b0, b1 (b0 < b1): min and max cordinates of line 2
    Return:
        Intersection between two lines: [a0, a1] vs [b0, b1]
    """
    if a0 >= b0 and a1 <= b1:  # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1:  # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0:  # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1:  # Intersects left
        intersection = b1 - a0
    else:  # No intersection (either side)
        intersection = 0

    return intersection

def merge_key_value_smtb(regions):
    merged_regions = []
    limit_distance = 150
    is_merged = [False for i in range(len(regions))]
    for idx, region_1 in enumerate(regions):
        if is_merged[idx] or region_1['region_attributes']['key_type'] != "key" or region_1['shape_attributes']['name'] != 'rect':
            continue
        minx, miny, maxx, maxy = region_1['shape_attributes']["x"], region_1['shape_attributes']["y"],region_1['shape_attributes']["x"] + region_1['shape_attributes']["width"],region_1['shape_attributes']["y"] + region_1['shape_attributes']["height"]

        same_row_textlines = []
        for idx_2, region_2 in enumerate(regions):
            if idx == idx_2 or region_2['region_attributes']['key_type'] != "value" or region_2['shape_attributes']['name'] != 'rect':
                continue

            minX, minY, maxX, maxY = region_2['shape_attributes']["x"], region_2['shape_attributes']["y"],region_2['shape_attributes']["x"] + region_2['shape_attributes']["width"],region_2['shape_attributes']["y"] + region_2['shape_attributes']["height"]
            if minX <= minx or abs(minX - maxx) > limit_distance:
                continue
            overlap_h = calculateIntersection(miny, maxy, minY, maxY)
            if overlap_h / (maxy - miny) > 0.4:  
                same_row_textlines.append((minX, idx_2, minX, minY, maxX, maxY))
        
        if len(same_row_textlines):
            same_row_textlines.sort(key = lambda x : x[0])
            is_merged[same_row_textlines[0][1]] = True
            is_merged[idx] = True

            minX, minY, maxX, maxY = same_row_textlines[0][2], same_row_textlines[0][3], same_row_textlines[0][4], same_row_textlines[0][5] 
            new_minx = min(minx, minX)
            new_miny = min(miny, minY)
            new_maxx = max(maxx, maxX)
            new_maxy = max(maxy, maxY)
            
            merged_regions.append({
                    "shape_attributes": {
                        "name": "rect",
                        "x": new_minx,
                        "y": new_miny,
                        "width": new_maxx - new_minx,
                        "height": new_maxy - new_miny
                    },
                    "region_attributes": {
                        "label": "",
                        "key_type": "key_value",
                        "text_category": "mix",
                        "text_type": "printed",
                        "formal_key": "",
                        "note": ""
                    }
                })
    
    for i in range(len(regions)):
        if not is_merged[i]:
            merged_regions.append(regions[i])
    
    return merged_regions

def measure_np_iou(bboxes1, bboxes2, mode='iou'):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # for i in range(x11.shape[0]):
    #     if x11[i] > x12[i] or y11[i] > y12[i]:
    #         raise ValueError("Invalid boxes: {}".format(bboxes1[i, :]))

    # for i in range(x21.shape[0]):
    #     if x21[i] > x22[i] or y21[i] > y22[i]:
    #         raise ValueError("Invalid boxes: {}".format(bboxes2[i, :]))

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    if mode == 'min':
        minArea = np.minimum((x12 - x11 + 1) * (y12 - y11 + 1), np.transpose((x22 - x21 + 1) * (y22 - y21 + 1)))
        iou = interArea / minArea
    else:
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou