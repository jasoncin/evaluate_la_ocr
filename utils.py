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

#
# def parse_master_label(filename, kw_file_fullname=None, save_csv=True, min_iou=0.65, actual_failed_min_iou=1000.0):
#     df = pd.DataFrame()
#     list_labels = []
#
#     list_nolabel_dict = []
#
#     if kw_file_fullname and os.path.exists(kw_file_fullname):
#         with open(kw_file_fullname) as f:
#             list_kws = f.readlines()
#     else:
#         list_kws = None
#
#     list_kws = [normalize_text(kw.strip()) for kw in list_kws]
#     print('list_kws: {}'.format(list_kws))
#
#     json_base = os.path.basename(os.path.basename(filename))
#
#     list_wrong_type = []
#     if filename.split('\\')[-1] in SPECIAL_JSON_FILES:
#         json_file = json.load(codecs.open(filename, 'r', 'utf-8-sig'))
#     else:
#         with open(filename, 'rt', encoding='utf-8') as f:
#             json_file = json.load(f)
#
#     # basename = os.path.splitext(os.path.basename(filename))[0]
#     doc_dict = dict()
#
#     for key in json_file:
#         label_dict = json_file[key]
#         img_name = label_dict['filename']
#         print('Base img name: {}, base json name: {}'.format(img_name.split('.')[0], json_base.split('.')[0]))
#         if not img_name.endswith(IMG_EXTENSION) or img_name.split('.')[0] != json_base.split('.')[0]:
#             print('Skip image {} for not relate to {}'.format(img_name, json_base))
#             continue
#         else:
#             print('Processing {}'.format(img_name))
#
#         _size = label_dict['size']
#
#         # List
#         regions = label_dict['regions']
#         df = pd.DataFrame(regions)
#         list_rows = []
#         non_label_count = {'jp': 0, 'jp-text': 0, 'latin': 0, 'latin-text': 0, 'text': 0}
#         total_label_count = {'jp': 0, 'jp-text': 0, 'latin': 0, 'latin-text': 0, 'text': 0}
#         label_count = 0
#         list_rects = []
#
#         for ind, reg in enumerate(regions):
#             row = dict()
#
#             has_label = False
#             for key, value in reg['region_attributes'].items():
#                 if key in LABEL_NAMES:
#                     _label = normalize_text(value.strip())
#                     has_label = True
#                     break
#
#             if 'Type' in reg['region_attributes']:
#                 _type = reg['region_attributes']['Type'].lower()
#             elif 'type' in reg['region_attributes']:
#                 _type = reg['region_attributes']['type'].lower()
#             else:
#                 _type = 'text'
#
#             if has_label:
#                 for k in LIST_TYPE:
#                     if k == _type:
#                         if not _label:
#                             non_label_count[k] += 1
#                         total_label_count[k] += 1
#                         break
#             else:
#                 _label = ''
#                 # raise ValueError("Don't have label: {}".format(filename))
#
#             # _type = reg['region_attributes']['Type']
#             _shape = reg['shape_attributes']['name']
#
#             # print(_shape)
#             if _shape not in SHAPE:
#                 print('>> Skip because {} is not in {}'.format(_shape, SHAPE))
#                 # raise ValueError('Invalid shape!')
#                 continue
#
#             _x1 = reg['shape_attributes']['x']
#             _y1 = reg['shape_attributes']['y']
#             _x2 = reg['shape_attributes']['width'] + _x1 - 1
#             _y2 = reg['shape_attributes']['height'] + _y1 - 1
#
#             label_count += 1
#             if _type:
#                 _type = _type.strip()
#             valid_script = 1
#             if _type:
#                 if not _type in LIST_TYPE:
#                     # print('[WARN] INVALID type: {}'.format(_type))
#                     if _type in CORRECT_TYPE_MAP:
#                         valid_script = 1
#                         _type_ = CORRECT_TYPE_MAP[_type]
#                         # print('[INFO] Correct {} to {}'.format(_type, _type_))
#                         _type = _type_
#                     else:
#                         valid_script = 0
#                         list_wrong_type.append(_type)
#                         print('[ERR] Cannot find correction of {}'.format(_type))
#
#             _label = normalize_text(_label)
#             kws = [{"text": normalize_text(kw), "score": 1.0} for kw in list_kws if kw in _label]
#
#             str_ind = ""
#             if 'text' in _type:
#                 str_ind = "img%02d" % ind
#             else:
#                 str_ind = "box%02d" % ind
#
#             row = {"index": str_ind, "father_index": "", "text": _label, "score": 1.0, "kws": kws, "script": _type,
#                    "valid_script": valid_script, "x1": _x1, "y1": _y1, "x2": _x2, "y2": _y2, "shape": _shape}
#             list_rows.append(row)
#             list_rects.append([_x1, _y1, _x2, _y2])
#
#         arr_rect = np.stack(list_rects, axis=0)
#         arr_iou = measure_np_iou(arr_rect, arr_rect, mode='min')
#
#         # Ignore self comparisons
#         np.fill_diagonal(arr_iou, 0.0)
#
#         # print("arr_iou: {}".format(arr_iou))
#         print("arr_iou shape: {}".format(arr_iou.shape))
#
#         # raise ValueError('Stop debugging!')
#         num_row = len(list_rows)
#         for c, rect_c in enumerate(list_rects):
#             if "text" not in list_rows[c]["script"]:
#                 arr_iou[:, c] = 0.0
#
#         # arr_iou[arr_iou < min_iou] = 0.0
#         max_iou_idx = np.argmax(arr_iou, axis=1)
#         print("max_idx: {}".format(np.max(arr_iou, axis=1)))
#         for r, rect_r in enumerate(list_rects):
#             if "text" not in list_rows[r]["script"]:
#                 best_iou = arr_iou[r, max_iou_idx[r]]
#                 if best_iou > min_iou:
#                     if list_rows[max_iou_idx[r]]["index"] == list_rows[r]["index"]:
#                         raise ValueError("[WARN] >>> Incest detected {}!".format(list_rows[r]["index"]))
#                     list_rows[r]["father_index"] = list_rows[max_iou_idx[r]]["index"]
#
#                     print("Father of {} is {} with IoU min: {}".format(list_rows[r]["index"],
#                                                                        list_rows[max_iou_idx[r]]["index"], best_iou))
#                 else:
#                     # print('actual_failed_min_iou: {}, best_iou: {}'.format(actual_failed_min_iou, best_iou))
#                     if actual_failed_min_iou > best_iou and best_iou > 0.5:
#                         actual_failed_min_iou = best_iou
#                     # print('actual_failed_min_iou: {}, best_iou: {}'.format(actual_failed_min_iou, best_iou))
#                     print("[WARN] {} has no father because max IoU {} < {}".format(list_rows[r]["index"], best_iou,
#                                                                                    min_iou))
#             else:
#                 print("Skip father {}".format(list_rows[r]["index"]))
#
#         for i, row in enumerate(list_rows):
#             if row["father_index"] == "" or "text" in row["script"]:
#                 list_children = []
#                 for j in range(len(list_rows)):
#                     if list_rows[j]["father_index"] == row["index"]:
#                         list_children.append(list_rows[j])
#                 sorted_list_children = sorted(list_children, key=lambda k: k['x1'])
#
#                 list_text = ""
#                 list_kws = []
#                 for rw in sorted_list_children:
#                     list_text += rw["text"]
#                     list_kws += [kw["text"] for kw in rw["kws"]]
#
#                 if list_text:
#                     list_rows[i]["text"] = normalize_text(''.join(list_text))
#                     list_rows[i]["kws"] = [{"text": tx, "score": 1.0} for tx in list(set(list_kws))]
#                 # list_rows[i]["text"] = ''.join([rw["text"] for rw in sorted_list_children])
#
#         for i, row in enumerate(list_rows):
#             doc_dict[row["index"]] = row
#
#         # Exit the loop
#         break
#
#     print('actual_failed_min_iou: {}'.format(actual_failed_min_iou))
#     # df = pd.DataFrame(list_rows)
#     return doc_dict, actual_failed_min_iou
#

def measure_np_iou(bboxes1, bboxes2, mode='iou'):
    # print("bboxes1", bboxes1)
    # print("bboxes2", bboxes2)
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