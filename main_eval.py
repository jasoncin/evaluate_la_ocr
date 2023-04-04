# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import os
import json
import numpy as np
import utils as utils
# from ocr.cannet.model import CannetOCR
# from ocr import JeffOCR
import cv2
import pandas as pd
import datetime
from operator import itemgetter
from PIL import Image, ImageDraw, ImageFont
from shutil import copyfile


LIST_TYPE = ['jp', 'latin', 'jp-text', 'latin-text', 'text']


def get_info_from_gt_dict(_dict, script=[], ignore_formal_key=[], filter_vertical=True, filter_horizontal=False):
    idx2key = []
    list_rect = []

    if 'regions' in _dict:
        region_list = _dict['regions']
    else:
        region_list = _dict['attributes']['_via_img_metadata']['regions']

    # MERGE H6
    # region_list = utils.merge_key_value_smtb(region_list)

    for i, region in enumerate(region_list):
        if "key_type" in region['region_attributes']:
            region['region_attributes']['structure_type'] = region['region_attributes']['key_type']

        if "label" in region['region_attributes']:
            region['region_attributes']['text'] = region['region_attributes']['label']

        key_type = region['region_attributes']['structure_type']
        fm_key = region['region_attributes']['formal_key']

        # FILTER vertical textline
        if filter_vertical:
            if region['shape_attributes']['name'] == 'rect' and region['shape_attributes']["height"] / region['shape_attributes']["width"] >= 2:
                continue
        
        if filter_horizontal:
            if region['shape_attributes']['name'] == 'rect' and region['shape_attributes']["width"] / region['shape_attributes']["height"] >= 2:
                continue

        if key_type.strip() in script and region['shape_attributes']['name'] == 'rect' and fm_key not in ignore_formal_key:
            idx2key.append(region)
            list_rect.append(np.array([region['shape_attributes']["x"], region['shape_attributes']["y"],
                                       region['shape_attributes']["x"] + region['shape_attributes']["width"],
                                       region['shape_attributes']["y"] + region['shape_attributes']["height"]
                                       ]))
    
    if list_rect:
        arr = np.stack(list_rect, axis=0)
    else:
        arr = np.array([])
    return arr, idx2key


def get_info_from_pred_dict(_dict, script=""):
    idx2key = []
    list_rect = []
    script = script.strip()

    for i, region in enumerate(_dict):
        try:
            list_rect.append(np.array([region['location'][0][0], region['location'][0][1],
                                    region['location'][2][0], region['location'][2][1]
                                    ]))
            idx2key.append(region)
        except Exception as e:
            print(e)
            continue

    if list_rect:
        arr = np.stack(list_rect, axis=0)
    else:
        arr = np.array([])
    return arr, idx2key


def match_bboxes(gt_dict, pred_dict, script="", filter_vertical=True, filter_horizontal=False):
    arr_gt_rect, gt_i2k = get_info_from_gt_dict(gt_dict, script=script, filter_vertical=filter_vertical, filter_horizontal=filter_horizontal)
    arr_pred_rect, pred_i2k = get_info_from_pred_dict(pred_dict)

    if (len(arr_gt_rect) > 0 and len(arr_pred_rect) > 0):
        iou_gt_pred = utils.measure_np_iou(arr_gt_rect, arr_pred_rect, mode='union')
        # iou_min_gt_pred = util.measure_np_iou(arr_gt_rect, arr_pred_rect, mode='min')
    else:
        print(len(arr_gt_rect), len(arr_pred_rect))
        return [], [], [], []
    # print("iou_gt_pred:\n{}\n".format(iou_gt_pred))
    # print("iou_gt_pred shape: {}".format(iou_gt_pred.shape))

    idx_max_iou_pred = iou_gt_pred.argmax(axis=1)

    list_pair_idx_gt_pred = []
    for gt_i in range(iou_gt_pred.shape[0]):
        # if iou_gt_pred[gt_i, idx_max_iou_pred[gt_i]] > min_iou:
        list_pair_idx_gt_pred.append([gt_i, idx_max_iou_pred[gt_i]])

    list_pair_keys_gt_pred = []
    list_text_dist = []
    list_iou = []
    for pair in list_pair_idx_gt_pred:
        gt_i = pair[0]
        pred_i = pair[1]
        gt_text = gt_i2k[gt_i]['region_attributes']['text']
        pred_text = pred_i2k[pred_i]['text']
        # gt_script = gt_dict[gt_i2k[gt_i]]["script"]
        # pred_script = pred_dict[pred_i2k[pred_i]]["script"]
        list_pair_keys_gt_pred.append({"gt": gt_i2k[gt_i], "pred": pred_i2k[pred_i], 'iou': iou_gt_pred[gt_i, pred_i]})

        # list_text_dist.append(textdistance.levenshtein.distance(gt_text, pred_text))
        list_iou.append(iou_gt_pred[gt_i, pred_i])

        # print("IoU: {}, gt_script: {}, pred_script: {}\ngt_text:   {}\npred_text: {}\n".format(iou_gt_pred[gt_i, pred_i], gt_script, pred_script, gt_text, pred_text))

    return list_pair_keys_gt_pred


def _reformat_dict(model, list_pair, img):
    list_pair_processed = []
    if len(list_pair[0]) > 0:
        for pair in list_pair:
            try:
                gt_coor = pair['gt']['shape_attributes']
                pred_coor = pair['pred']['location']
                dict_info = {
                    'gt': {
                        'coor': [gt_coor['y'], gt_coor['y'] + gt_coor['height'], gt_coor['x'],
                                 gt_coor['x'] + gt_coor['width']],
                        # y_top, y_bot, x_left, x_right
                        'label': pair['gt']['region_attributes']['text'],
                        'img': None,
                        'key_type': pair['gt']['region_attributes']['structure_type'],
                        'info': pair['gt'],
                        'text_pr': ''  # text was predicted by same ocr model with LA predicted model
                    },
                    'pred': {
                        'coor': [pred_coor[0][1], pred_coor[2][1], pred_coor[0][0], pred_coor[2][0]],
                        # y_top, y_bot, x_left, x_right
                        'text': pair['pred']['text'],
                        'img': None,
                        'key_type': '',
                        'info': pair['pred'],
                        'text_pr': ''  # text was predicted by same ocr model with LA predicted model
                    },
                    'iou': pair['iou'],
                    'key_type': pair['gt']['region_attributes']['structure_type']
                }
                if pair['iou'] > 0.01:
                    # crop image
                    dict_info['gt']['img'] = img[dict_info['gt']['coor'][0]: dict_info['gt']['coor'][1],
                                             dict_info['gt']['coor'][2]:dict_info['gt']['coor'][
                                                 3]]  # get format [y_top:y_bottom, x_left:x_right]
                    dict_info['pred']['img'] = img[dict_info['pred']['coor'][0]: dict_info['pred']['coor'][1],
                                               dict_info['pred']['coor'][2]:dict_info['pred']['coor'][
                                                   3]]  # get format [y_top:y_bottom, x_left:x_right]
                    # import pdb; pdb.set_trace()
                    
                    # dict_info['pred']['text_pr'] = model.process(dict_info['pred']['img'])['text']
                    dict_info['gt']['text_pr'] = model.process(dict_info['gt']['img'])['text']
                    # dict_info['gt']['text_pr'] = ""

                    dict_info['is_la_ok'] = (dict_info['gt']['label'] == dict_info['pred']['text'] and pair['iou'] > 0.3) or (pair['iou'] > 0.6)
                    # dict_info['is_la_ok'] =  (pair['iou'] > 0.6)
                    
                    # if dict_info['gt']['key_type'] == "key":
                    #     dict_info['is_la_ok'] =  (pair['iou'] > 0.6)
                    # else:
                    #     dict_info['is_la_ok'] =  (pair['iou'] > 0.6) 
                                     
                    dict_info['is_la_ocr_ok'] = (dict_info['gt']['label'] == dict_info['pred']['text']) and (pair['iou'] > 0.3)
                    dict_info['is_ocr_only_ok'] = (dict_info['gt']['label'].strip() == dict_info['gt']['text_pr'].strip())
                else:
                    dict_info['is_la_ok'] = False
                    dict_info['is_la_ocr_ok'] = False
                    dict_info['is_ocr_only_ok'] = False
                list_pair_processed.append(dict_info)
            except Exception as e:
                print(e)
    return list_pair_processed


def makedir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def _update_excel_file(list_result, out_dir):
    list_detail = []  # for saving data to excel file
    list_dict_metric = []
    dict_metric_general = {}
    dict_metric_fomalkey = {}

    for result in list_result:
        list_order = sorted(result['result'], key=itemgetter('key_type'))
        dict_metric = {
            'basename': result['basename'],
            'key_type': {},
            'total': {
                'lc_ok': 0,
                'ocr_ok': 0,
                'ocr_only_ok': 0,
                'total': 0
            }
        }
        for i, detail in enumerate(list_order):
            output_update_hlayout_24_12_row = [
                result['basename'], i, detail['gt']['key_type'], detail['gt']['info']['region_attributes']['formal_key'], detail['iou'],
                detail['gt']['text_pr'], detail['pred']['text_pr'],
                detail['pred']['text'], detail['gt']['label'],
                detail['is_la_ok'],
                detail['is_la_ocr_ok'],
                detail['is_ocr_only_ok']
            ]
            list_detail.append(output_update_hlayout_24_12_row)

            #process metric
            if dict_metric['key_type'].get(detail['key_type']) is None:
                dict_metric['key_type'][detail['key_type']] = {
                    'lc_ok': 0,
                    'ocr_ok': 0,
                    'ocr_only_ok': 0,
                    'total': 0
                }
            if dict_metric_general.get(detail['key_type']) is None:
                dict_metric_general[detail['key_type']] = {
                    'lc_ok': 0,
                    'ocr_ok': 0,
                    'ocr_only_ok': 0,
                    'total': 0
                }
            if len(detail['gt']['info']['region_attributes']['formal_key'])>0:
                if dict_metric_fomalkey.get(detail['gt']['info']['region_attributes']['formal_key']) is None:
                    dict_metric_fomalkey[detail['gt']['info']['region_attributes']['formal_key']] = {
                        'key': {
                            'lc_ok': 0,
                            'ocr_ok': 0,
                            'ocr_only_ok': 0,
                            'total': 0
                        },
                        'value': {
                            'lc_ok': 0,
                            'ocr_ok': 0,
                            'ocr_only_ok': 0,
                            'total': 0
                        },
                        'common_key': {
                            'lc_ok': 0,
                            'ocr_ok': 0,
                            'ocr_only_ok': 0,
                            'total': 0
                        }
                    }
            if detail['is_la_ok']:
                dict_metric['key_type'][detail['key_type']]['lc_ok'] += 1
                dict_metric['total']['lc_ok'] +=1
                dict_metric_general[detail['key_type']]['lc_ok'] +=1
                if len(detail['gt']['info']['region_attributes']['formal_key']) > 0:
                    dict_metric_fomalkey[detail['gt']['info']['region_attributes']['formal_key']][detail['key_type']]['lc_ok'] +=1
            if detail['is_la_ocr_ok']:
                dict_metric['key_type'][detail['key_type']]['ocr_ok'] += 1
                dict_metric['total']['ocr_ok'] += 1
                dict_metric_general[detail['key_type']]['ocr_ok'] += 1
                if len(detail['gt']['info']['region_attributes']['formal_key']) > 0:
                    dict_metric_fomalkey[detail['gt']['info']['region_attributes']['formal_key']][detail['key_type']]['ocr_ok'] +=1
            if detail['is_ocr_only_ok']:
                dict_metric['key_type'][detail['key_type']]['ocr_only_ok'] += 1
                dict_metric['total']['ocr_only_ok'] += 1
                dict_metric_general[detail['key_type']]['ocr_only_ok'] += 1
                if len(detail['gt']['info']['region_attributes']['formal_key']) > 0:
                    dict_metric_fomalkey[detail['gt']['info']['region_attributes']['formal_key']][detail['key_type']][
                        'ocr_only_ok'] += 1
            dict_metric['key_type'][detail['key_type']]['total'] += 1
            dict_metric['total']['total'] += 1
            dict_metric_general[detail['key_type']]['total'] += 1
            if len(detail['gt']['info']['region_attributes']['formal_key']) > 0:
                dict_metric_fomalkey[detail['gt']['info']['region_attributes']['formal_key']][detail['key_type']][
                    'total'] += 1

        # add info to metric
        list_dict_metric.append(dict_metric)

    #write to general
    labels = [
        'base_name', 'index', 'key_type', 'kv_name', 'iou',
        'ocr_gt_LA', 'ocr_pred_LA',
        'ocr_full_flow','qa_label',
        'is_lc_ok',
        'is_la_ocr_ok', 'is_ocr_only_ok'
    ]
    df = pd.DataFrame.from_records(list_detail, columns=labels)

    # write to result metric of pages
    value_metric = []
    labels_metric = ['basename', 'type', 'lc_ok', 'ocr_ok', 'ocr_only_ok', 'total', 'acc_lc', 'accBF_ocr', 'accBF_ocr_only']
    for dict_metric in list_dict_metric:
        if dict_metric['total']['total'] == 0:
            dict_metric['total']['total'] = 0.01
        for key_ind in dict_metric['key_type'].keys():
            key = dict_metric['key_type'][key_ind]
            value_metric.append([dict_metric['basename'], key_ind, key['lc_ok'], key['ocr_ok'], key['ocr_only_ok'],key['total'],
                                 key['lc_ok']/key['total'], key['ocr_ok']/key['total'],
                                 key['ocr_only_ok']/key['total']])
        value_metric.append([dict_metric['basename'], 'total', dict_metric['total']['lc_ok'], dict_metric['total']['ocr_ok'],
                             dict_metric['total']['ocr_only_ok'],
                             dict_metric['total']['total'],
                             dict_metric['total']['lc_ok'] / dict_metric['total']['total'],
                             dict_metric['total']['ocr_ok'] / dict_metric['total']['total'],
                             dict_metric['total']['ocr_only_ok'] / dict_metric['total']['total']])
    df_metric_page = pd.DataFrame.from_records(value_metric, columns=labels_metric)

    # write to result metric of general
    labels_metric_gen = ['type', 'lc_ok', 'ocr_ok', 'ocr_only_ok', 'total', 'acc_lc', 'accBF_ocr', 'accBF_ocr_only']
    value_metric_general = []
    for key in dict_metric_general.keys():
        value_metric_general.append([key, dict_metric_general[key]['lc_ok'], dict_metric_general[key]['ocr_ok'],
                                     dict_metric_general[key]['ocr_only_ok'],
                                     dict_metric_general[key]['total'],
                                     dict_metric_general[key]['lc_ok']/dict_metric_general[key]['total'],
                                     dict_metric_general[key]['ocr_ok']/dict_metric_general[key]['total'],
                                     dict_metric_general[key]['ocr_only_ok']/dict_metric_general[key]['total']])
    df_metric_general = pd.DataFrame.from_records(value_metric_general, columns=labels_metric_gen)

    # write to result of formal key
    value_metric_fomal = []
    labels_metric_fomal = ['formal_key', 'key_type', 'lc_ok', 'ocr_ok', 'ocr_only_ok', 'total', 'acc_lc', 'accBF_ocr', 'accBF_ocr_only']
    for formal_key in dict_metric_fomalkey:
        if dict_metric_fomalkey[formal_key]['key']['total'] == 0:
            dict_metric_fomalkey[formal_key]['key']['total'] = 0.01
        if dict_metric_fomalkey[formal_key]['value']['total'] == 0:
            dict_metric_fomalkey[formal_key]['value']['total'] = 0.01
        value_metric_fomal.append([formal_key, 'key', dict_metric_fomalkey[formal_key]['key']['lc_ok'],
                                   dict_metric_fomalkey[formal_key]['key']['ocr_ok'],
                                   dict_metric_fomalkey[formal_key]['key']['ocr_only_ok'],
                                  dict_metric_fomalkey[formal_key]['key']['total'],
                                  dict_metric_fomalkey[formal_key]['key']['lc_ok']/dict_metric_fomalkey[formal_key]['key']['total'],
                                   dict_metric_fomalkey[formal_key]['key']['ocr_ok']/dict_metric_fomalkey[formal_key]['key']['total'],
                                   dict_metric_fomalkey[formal_key]['key']['ocr_only_ok']/dict_metric_fomalkey[formal_key]['key']['total']])
        value_metric_fomal.append([formal_key, 'value', dict_metric_fomalkey[formal_key]['value']['lc_ok'],
                                   dict_metric_fomalkey[formal_key]['value']['ocr_ok'],
                                   dict_metric_fomalkey[formal_key]['value']['ocr_only_ok'],
                                   dict_metric_fomalkey[formal_key]['value']['total'],
                                   dict_metric_fomalkey[formal_key]['value']['lc_ok'] /
                                   dict_metric_fomalkey[formal_key]['value']['total'],
                                   dict_metric_fomalkey[formal_key]['value']['ocr_ok'] /
                                   dict_metric_fomalkey[formal_key]['value']['total'],
                                   dict_metric_fomalkey[formal_key]['value']['ocr_only_ok'] /
                                   dict_metric_fomalkey[formal_key]['value']['total']
                                   ])
    df_metric_formalkey = pd.DataFrame.from_records(value_metric_fomal, columns=labels_metric_fomal)

    # write all data to excel file
    result_filename = os.path.join(out_dir, "Fullflow_result")
    writer = pd.ExcelWriter(result_filename + '.xlsx')
    df.to_excel(writer, 'Detail')
    df_metric_page.to_excel(writer, 'Metric_pages')
    df_metric_formalkey.to_excel(writer, 'Metric_formalKey')
    df_metric_general.to_excel(writer, 'Metric_general')
    writer.save()


def _write_debug_image(list_result, out_dir):
    debug_dir = os.path.join(out_dir, 'debug_image_1')
    makedir(debug_dir)
    unicode_font = ImageFont.truetype('Roboto-Light.ttf', 20)
    for result in list_result:
        img = result['img']
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img, 'RGB')
        elif isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                pil_img = Image.open(img).convert('RGB')
            else:
                pil_img = Image.fromarray(img, 'RGB')
        else:
            raise ValueError('Not support type {}'.format(type(img)))
        draw = ImageDraw.Draw(pil_img)
        list_order = sorted(result['result'], key=itemgetter('key_type'))
        for i, detail in enumerate(list_order):
            draw.rectangle([detail['gt']['coor'][2], detail['gt']['coor'][0], detail['gt']['coor'][3], detail['gt']['coor'][1]],
                           outline = (255,0,0), fill=None, width=3)
            if detail['is_la_ok']:
                if (not detail['is_la_ocr_ok']):
                    color = (0,0,255)
                else:
                    color = (0, 255, 0)  #green
            else:
                color = (0, 255, 255) #cyan

            draw.rectangle(
                [detail['pred']['coor'][2], detail['pred']['coor'][0], detail['pred']['coor'][3], detail['pred']['coor'][1]],
                outline=color, fill=None, width=3)
            color_index = (128,0,128) #purple for not value key_type
            if detail['key_type'] == 'value':
                color_index = (255,0,0) #red for value key_type
            draw.text([detail['gt']['coor'][3]+3, detail['pred']['coor'][0]], str(i), font=unicode_font, fill=color_index, align='center')

        del draw
        output_update_hlayout_24_12_fullname = os.path.join(debug_dir, result['basename']+'.png')
        pil_img.save(output_update_hlayout_24_12_fullname)

def expand_textline_wh(linecut_data, pad_height = 3, pad_width = 5):
    for linecut in linecut_data:
        loc = linecut['location']
        minx = loc[0][0]
        miny = loc[0][1]
        maxx = loc[2][0]
        maxy = loc[2][1]

        linecut['location'] = [[minx - pad_width, miny - pad_height], [maxx + pad_width, miny - pad_height], [maxx + pad_width, maxy + pad_height], [minx - pad_width, maxy + pad_height]]
    return linecut_data

def convert_to_ocr_output_dir(input_path, output_path):
    for img_name in os.listdir(input_path):
        if '.DS_Store' in img_name:
            continue
        try:
            copyfile(os.path.join(os.path.join(input_path, img_name), "ocr_output.json"),
                    os.path.join(output_path, "{}.json".format(img_name)))
        except Exception:
            copyfile(os.path.join(os.path.join(input_path, img_name), "ocr.json"),
                    os.path.join(output_path, "{}.json".format(img_name)))
        
# , "common_key", "key_value"
def _evaluate_dir(model, gt_dir, pred_dir, out_dir, image_root="", list_group_script=["key", "value"],
                  list_min_iou=[0.6], debug=True, filter_vertical=True, filter_horizontal=True):
    import tempfile

    ocr_output_dir = None
    if pred_dir.endswith("results") or pred_dir.endswith("results/"):
        ocr_output_dir = tempfile.mkdtemp()

        convert_to_ocr_output_dir(pred_dir, ocr_output_dir)
    
    pred_dir = ocr_output_dir

    list_gt_files = []
    list_pred_files = []

    for _root, _dirs, _files in os.walk(gt_dir):
        for f in _files:
            if f.endswith('.json'):
                filename = os.path.join(_root, f)
                list_gt_files.append(filename)

    dict_bn_img = dict()

    if os.path.exists(image_root):
        list_gt_basenames = []
        for gt_fullname in list_gt_files:
            basename = os.path.splitext(os.path.basename(gt_fullname))[0]
            list_gt_basenames.append(basename)
        for _root, _dirs, _files in os.walk(image_root):
            for f in _files:
                # print(f)
                # if f.endswith(IMG_EXTENSION):
                img_basename = os.path.splitext(os.path.basename(f))[0]
                if img_basename in list_gt_basenames:
                    dict_bn_img[img_basename] = os.path.join(_root, f)

    for _root, _dirs, _files in os.walk(pred_dir):
        for f in _files:
            if f.endswith('.json'):
                filename = os.path.join(_root, f)
                list_pred_files.append(filename)

    print("Total Groundtruth JSON: {}".format(len(list_gt_files)))
    print("Total Prediction JSON: {}".format(len(list_pred_files)))

    # if len(list_gt_files) > 1000:
    #     return
        
    list_final_result = []

    for gt_file in list_gt_files:
        dict_result = {}
        basename = os.path.splitext(os.path.basename(gt_file))
        # basename = ('2019080820400581-038_1', '.json')
        matched_pred_file = None
        if os.path.exists(os.path.join(pred_dir, basename[0] + '.png' + basename[1])):
            matched_pred_file = basename[0] + '.png' + basename[1]
        elif os.path.exists(os.path.join(pred_dir, basename[0] + basename[1])):
            matched_pred_file = basename[0] + basename[1]
        try:
            # img = np.asarray(Image.open(dict_bn_img[basename[0]]))

            img = cv2.imread(dict_bn_img[basename[0]])
            if img is None:
                pil_img = Image.open(dict_bn_img[basename[0]]).convert('RGB')
            else:
                pil_img = Image.fromarray(img, 'RGB')
        except Exception as e:
            print(e)
            continue

        # img = cv2.imread(dict_bn_img[basename[0]])
        dict_result['img'] = img
        dict_result['basename'] = basename[0]
        if matched_pred_file is not None:
            # print("Opening ",matched_pred_file)
            try:
                with open(gt_file, 'rt', encoding='utf-8') as f:
                    gt_dict = json.load(f)
                with open(os.path.join(pred_dir, matched_pred_file), 'rt', encoding='utf-8') as f:
                    pred_dict = json.load(f)

                    padded_pred_dict = expand_textline_wh(pred_dict)
            except Exception as e:
                print(e, matched_pred_file)
                continue
            #measure the iou
            list_pair_keys_gt_pred = match_bboxes(gt_dict, pred_dict, script=list_group_script, filter_vertical=filter_vertical, filter_horizontal=filter_horizontal)

            #reformat data and run ocr
            list_pair_keys_gt_pred = _reformat_dict(model, list_pair_keys_gt_pred, img)

            #update result to dict
            dict_result['result'] = list_pair_keys_gt_pred
            list_final_result.append(dict_result)

            # import pdb; pdb.set_trace()

    ### UPDATE RESULT TO EXCEL FILE
    _update_excel_file(list_final_result, out_dir)

    ### WRITE TO DEBUG IMAGES
    if debug:
        _write_debug_image(list_final_result, out_dir)

    # if ocr_output_dir is not None:
    #     tempfile.rmtree(ocr_output_dir)

if __name__ == '__main__':
    from ocr.robust.model import RobustOCR
    # from ocr.autoregressive.model import AutoregressiveOCR
    # test_textseg()
    LIST_INPUT_DATA = [
        {
            'PRED_DIR': '/Users/jason/Work/Cinnamon/evaluate_la_ocr/data/test_output_mixed_kv_layout/results',
            'OUT_DIR': '/Users/jason/Work/Cinnamon/evaluate_la_ocr/data/test_output_mixed_kv_layout/report_benchmark_ocr_finetuned_101',
            'GT_DIR': '/Users/jason/Work/Cinnamon/Datasets/SumitomoLife/full_test/label',
            'IMAGE_DIR': '/Users/jason/Work/Cinnamon/Datasets/SumitomoLife/full_test/input',
            'IS_RUN': True,
            'FILTER_VERTICAL': True,
            'FILTER_HORIZONTAL': False
        },
    ]

    # model = CannetOCR(weights_path=r'/mnt/ai_filestore/home/jason/laocrkv/LaOcr/models/ocr/CannetOCR-v2.6.0.pt', device="gpu")
    
    # model = JeffOCR(weights_path=r'/mnt/ai_filestore/home/jason/laocrkv/LaOcr/models/ocr/JeffOCR_Tokyomarine.pth')
    # model = RobustOCR(weights_path=r'/Users/jason/Work/Cinnamon/prj_flax_sumitomo_life_prod2_ai/weight/ocr/RobustOCR.pt')
    model = RobustOCR(weights_path=r'weights/Robust-SumitomoLife-General-Ratio-0.5-100_20230402_162046UTC_bestAC3_9468_Epoch19Fit1.pt')

    # model = RobustOCR(weights_path=r'/Users/jason/Work/Cinnamon/evaluate_la_ocr/Robust-SumitomoLife_20230326_161125UTC_bestAC3_9708_Epoch32Fit0.pt')
    # model = AutoregressiveOCR(weights_path=r'/Users/jason/Work/Cinnamon/lib-ocr/weight/autoregressive/par_general.pth')

    for info in LIST_INPUT_DATA:
        if info['IS_RUN']:
            print("Processing ", info['IMAGE_DIR'])
            makedir(info['OUT_DIR'])
            _evaluate_dir(model, info['GT_DIR'], info['PRED_DIR'], out_dir=info['OUT_DIR'], image_root=info['IMAGE_DIR'], debug=False, filter_vertical=info["FILTER_VERTICAL"], filter_horizontal=info["FILTER_HORIZONTAL"])
