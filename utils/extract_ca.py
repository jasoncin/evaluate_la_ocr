import os
import json
import pandas as pd


def _makedir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def _process_json(gt_dict, basename, list_value):
    region_list = gt_dict['attributes']['_via_img_metadata']['regions']
    for i, region in enumerate(region_list):
        key_type = region['region_attributes']['key_type']
        formal_key = region['region_attributes']['formal_key']
        label = region['region_attributes']['label']
        if key_type == 'value':
            list_value.append([basename[0], formal_key, label])
    return list_value

def _build_excel_file(list_value, writer, code):
    labels = [
        'base_name', 'formal_key', 'label'
    ]
    df = pd.DataFrame.from_records(list_value, columns=labels)
    df.to_excel(writer, code)
    return writer


def extract_ca(gt_dir, writer, code):
    list_gt_files = []
    list_value = []
    for _root, _dirs, _files in os.walk(gt_dir):
        for f in _files:
            if f.endswith('.json'):
                filename = os.path.join(_root, f)
                list_gt_files.append(filename)

    for gt_file in list_gt_files:
        with open(gt_file, 'rt', encoding='utf-8') as f:
            gt_dict = json.load(f)
            basename = os.path.splitext(os.path.basename(gt_file))
            list_value = _process_json(gt_dict, basename, list_value)

    return _build_excel_file(list_value, writer, code)


if __name__ == '__main__':
    LIST_INPUT = [
        {
            'GT': 'C:\\Users\\lucas\\Documents\\Cinnamon\\Projects\\Prudential\\Input_data\\1569900164_Prudential_Validation_Test_190919\\labels',
            'CODE': 'Validation_190919'
        },
        {
            'GT': 'C:\\Users\\lucas\\Documents\\Cinnamon\\Projects\\Prudential\\Input_data\\1570510276_Prudential_Validation_Test_190927\\labels',
            'CODE': 'Validation_270919'
        },
        {
            'GT': 'C:\\Users\\lucas\\Documents\\Cinnamon\\Projects\\Prudential\\Input_data\\1570510324_Prudential_Private_test_1\\labels',
            'CODE': 'Private_1'
        },
        {
            'GT': 'C:\\Users\\lucas\\Documents\\Cinnamon\\Projects\\Prudential\\Input_data\\1570510306_Prudential_Private_test_2\\labels',
            'CODE': 'Private_2'
        }
    ]
    OUT_DIR = 'output_ca'
    _makedir(OUT_DIR)
    excel_file = os.path.join(OUT_DIR, 'ca.xlsx')
    writer = pd.ExcelWriter(excel_file)

    for input in LIST_INPUT:
        writer = extract_ca(input['GT'], writer, input['CODE'])

    # build excel file
    writer.save()


