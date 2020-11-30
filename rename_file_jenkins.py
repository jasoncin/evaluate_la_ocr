import os
from shutil import copyfile

debug_data = r'pred_data/test_v5_can_jeff_tkomr/jenkins'
out_dir = r'pred_data/test_v5_can_jeff_tkomr/ocr_output'

for img_name in os.listdir(debug_data):
    # json_name = [file_name.replace("jpg", "json").replace("png", "json") for file_name in os.listdir(os.path.join(debug_data, img_name)) if file_name.endswith("jpg")][0]

    copyfile(os.path.join(os.path.join(debug_data, img_name), "1/kv_input.json"),
             os.path.join(out_dir, "{}.json".format(img_name)))