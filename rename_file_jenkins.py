import os
from shutil import copyfile


# debug_data = r'/mnt/lustre/home/jason/projects/prj_tmc4/data/Test_Phase_6_LayoutLM_Tadashi/debugs'
# out_dir = r'/mnt/lustre/home/jason/projects/prj_tmc4/data/Test_Phase_6/ocr_output'

# for img_name in os.listdir(debug_data):
#     # json_name = [file_name.replace("jpg", "json").replace("png", "json") for file_name in os.listdir(os.path.join(debug_data, img_name)) if file_name.endswith("jpg")][0]

#     copyfile(os.path.join(os.path.join(debug_data, img_name), "ocr_output.json"),
#              os.path.join(out_dir, "{}.json".format(img_name)))


input_path = '/Users/jason/Work/Cinnamon/evaluate_la_ocr/data/test_output_mixed_kv_layout/results'
output_path = '/Users/jason/Work/Cinnamon/evaluate_la_ocr/data/test_output_mixed_kv_layout/ocr_output'

if not os.path.exists(output_path):
    os.mkdir(output_path)
    
for img_name in os.listdir(input_path):
    if '.DS_Store' in img_name:
        continue
    
    try:
        copyfile(os.path.join(os.path.join(input_path, img_name), "ocr_output.json"),
                os.path.join(output_path, "{}.json".format(img_name)))
    except Exception:
        copyfile(os.path.join(os.path.join(input_path, img_name), "ocr.json"),
                os.path.join(output_path, "{}.json".format(img_name)))
