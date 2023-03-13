import os
from shutil import copyfile


# debug_data = r'/mnt/lustre/home/jason/projects/prj_tmc4/data/Test_Phase_6_LayoutLM_Tadashi/debugs'
# out_dir = r'/mnt/lustre/home/jason/projects/prj_tmc4/data/Test_Phase_6/ocr_output'

# for img_name in os.listdir(debug_data):
#     # json_name = [file_name.replace("jpg", "json").replace("png", "json") for file_name in os.listdir(os.path.join(debug_data, img_name)) if file_name.endswith("jpg")][0]

#     copyfile(os.path.join(os.path.join(debug_data, img_name), "ocr_output.json"),
#              os.path.join(out_dir, "{}.json".format(img_name)))

root_dir = '/mnt/lustre/data/flax/hogwarts/scm_out/'
for prj_name in os.listdir(root_dir):
    input_path = os.path.join(root_dir, prj_name, 'results')
    output_path = os.path.join('/mnt/lustre/data/flax/hogwarts/scm_data', prj_name, 'ocr_output')
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    for img_name in os.listdir(input_path):
        try:
            copyfile(os.path.join(os.path.join(input_path, img_name), "ocr_output.json"),
                    os.path.join(output_path, "{}.json".format(img_name)))
        except Exception:
            copyfile(os.path.join(os.path.join(input_path, img_name), "ocr.json"),
                    os.path.join(output_path, "{}.json".format(img_name)))
