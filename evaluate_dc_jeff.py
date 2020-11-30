from layout.dc.model import DcLayout
from layout.jeff.model import JeffLayout
from layout.evaluation import evaluate
import codecs
import time
import json
import cv2
import numpy as np

def draw_boxes(img, textline_data):
    for  textline in textline_data:
        textline = textline['location']
        pts = np.array(textline, np.int32)
        pts = pts.reshape((-1,1,2))

        cv2.polylines(img,[pts],True,(255,0,0), 	thickness = 4, 	lineType = cv2.LINE_4)
        # cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), thickness=5)
    return img

img_dir = r''
out_dir = r'visualize'

jeff_model = JeffLayout(weights_path='/mnt/ai_filestore/home/jason/laocrkv/LaOcr/models/layout/Jeff-Stamp-Sompo-Focus_0.91_0.92.pth',
config='/mnt/ai_filestore/home/jason/lib-layout/layout/jeff/config/general_config.yaml', device="0")
dc_model = DcLayout(weights_path='/mnt/ai_filestore/home/jason/projects/flax_sompo_holdings_poc/model/layout/DcLayout_Final_Ref_v0')

jeff_st_time = time.time()
jeff_predictions = jeff_model.process("/mnt/ai_filestore/home/jason/trash/trash2/DCNet/assets/datasets/sompo_test/images/921.jpg")
jeff_process_time = time.time() - jeff_st_time

img = cv2.imread("/mnt/ai_filestore/home/jason/trash/trash2/DCNet/assets/datasets/sompo_test/images/921.jpg")
draw_image = draw_boxes(img, jeff_predictions)
cv2.imwrite("{}/{}".format(out_dir, "921_jeff.jpg"), img)

dc_st_time = time.time()
dc_predictions = dc_model.process("/mnt/ai_filestore/home/jason/trash/trash2/DCNet/assets/datasets/sompo_test/images/921.jpg")
dc_process_time = time.time() - dc_st_time

img = cv2.imread("/mnt/ai_filestore/home/jason/trash/trash2/DCNet/assets/datasets/sompo_test/images/921.jpg")
draw_image = draw_boxes(img, dc_predictions)
cv2.imwrite("{}/{}".format(out_dir, "921_dc.jpg"), img)

# with codecs.open("dc_predictions.json", "w", "utf-8") as f:
#     f.write(json.dumps(dc_predictions, indent=4, ensure_ascii=False))

jeff_results_dict = evaluate(predictions=[jeff_predictions], targets=["/mnt/ai_filestore/home/jason/trash/trash2/DCNet/assets/datasets/sompo_test/labels/921.json"])
dc_results_dict = evaluate(predictions=[dc_predictions], targets=["/mnt/ai_filestore/home/jason/trash/trash2/DCNet/assets/datasets/sompo_test/labels/921.json"])

print("Jeff result", jeff_results_dict)
print("In {}s.".format(jeff_process_time))

print("\nDc result", dc_results_dict)
print("In {}s.".format(dc_process_time))
