import json
import codecs
import os 


label_dir = r'gt_data/val_v4/labels'
n_header = 0
n_noise = 0
for file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, file)
    with codecs.open(label_path, "r", "utf-8") as f:
        content = json.loads(f.read())
    
    regions = content["attributes"]["_via_img_metadata"]["regions"]
    for region in regions:
        region_attributes = region['region_attributes']
        if "header" in region_attributes["note"]:
            n_header += 1
        
        if "noise" in region_attributes["note"]:
            n_noise += 1
print(n_header)
print(n_noise)
    
