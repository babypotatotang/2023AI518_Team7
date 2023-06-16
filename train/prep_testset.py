import os
import json
from json2txttree import json2txttree
from datasets import load_from_disk 

ds = load_from_disk('/home1/s20225114/GM_Melon/Melon_test')
out_image_path = './output/gt'

if not os.path.exists(out_image_path):
    os.mkdir(out_image_path)

gt_dict = {}

for index, data in enumerate(ds):
    gt_dict[f"{index:04d}.png"] = data['text']
    data['image'].save(f"/home1/s20225114/GM_Melon/Melon_ai518/output/gt/{index:04d}.png")

with open('./gt_captions.json', 'w') as f:
    json.dump(gt_dict, f)