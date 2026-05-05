import os
import re
import json

def _rename():
    logs_dir = "logs_3"
    pattern = re.compile(r"^(ep\d+)-loss[\d.]+-val_loss[\d.]+\.pth$")

    for filename in os.listdir(logs_dir):
        match = pattern.match(filename)
        if match:
            new_name = f"{match.group(1)}.pth"
            src = os.path.join(logs_dir, filename)
            dst = os.path.join(logs_dir, new_name)
            os.rename(src, dst)
            print(f"{filename} -> {new_name}")

def _rewrite_json_for_bbox():
    for i in range(200):
        json_path = f"/data/mml/data_debugging_data/collection_indicator_bbox_level/KITTI_8/SSD/predicted_bbox/epoch_{i}_predicted_bboxs.json"
        with open(json_path, "r") as f:
             _json = json.load(f)
        for imagename in list(_json.keys()):
            predicted_bboxs = _json[imagename]["predicted_bboxs"]
            for predicted_bbox in predicted_bboxs:
                bbox = predicted_bbox["bbox"]
                bbox[0],bbox[3] = bbox[3], bbox[0]
        save_path = f"/data/mml/data_debugging_data/collection_bbox_level/KITTI_8/SSD/predicted_bbox_new/epoch_{i}_predicted_bboxs.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(_json, f, indent=4, ensure_ascii=False)
    print("END")

def _rewrite_json_for_pid():
    for i in range(200):
        json_path = f"/data/mml/data_debugging_data/collection_bbox_level/KITTI_8/SSD/predicted_bbox/epoch_{i}_predicted_bboxs.json"
        with open(json_path, "r") as f:
             _json = json.load(f)
        p_id = 0
        for imagename in list(_json.keys()):
            predicted_bboxs = _json[imagename]["predicted_bboxs"]
            for predicted_bbox in predicted_bboxs:
                predicted_bbox["predicted_box_id"] = p_id
                p_id += 1
        save_path = f"/data/mml/data_debugging_data/collection_bbox_level/KITTI_8/SSD/predicted_bbox_new/epoch_{i}_predicted_bboxs.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(_json, f, indent=4, ensure_ascii=False)
    print("END")

if __name__ == "__main__":
    _rewrite_json_for_pid()


