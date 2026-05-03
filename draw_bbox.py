
import cv2
import matplotlib.pyplot as plt
from utils.utils import get_classes

def vis_correct_bbox(img_path:str, bbox_list:list, save_path:str):
    # 读取图像（注意 cv2 是 BGR，需要转 RGB）
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img)
    ax.set_title('Correct Annotation', fontsize=14, color='green')
    ax.axis('off')
    for bbox in bbox_list:
        x1,y1,x2,y2 = bbox["bbox"]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        w = x2-x1
        h = y2-y1
        class_name = class_names[bbox["predicted_cls"]]
         # 绘制边界框
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor='none')
        ax.add_patch(rect)
        # 添加类别标签
        ax.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="green",
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"image save path: {save_path}")

def main():
    image_path = f"datasets/{dataset_name}/JPEGImages/{image_name}"
    
    bbox_list = [
            {
                "predicted_box_id": 0,
                "img_name": "007271.png",
                "predicted_cls": 0,
                "conf": 0.6230435371398926,
                "bbox": [
                    0.0,
                    178.2677764892578,
                    176.93856811523438,
                    249.85887145996094
                ]
            },
            {
                "predicted_box_id": 0,
                "img_name": "007271.png",
                "predicted_cls": 0,
                "conf": 0.6217337250709534,
                "bbox": [
                    802.9204711914062,
                    169.7057342529297,
                    1128.6856689453125,
                    274.89556884765625
                    
                ]
            }
        ]
    vis_correct_bbox(image_path, bbox_list, save_path)

if __name__ == "__main__":
    dataset_name = "KITTI_8"
    image_name = "007271.png"
    classes_path = f"datasets/{dataset_name}/classes.txt"
    class_names, _ = get_classes(classes_path)
    save_path = "img/demo.png"
    main()
