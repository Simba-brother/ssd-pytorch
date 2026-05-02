'''
收集每个epoch model在数据集上的预测结果
'''
import os

import time
import json
import torch
import numpy as np
from PIL import Image
from nets.ssd import SSD300
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image)
from utils.anchors import get_anchors
from utils.utils_bbox import BBoxUtility # bg class 对应处理了
from small_utils import format_timestamp
from pprint import pprint


def build_model(num_classes):
    num_classes = num_classes + 1 # bg class + 1在build模型时
    model = SSD300(num_classes, backbone)
    return model

class Detector(object):
    def __init__(self, 
                 model:SSD300,
                 num_classes:int,
                 input_shape:list=[300,300],
                 anchors_size:list=[30, 60, 111, 162, 213, 264, 315],
                 backbone:str="vgg",
                 nms_iou:float=0.45,
                 confidence:float=0.5
                 ):
         model = torch.nn.DataParallel(model)
         self.model = model.cuda()
         self.num_classes = num_classes + 1
         self.input_shape = input_shape
         self.bbox_util = BBoxUtility(self.num_classes)
         self.anchors_size = anchors_size
         self.backbone = backbone
         self.anchors = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size, self.backbone)).type(torch.FloatTensor)
         self.anchors = self.anchors.cuda()
         self.letterbox_image = False
         self.nms_iou = nms_iou
         self.confidence = confidence


    def detect_image(self,image:Image):
        #   计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        #   转为RGB
        image = cvtColor(image)
        #   resize
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #   添加上batch_size维度，图片预处理，归一化。
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            #   转化成torch的形式
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            images = images.cuda()
            #   将图像输入网络当中进行预测!
            outputs = self.model(images)
            #   将预测结果进行解码
            results = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                nms_iou = self.nms_iou, confidence = self.confidence)
            #   如果没有检测到物体，则返回 None
            if len(results[0]) <= 0:
                return None

            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]
            return results[0]
        
    def detect_images(self,images:list[Image.Image]):
        res_list = []
        for image in images:
            # res.shape = (objnums,6), 6: [top, left, bottom, right, cls_i, conf]
            res = self.detect_image(image)
            res_list.append(res)
        return res_list

def get_dataset(dataset_dir)->list[Image.Image]:
    images = []
    image_paths = []
    image_ids = open(os.path.join(dataset_dir, "ImageSets/Main/trainval.txt")).read().strip().split()
    for image_id in image_ids:
        image_path = os.path.join(dataset_dir, "JPEGImages/"+image_id+".png")
        image = Image.open(image_path)
        images.append(image)
        image_paths.append(image_path)
    return images,image_paths

def result_to_dict(image_paths:list[str], result_list:list)->dict:
    result_dict = {}
    predicted_bbox_id = 0
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image_name = os.path.basename(image_path)
        result = result_list[i]
        if result is None:
            continue
        result_dict[image_name] = {
            "predicted_bboxs":[]
        }
        for obj_i in range(result.shape[0]):
            ymin,ymax,xmin,xmax = result[obj_i][0:4].tolist()
            predicted_cls = result[obj_i][4]
            conf = result[obj_i][5]
            predicted_bbox = {
                "predicted_box_id":predicted_bbox_id,
                "img_name":image_name,
                "predicted_cls":int(predicted_cls),
                "conf":conf.item(),
                "bbox":[xmin,ymin,xmax,ymax]
            }
            result_dict[image_name]["predicted_bboxs"].append(predicted_bbox)
    return result_dict

def main():
    # 获得模型
    print("获类别信息...")
    class_names, num_classes  = get_classes(classes_path) # 不含bg class
    print("获得模型结构...")
    model = build_model(num_classes)
    # 获得数据集
    print("获得数据集...")
    images, image_paths = get_dataset(dataset_dir)
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"===Epoch:{epoch+1}/{epochs}===")
         # 装载模型权重
        print("load weight...")
        epoch_str = str(epoch+1).zfill(3)
        weight_path = os.path.join(f"/home/mml/workspace/ssd-pytorch/logs_3/ep{epoch_str}.pth")
        print(f"weight_path:{weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model = model.eval()
        # 开始推理
        print("开始推理...")
        detector = Detector(model=model,num_classes=num_classes)
        result_list = detector.detect_images(images)
        result_dict = result_to_dict(image_paths,result_list)
        
        # 整理结果到json
        print("开始保存结果...")
        json_file_name = f"epoch_{epoch}_predicted_bboxs.json"
        save_json_path = os.path.join(save_dir,json_file_name)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=4)
        print(f"数据保存在:{save_json_path}")
        epoch_end_time = time.time()
        epoch_cost_time = epoch_end_time - epoch_start_time
        print(f"epoch耗时:{format_timestamp(epoch_cost_time)}")


if __name__ == '__main__':
    pid = os.getpid()
    exp_data_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8"
    model_name = "SSD"
    backbone = "vgg"
    epochs = 200
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    dataset_dir = f"datasets/{dataset_name}"
    classes_path = f"{dataset_dir}/classes.txt"
    save_dir = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level", dataset_name, model_name, "predicted_bbox")
    os.makedirs(save_dir,exist_ok=True)

    # 打印实验基本信息
    exp_base_info = {
        "pid":pid,
        "dataset_name":dataset_name,
        "model_name":model_name,
        "backbone_name":backbone,
        "epoch":epochs,
        "gpu_id":0,
        "save_dir":save_dir
    }
    pprint(exp_base_info,sort_dicts=False)
    start_time = time.time()
    print(f"实验开始时间:{format_timestamp(start_time)}")
    main()
    end_time = time.time()
    print(f"实验结束时间:{format_timestamp(end_time)}")
    cost_time = end_time - start_time()
    print(f"实验消耗时间:{format_timestamp(cost_time)}")