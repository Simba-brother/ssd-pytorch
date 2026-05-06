import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

stage = 2 # 0,1,2
dataset_name = "VisDrone_clean" # KITTI_8|KITTI_8_clean|VisDrone_clean
img_ext = "jpg"
classes_path        = f'datasets/{dataset_name}/classes.txt'
trainval_percent    = 0.9 # train:val

dataset_dir  = f'datasets/{dataset_name}'

dataset_splitsets  = ["train","val","test"]
classes, _      = get_classes(classes_path)

photo_nums  = np.zeros(len(dataset_splitsets))
nums        = np.zeros(len(classes)) # 统计各个cls下的img数量

def convert_annotation(image_id, list_file):
    # 打开一张图像的xml文件
    in_file = open(os.path.join(dataset_dir, f'Annotations/{image_id}.xml'),encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    # 遍历这张图像下所有的object
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            # 如果这个object的class name不在我们的classes.txt中，则跳过这个object
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        # 与image abs path 之间空一个空格
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        # 统计每个class_idx的obj nums
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
        
if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(dataset_dir):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if stage == 0 or stage == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(dataset_dir, 'Annotations')
        saveBasePath    = os.path.join(dataset_dir, 'ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        train_val_num = num # 训练集+验证集总数据量
        train_num      = int(train_val_num*trainval_percent)  
        train   = random.sample(list,train_num)  
        
        print("train and val size",train_val_num)
        print("train size",train_num)

        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  

        ftrain.close()  
        fval.close()  
        
        print("Generate txt in ImageSets done.")
    
    if stage == 0 or stage == 2:
        print("Generate train.txt and val.txt for train.")
        tvt_index = 0
        # 遍历 tvt:[train,val,test]
        for tvt in dataset_splitsets:
            image_ids = open(os.path.join(dataset_dir, f'ImageSets/Main/{tvt}.txt'), encoding='utf-8').read().strip().split()
            list_file = open(f'datasets/{dataset_name}/{tvt}.txt', 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write(f'{os.path.abspath(dataset_dir)}/JPEGImages/{image_id}.{img_ext}')
                convert_annotation(image_id, list_file)
                list_file.write('\n') # 每张image换一行
            photo_nums[tvt_index] = len(image_ids)
            tvt_index += 1
            list_file.close()
        print("Generate train.txt and val.txt for train done.")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")
