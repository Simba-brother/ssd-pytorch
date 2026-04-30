'''
收集每个epoch model在数据集上的预测结果
'''
import torch
from nets.ssd import SSD300
from utils.utils import get_classes


def load_model():
    class_names, num_classes  = get_classes(classes_path)
    model = SSD300(num_classes, backbone)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()
    print('{} model, anchors, and classes loaded.'.format(model_path))
    pass



def main():
    # 获得模型
    pass


if __name__ == '__main__':
    dataset_name = "KITTI_8"
    classes_path = f"datasets/{dataset_name}/classes.txt"
    backbone = "vgg"
    device = torch.device('cuda:0')
    model_path = "logs_2/best_epoch_weights.pth"

    main()