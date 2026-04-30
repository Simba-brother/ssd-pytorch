'''
基于文件系统制作数据集
'''

import os

def main():
    test_dir = "/data/mml/data_debugging_data/datasets/KITTI_8-voc/test"
    test_files = [
        os.path.splitext(f)[0]
        for f in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, f))]

    with open("datasets/KITTI_8/ImageSets/Main/test.txt", "w", encoding="utf-8") as f:
        for fname in test_files:
            f.write(fname + "\n")

    print("build finished")


if __name__ == "__main__":
    main()