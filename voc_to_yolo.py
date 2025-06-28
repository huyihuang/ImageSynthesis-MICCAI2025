import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
from tqdm import tqdm

def convert_voc_to_yolo(voc_annotation_dir, image_dir, yolo_annotation_dir, yolo_image_dir, class_names, classes_to_convert):
    """
    将 VOC 标注格式转换为 YOLO 格式

    voc_annotation_dir: VOC 的标注目录
    image_dir: 对应的图像目录
    yolo_annotation_dir: 保存 .txt 的目标目录
    yolo_image_dir: 保存图像的目标目录
    class_names: 所有类别名列表（顺序即类别编号）
    classes_to_convert: 只转换这些类名
    """
    xml_files = [f for f in os.listdir(voc_annotation_dir) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files, desc=f"Converting {os.path.basename(yolo_annotation_dir)}"):
        xml_path = os.path.join(voc_annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_name = root.find('filename').text
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue

        img = Image.open(image_path)
        width, height = img.size

        # YOLO 标签保存路径
        yolo_txt_path = os.path.join(yolo_annotation_dir, xml_file.replace('.xml', '.txt'))
        os.makedirs(os.path.dirname(yolo_txt_path), exist_ok=True)

        with open(yolo_txt_path, 'w') as f:
            for obj in root.iter('object'):
                class_name = obj.find('name').text
                if class_name not in classes_to_convert:
                    continue

                class_id = class_names.index(class_name)

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                center_x = (xmin + xmax) / 2 / width
                center_y = (ymin + ymax) / 2 / height
                obj_width = (xmax - xmin) / width
                obj_height = (ymax - ymin) / height

                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {obj_width:.6f} {obj_height:.6f}\n")

        # 同时复制图像到目标目录
        shutil.copy(image_path, os.path.join(yolo_image_dir, image_name))

def parse_args():
    parser = argparse.ArgumentParser(description="将VOC格式数据转换为YOLO格式")
    parser.add_argument('--voc_root', type=str, required=True, help='VOC数据集根目录（包含Annotations和JPEGImages）')
    parser.add_argument('--output_dir', type=str, required=True, help='YOLO格式数据集保存目录')
    return parser.parse_args()

def main():
    args = parse_args()

    voc_root = args.voc_root
    output_dir = args.output_dir

    # 全部类别
    class_names = ['HSIL', 'atrophy', 'SCC', 'bare_nucleus',
                   'trichomonad', 'LSIL', 'ASC_US', 'ASC_H',
                   'inflammation', 'normal']

    # 只转换以下 8 类（其余跳过）
    classes_to_convert = {'HSIL', 'atrophy', 'SCC', 'bare_nucleus',
                          'trichomonad', 'LSIL', 'ASC_US', 'ASC_H'}

    for split in ['train', 'val', 'test']:
        voc_ann_dir = os.path.join(voc_root, 'Annotations', split)
        voc_img_dir = os.path.join(voc_root, 'JPEGImages', split)
        yolo_lbl_dir = os.path.join(output_dir, 'labels', split)
        yolo_img_dir = os.path.join(output_dir, 'images', split)

        os.makedirs(yolo_lbl_dir, exist_ok=True)
        os.makedirs(yolo_img_dir, exist_ok=True)

        convert_voc_to_yolo(
            voc_annotation_dir=voc_ann_dir,
            image_dir=voc_img_dir,
            yolo_annotation_dir=yolo_lbl_dir,
            yolo_image_dir=yolo_img_dir,
            class_names=class_names,
            classes_to_convert=classes_to_convert
        )

if __name__ == '__main__':
    main()
