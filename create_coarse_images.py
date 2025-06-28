import os
from PIL import Image, ImageDraw
import cv2  
import numpy as np
import random
random.seed(1024)
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse

def parse_img_position(img_name):
    parts = img_name.split('_')
    xmin = int(parts[-2])
    ymin = int(parts[-1].split('.')[0])
    obj_img = Image.open(img_name)
    obj_width, obj_height = obj_img.size
    xmax = xmin + obj_width
    ymax = ymin + obj_height
    return xmin, ymin, xmax, ymax

def determine_paste_position(xmin, ymin, xmax, ymax, width, height):
    is_inside = False
    if xmin > 0 and ymin > 0 and xmax < width and ymax < height:
        paste_x = random.randint(0, width - xmax)
        paste_y = random.randint(0, height - ymax)
        is_inside = True
    elif xmin == 0 and ymin > 0 and ymax < height:
        paste_x = 0
        paste_y = random.randint(0, height - ymax)
    elif xmin == 0 and ymin == 0:
        paste_x = 0
        paste_y = 0
    elif xmin == 0 and ymax == height:
        paste_x = 0
        paste_y = height - ymax
    elif xmax == width and ymin > 0 and ymax < height:
        paste_x = width - xmax
        paste_y = random.randint(0, height - ymax)
    elif xmax == width and ymin == 0:
        paste_x = width - xmax
        paste_y = 0
    elif xmax == width and ymax == height:
        paste_x = width - xmax
        paste_y = height - ymax
    elif xmin > 0 and xmax < width and ymin == 0:
        paste_x = random.randint(0, width - xmax)
        paste_y = 0
    elif xmin > 0 and xmax < width and ymax == height:
        paste_x = random.randint(0, width - xmax)
        paste_y = height - ymax
    else:
        raise ValueError(f"Unsupported coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

    return paste_x, paste_y, is_inside

def is_overlap(existing_boxes, new_box):
    for box in existing_boxes:
        if not (new_box[2] <= box[0] or new_box[0] >= box[2] or new_box[3] <= box[1] or new_box[1] >= box[3]):
            return True
    return False

def create_xml(xml_path, filename, width, height, object_elements):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for obj in object_elements:
        object_element = ET.SubElement(annotation, "object")
        ET.SubElement(object_element, "name").text = obj["name"]
        bndbox = ET.SubElement(object_element, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])
    tree = ET.ElementTree(annotation)
    with open(xml_path, "wb") as xml_file:
        tree.write(xml_file, encoding="utf-8", xml_declaration=True, method="xml")

def create_mask(img):
    img_array = np.array(img)
    mask = np.any(img_array[:, :, :3] != 0, axis=-1).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    return Image.fromarray(mask)

def proc_img_aug(img_pil, is_inside):
    scale_w = random.uniform(0.9, 1.1)
    scale_h = random.uniform(0.9, 1.1)
    new_width = int(img_pil.width * scale_w)
    new_height = int(img_pil.height * scale_h)
    img_pil_aug = img_pil.resize((new_width, new_height))
    if is_inside:
        angle = random.uniform(0, 360)
        img_pil_aug = img_pil_aug.rotate(angle, expand=True)
    return img_pil_aug

def scale_bounding_box(obj, image_width, image_height):
    xmin, ymin, xmax, ymax = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    scale_w = random.uniform(0.9, 1.1)
    scale_h = random.uniform(0.9, 1.1)
    new_width = (xmax - xmin) * scale_w
    new_height = (ymax - ymin) * scale_h
    new_xmin = int(center_x - new_width / 2)
    new_ymin = int(center_y - new_height / 2)
    new_xmax = int(center_x + new_width / 2)
    new_ymax = int(center_y + new_height / 2)
    new_xmin = max(0, new_xmin)
    new_ymin = max(0, new_ymin)
    new_xmax = min(image_width, new_xmax)
    new_ymax = min(image_height, new_ymax)
    return {"name": obj["name"], "xmin": new_xmin, "ymin": new_ymin, "xmax": new_xmax, "ymax": new_ymax}

def paste_objects_and_generate_xml(generated_img, cell_dirs, cell_names, cell_range_nums, save_dirs, img_index):
    img_width, img_height = generated_img.size
    existing_boxes = []
    object_elements = []
    for cell_name, range_num in zip(cell_names, cell_range_nums):
        cell_dir = os.path.join(cell_dirs, cell_name)
        chosen_cell_num = random.randint(range_num[0], range_num[1])
        check_overlap = True
        cells = os.listdir(cell_dir)
        chosen_cells = random.sample(cells, min(len(cells), chosen_cell_num))
        for chosen_cell in chosen_cells:
            chosen_cell_path = os.path.join(cell_dir, chosen_cell)
            chosen_cell_img = Image.open(chosen_cell_path)
            xmin, ymin, xmax, ymax = parse_img_position(chosen_cell_path)
            paste_x, paste_y, is_inside = determine_paste_position(xmin, ymin, xmax, ymax, img_width, img_height)
            chosen_cell_img = proc_img_aug(chosen_cell_img, is_inside)
            obj_width, obj_height = chosen_cell_img.size    
            new_box = (paste_x, paste_y, min(paste_x + obj_width, img_width), min(paste_y + obj_height, img_height))
            paste = False
            for _ in range(100):
                if not is_overlap(existing_boxes, new_box):
                    paste = True
                    existing_boxes.append(new_box)
                    break
                paste_x, paste_y, _ = determine_paste_position(xmin, ymin, xmax, ymax, img_width, img_height)
                new_box = (paste_x, paste_y, min(paste_x + obj_width, img_width), min(paste_y + obj_height, img_height))
            if paste:
                cell_mask_img = create_mask(chosen_cell_img)
                generated_img.paste(chosen_cell_img, (paste_x, paste_y), cell_mask_img)
                obj = {"name": cell_name, "xmin": new_box[0], "ymin": new_box[1], "xmax": new_box[2], "ymax": new_box[3]}
                obj = scale_bounding_box(obj, img_width, img_height) 
                object_elements.append(obj)
    generated_img_name = f"generated_{cell_names[0]}_{img_index}.jpg"
    generated_img_path = os.path.join(save_dirs['jpg'], generated_img_name)
    generated_img.save(generated_img_path)
    xml_path = os.path.join(save_dirs['xml'], f"generated_{cell_names[0]}_{img_index}.xml")
    create_xml(xml_path, generated_img_name, img_width, img_height, object_elements)

def generate_background(width, height):
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    return Image.fromarray(background)

def generate_multiple_imgs(num_imgs, width, height, cell_dirs, cell_names, cell_range_nums, save_dirs):
    print(f'processing {cell_names[0]}...')
    for img_index in tqdm(range(1, num_imgs + 1)):
        generated_img = generate_background(width, height)
        paste_objects_and_generate_xml(generated_img, cell_dirs, cell_names, cell_range_nums, save_dirs, img_index)

# ------------------------- argparse + main -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate augmented images and annotations in VOC format")
    parser.add_argument('--save_dir', type=str, required=True, help='Root directory for saving augmented images and annotations')
    parser.add_argument('--cell_dir', type=str, required=True, help='Segmented cells image folder')
    parser.add_argument('--width', type=int, default=512, help='Width of the generated image')
    parser.add_argument('--height', type=int, default=512, help='Height of the generated image')
    parser.add_argument('--phase', type=str, default='train', help='')
    parser.add_argument('--img_num', type=int, default=1000, help='The quantity of generated cells for each category')    
    return parser.parse_args()

def main():
    args = parse_args()
    generated_dict = {'HSIL': args.img_num, 'atrophy': args.img_num, 'SCC': args.img_num, 'bare_nucleus': args.img_num,
                          'trichomonad': args.img_num, 'LSIL': args.img_num, 'ASC_US': args.img_num, 'ASC_H': args.img_num}
    save_dirs = {
            'jpg': os.path.join(args.save_dir, '{}A'.format(args.phase)),
            'xml': os.path.join(args.save_dir, '{}A_Annotations'.format(args.phase))
        }

    os.makedirs(save_dirs['jpg'], exist_ok=True)
    os.makedirs(save_dirs['xml'], exist_ok=True)
    generated_dict = {k: v for k, v in generated_dict.items() if v > 0}

    for cell_name, num_imgs in generated_dict.items():
        generate_multiple_imgs(
            num_imgs=num_imgs,
            width=args.width,
            height=args.height,
            cell_dirs=args.cell_dir,
            cell_names=[cell_name, 'normal'],
            cell_range_nums=[(1, 2), (0, 5)],
            save_dirs=save_dirs
        )

if __name__ == '__main__':
    main()
