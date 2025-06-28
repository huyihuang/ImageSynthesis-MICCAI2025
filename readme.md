# 📄 Controllable Image Synthesis for Cervical Cell Detection

> MICCAI 2025 Paper: *Controllable Image Synthesis Workflow with Adaptive Cell Segmentation and Style Transfer for Cervical Cell Detection*


## 🚀 How to Run

### 1. Setup Environment

```bash
conda create -n tct-syn python=3.10
conda activate tct-syn
pip install -r requirements.txt
```

### 2. Segment Cells 

```bash
python segmentation/run_cellpose.py --input /path/to/JPEGImages --xml /path/to/Annotations --output /path/to/save/cells --gpu --device 0
```

### 3. Generate Coarse Images(Batch generation of synthetic cell images and corresponding VOC format annotation files)

```bash
python synthesis/create_coarse_images.py --save_dir /path/to/save_dir --cell_dir /path/to/cells/ 
```

### 4. Style Transfer 
For the generated **coarse images**, we apply style transfer using **Cut** and **CycleGAN**.  
Model training and inference follow the official implementation:  
👉 [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

👉 [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation)
### 5.VOC to YOLO(Convert a VOC-format dataset to YOLO format, including images and annotation files.)
**The VOC dataset is expected to have the following directory structure:**
```
VOC_ROOT/
├── Annotations/ # XML annotation files (VOC format)
│   ├── train/ 
│   ├── val/ 
│   └── test/ 
└── JPEGImages/ # Corresponding images
    ├── train/ 
    ├── val/ 
    └── test/ 
```
```bash
python voc_to_yolo.py --voc_root /path/to/voc_dataset --output_dir /path/to/yolo_dataset
```
**Running this script will generate a YOLO-format dataset with the following directory structure:**
```
OUTPUT_DIR/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── images/
    ├── train/
    ├── val/
    └── test/
```

### 6. Train Detector

```bash
python detection/train_yolov11.py --data /path/to/train.yaml --weights /path/to/yolo11n.pt
```

### 7. Predict Detector

```bash
python detection/test_yolov11.py --data /path/to/train.yaml --weights /path/to/best.pt
```
---

## 📈 Results

| Model         | mAP50     | mAP50–95  |
| ------------- | --------- | --------- |
| Baseline      | 0.433     | 0.326     |
| +Coarse       | 0.463     | 0.344     |
| +Refined(CUT) | **0.481** | **0.356** |

See full results in `results/`

---


## 📚 Citation

Please cite our paper if this work is helpful to your research.


---

## 💬 Contact

If you have any questions, please contact Yihuang Hu (huyihuang@stu.xmu.edu.cn) and Qi Chen(qchen@stu.xmu.edu.cn)

---


