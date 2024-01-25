# Circle Detection using YOLOv8

This project demonstrates circle detection using YOLO (You Only Look Once) with the Ultralytics library for the assignment under the company BigVision AI.
It's executed using Jupyter Notebook in Python Language.
## Purpose:
To assess the candidate's capabilities in custom dataset creation, model training, and object detection specifically tailored for geometric shapes. This assignment focuses on detecting circles in images, a common task in various applications like quality control or medical image analysis.

## Details:
- Task: Develop a YOLO object detection system to identify and locate circles.
- Dataset Creation: We've created a synthetic dataset of images containing circles with varying sizes, colors, and backgrounds. The dataset has also include negative examples (images without circles)


## Dependencies

Make sure you have the following dependencies installed:

- [Ultralytics](https://github.com/ultralytics/yolov5): `!pip install ultralytics` Make sure Ultralytics is installed.
- [Pandas](https://pandas.pydata.org/): `import pandas as pd`
- [NumPy](https://numpy.org/): `import numpy as np`
- [OpenCV](https://opencv.org/): `import cv2`
- [PyTorch](https://pytorch.org/): `import torch`
- [Pillow](https://pillow.readthedocs.io/): `from PIL import Image`
- [Scikit-Image](https://scikit-image.org/): `from skimage import draw`
- [Random](https://docs.python.org/3/library/random.html): `import random`
- [Pathlib](https://docs.python.org/3/library/pathlib.html): `from pathlib import Path`
- [Rasterio](https://rasterio.readthedocs.io/): `import rasterio`
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5): `from ultralytics import YOLO`

## Data Generation

The code includes functions to generate positive and negative samples of images with circles. Images are created in the `train`, `val`, and `test` directories.
It includes the use of YOLO Model having following parameers
- image size = 120 x 120 (size of image for training will be 120 pixels)
- epochs = 7 (number of training epochs.)
- batch = 8 (batch size for training)
- model file used = yolov8s.pt 
- device = cpu (Sets the training device to CPU. We can use GPU, by calling "cuda" for faster training.)
- data = "/content/data.yaml"  (contains configuration details about the dataset, such as the paths to training images, labels, classes, etc.)

## Maintain yaml.file
train: /content/trial_data/train/images

val: /content/trial_data/val/images

test: /content/trial_data/test/images

names: ['circle']
## Generating Samples

```python
#We used the Python Imaging Library (PIL) for image manipulation and NumPy for array operations.
def create_positive_sample(path, img_size, min_radius): #to generate image having circle
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.zeros((img_size, img_size)).astype(np.uint8)
    #Randomly selects a center (center_x, center_y) for the circle within the image boundaries, considering the min_radius.
    center_x = random.randint(min_radius, (img_size - min_radius))
    center_y = random.randint(min_radius, (img_size - min_radius))
    max_radius = min(center_x, center_y, img_size - center_x, img_size - center_y)
    radius = random.randint(min_radius, max_radius)

    row_indxs, column_idxs = draw.ellipse(center_x, center_y, radius, radius, shape=arr.shape)

    arr[row_indxs, column_idxs] = 255

    im = Image.fromarray(arr)
    im.save(path)

def create_negative_sample(path, img_size): #to generate image not containing circle
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.zeros((img_size, img_size)).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(path)

```
## Creating Images for Training,Validation, Testing Phase
The below mentioned code defines a function create_images that generates a dataset of images with circles (positive samples) and without circles (negative samples) in a proportionate manner. The dataset is divided into training, validation, and test sets, and the generated images are saved in a specified directory structure.

```python
def create_images(data_root_path, train_num, val_num, test_num, img_size=640, min_radius=10): #generate the images through above function in proportionate manner
    data_root_path = Path(data_root_path)

    for i in range(train_num):
        create_positive_sample(data_root_path / 'train' / 'images' / f'1_img_{i}.png', img_size, min_radius)

    for i in range(val_num):
        create_positive_sample(data_root_path / 'val' / 'images' / f'1_img_{i}.png', img_size, min_radius)

    for i in range(test_num):
        create_positive_sample(data_root_path / 'test' / 'images' / f'1_img_{i}.png', img_size, min_radius)

    for i in range(int(0.1 * train_num)):
      create_negative_sample(data_root_path / 'train' / 'images' / f'0_img_{i}.png', img_size) #number of negative sample will be 10% of positive sample

    for i in range(int(0.1 * val_num)):
      create_negative_sample(data_root_path / 'val' / 'images' / f'0_img_{i}.png', img_size)

    for i in range(int(0.1 * test_num)):
      create_negative_sample(data_root_path / 'test' / 'images' / f'0_img_{i}.png', img_size)


create_images('trial_data', train_num=700, val_num=200, test_num=100, img_size=120, min_radius=10)
```
## Creating Label for each Image

The below mentioned code provided generates label files for images that contain shapes (circles) and saves the coordinates of the shapes in a specific format. It iterates through the images in the 'train', 'val', and 'test' directories, checks if the image contains shapes, and if so, creates a label file with the corresponding coordinates.
```python
def create_label(image_path, label_path):
    arr = np.asarray(Image.open(image_path))
    cords_list = list(features.shapes(arr, mask=(arr > 0)))
    # Check if the list is not empty before accessing its elements
    if cords_list:
        cords = cords_list[0][0]['coordinates'][0]
        label_line = '0 ' + ' '.join([f'{int(cord[0])/arr.shape[0]} {int(cord[1])/arr.shape[1]}' for cord in cords])
        # Create the parent directory for the label_path
        label_path.parent.mkdir(parents=True, exist_ok=True)
        # Write the label_line to the file
        with label_path.open('w') as f:
            f.write(label_line)
        return label_line
    else:
        print("No shapes found.", image_path) #it ensures that the given image doesn't contain any shape
        return None
for images_dir_path in [Path(f'trial_data/{x}/images') for x in ['train', 'val', 'test']]: #for each and every file, we will verify whether image is labeled or not.
    for img_path in images_dir_path.iterdir():
        label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
        label_line = create_label(img_path, label_path)
        if label_line:
            print(f"Label file saved for {img_path}")
        else:
            print(f"Label file not saved for {img_path}")
```


## Show Result
To see the validation Prediction:-
```python
from IPython.display import Image as show_image
show_image(filename="/content/runs/detect/train/val_batch2_pred.jpg")
```
To show the output:-
```python
show_image(filename="/content/runs/detect/train/val_batch0_labels.jpg")
```

Model results:-
```python
show_image(filename="/content/runs/detect/train/results.png")
```
