import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import TRAIN_DIR, IMAGE_SIZE
from sklearn.preprocessing import LabelEncoder
import sys

class Preprocessor:
    def __init__(
            self, 
            root_dir=TRAIN_DIR, 
            input_bbox_format='OBB', 
            target_bbox_format='PASCAL_VOC', 
            ignore_invalids=True
            ):
        
        self.root_dir = root_dir
        self.input_bbox_format = input_bbox_format
        self.target_bbox_format = target_bbox_format
        self.ignore_invalids = ignore_invalids

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.images = []

        # Store bboxes for each image (default target format: 'pascal_voc')
        # bboxes with target format 'obb' are stored as [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in clockwise order
        self.bboxes = []

        # Store labels as strings
        self.labels = []

        self.load_data()

    def load_data(
            self
            ):
        annot_file_names = [f for f in os.listdir(self.root_dir + '/annotations')]
        img_file_names = [f for f in os.listdir(self.root_dir + '/images')]

        for idx, (annot_file_name, img_file_name) in enumerate(zip(annot_file_names, img_file_names)):
            img_path = self.root_dir + '/images/' + img_file_name
            annot_path = self.root_dir + '/annotations/' + annot_file_name

            img = Image.open(img_path)

            if img.mode != "RGB":
                img = img.convert("RGB")

            org_width, org_height = img.size

            self.images.append(img)

            with open(annot_path, 'r') as f:
                lines = f.readlines()
                bboxes = []
                labels = []

                if self.input_bbox_format == 'OBB':
                    for line in lines:
                        invalid_flag = False
                        bbox = []
                        bbox_mod = []

                        line_data = line.split(' ')
                        for data in line_data[:-2]:
                            bbox.append(float(data))

                        if self.target_bbox_format == 'OBB':
                            for i in range(0, len(bbox)-1, 2):
                                if (bbox[i] < 0 or bbox[i] > org_width or bbox[i+1] < 0 or bbox[i+1] > org_height):
                                    if self.ignore_invalids:
                                        invalid_flag = True
                                        break

                                    else:
                                        raise ValueError(
                                            f'Invalid Bounding Box when converting to OBB\n'
                                            f'BBox Coord: ({bbox[i]}, {bbox[i+1]})\n'
                                        )
                                
                                bbox_mod.append([bbox[i], bbox[i+1]])

                            xmin = ymin = sys.maxsize
                            xmax = ymax = -sys.maxsize - 1

                            for bb in bbox_mod:
                                xmin = min(xmin, bb[0])
                                xmax = max(xmax, bb[0])
                                ymin = min(ymin, bb[1])
                                ymax = max(ymax, bb[1])

                            if xmax <= xmin or ymax <= ymin:
                                if self.ignore_invalids:
                                    invalid_flag = True
                                    break

                                else:
                                    raise ValueError(
                                        f'Invalid Bounding Box when converting to PASCAL VOC\n'
                                        f'BBox Coord: [{xmin}, {ymin}, {xmax}, {ymax}]\n'
                                    )

                            if invalid_flag:
                                continue

                        elif self.target_bbox_format == 'PASCAL_VOC':
                            xmin = ymin = sys.maxsize
                            xmax = ymax = -sys.maxsize - 1

                            for i in range(0, len(bbox)-1, 2):
                                xmin = min(xmin, bbox[i])
                                xmax = max(xmax, bbox[i])
                                ymin = min(ymin, bbox[i+1])
                                ymax = max(ymax, bbox[i+1])
                            
                                if (bbox[i] < 0 or bbox[i] > org_width or bbox[i+1] < 0 or bbox[i+1] > org_height):
                                    if self.ignore_invalids:
                                        invalid_flag = True
                                        break

                                    else:
                                        raise ValueError(
                                            f'Invalid Bounding Box when converting to PASCAL VOC\n'
                                            f'BBox Coord: ({bbox[i]}, {bbox[i+1]})\n'
                                            f'Image Dims: ({org_width}, {org_height})'
                                        )
                                    
                            if xmax <= xmin or ymax <= ymin:
                                if self.ignore_invalids:
                                    invalid_flag = True

                                else:
                                    raise ValueError(
                                        f'Invalid Bounding Box when converting to PASCAL VOC\n'
                                        f'BBox Coord: [{xmin}, {ymin}, {xmax}, {ymax}]\n'
                                    )
                                
                            bbox_mod = [xmin, ymin, xmax, ymax]
                                
                            if invalid_flag:
                                continue
                        
                        bboxes.append(bbox_mod)

                        labels.append(line_data[-2])

                    self.bboxes.append(bboxes)
                    self.labels.append(labels)

                if self.input_bbox_format != 'OBB':
                    raise ValueError(
                        f'No implementation to handle direct bbox format {self.input_bbox_format}'
                    )
                

    def resize_images(
            self, 
            target_width=IMAGE_SIZE, 
            target_height=IMAGE_SIZE
        ):

        for idx, image in enumerate(self.images):
            width, height = image.size
            self.images[idx] = image.resize((target_width, target_height))

            scale_x = target_width / width
            scale_y = target_height / height
        
            if self.target_bbox_format == 'OBB':
                resized_bboxes = []
                for bbox in self.bboxes[idx]:
                    resized_bbox = [[vertex[0] * scale_x, vertex[1] * scale_y] for vertex in bbox]
                    resized_bboxes.append(resized_bbox)

                self.bboxes[idx] = resized_bboxes

            elif self.target_bbox_format == 'PASCAL_VOC':
                resized_bboxes = []
                for bbox in self.bboxes[idx]:
                    xmin, ymin, xmax, ymax = bbox
                    resized_bbox = [xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y]
                    resized_bboxes.append(resized_bbox)
                
                self.bboxes[idx] = resized_bboxes

          
    def visualize_random_image(
            self
        ):

        idx = random.randint(0, len(self.images)-1)
        random_image = self.images[idx]
        image_bboxes = self.bboxes[idx]
        image_labels = self.labels[idx]

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(random_image)

        print(image_bboxes)
        print(image_labels)

        for bbox, label in zip(image_bboxes, image_labels):
            corners = bbox

            if self.target_bbox_format == 'PASCAL_VOC':
                xmin, ymin, xmax, ymax = bbox

                corners = [
                [xmin, ymin],  
                [xmax, ymin],  
                [xmax, ymax],  
                [xmin, ymax],  
                ]

            poly = patches.Polygon(
                corners, 
                closed=True, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            
            ax.add_patch(poly)

            ax.text(
                xmin, 
                ymin - 5, 
                str(label), 
                color='red', 
                fontsize=12, 
            )

        ax.axis('off')
        plt.show()
    
    def transform_images(
            self, 
            transforms
        ):

        for idx, image in enumerate(self.images):
            image_array = np.array(image)

            bboxes = self.bboxes[idx]
            labels = self.labels[idx]

            # Convert to 'pascal voc' bbox format if not already
            if self.target_bbox_format == "OBB":
                for ix, bbox in enumerate(bboxes):
                    xmin = min(v[0] for v in bbox)
                    ymin = min(v[1] for v in bbox)
                    xmax = max(v[0] for v in bbox)
                    ymax = max(v[1] for v in bbox)
                    
                    bbox_mod = [xmin, ymin, xmax, ymax]
                    
                    bboxes[ix] = bbox_mod

            bboxes_array = np.array(bboxes)

            data_transformed = transforms(
                image=image_array,
                bboxes=bboxes_array,
                labels=labels
            )

            image_tr = np.array(data_transformed['image']).transpose(1, 2, 0)
            bboxes_tr = data_transformed['bboxes'].tolist()

            self.images[idx] = Image.fromarray(image_tr)

            if self.target_bbox_format == "OBB":
                for ix, bbox_tr in enumerate(bboxes_tr):

                    xmin, ymin, xmax, ymax = bbox_tr

                    top_left = [xmin, ymin]
                    top_right = [xmax, ymin]
                    bottom_right = [xmax, ymax]
                    bottom_left = [xmin, ymax]

                    bboxes_tr[ix] = [top_left, top_right, bottom_right, bottom_left]

            self.bboxes[idx] = bboxes_tr


def get_train_transforms():
    return A.Compose([
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        A.RandomGamma(p=0.3),
        ToTensorV2(p=1.0),
    ],
    bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_val_transforms():
    return A.Compose([
        ToTensorV2(p=1.0),
    ],
    bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

# train_ppr = Preprocessor()
# train_ppr.visualize_random_image()
# train_ppr.resize_images(400, 400)
# # # train_ppr.normalize_images()
# train_ppr.transform_images(get_train_transforms())
# train_ppr.visualize_random_image()

# # # print(train_ppr.bboxes[0])