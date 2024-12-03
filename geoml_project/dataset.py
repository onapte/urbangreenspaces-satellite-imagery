import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from itertools import chain
import matplotlib.pyplot as plt

from config import TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS, MODEL_BASE, IMAGE_SIZE
from data_processing import Preprocessor, get_train_transforms, get_val_transforms



class GeoDataset(Dataset):
    def __init__(self, data, transforms=None, bbox_format='PASCAL_VOC', target_bbox_format='PASCAL_VOC'):
        self.transforms = transforms
        self.PIL_images = data['images']
        self.bboxes = data['boxes']
        self.labels = data['labels']
        self.bbox_format = bbox_format
        self.target_bbox_format = target_bbox_format
        self.labels_mapping = None

        self.encode_labels()

    def encode_labels(self):
        unique_labels = ['background'] + sorted(list(set(chain.from_iterable(self.labels))))

        labels_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        labels_encoded = [[labels_mapping[label] for label in label_list] for label_list in self.labels]

        self.labels = labels_encoded
        self.labels_mapping = labels_mapping

    def visualize_image(self, idx):
        image = self.PIL_images[idx]
        plt.imshow(image)
        plt.show()

    def __getitem__(self, idx):
        PIL_image = self.PIL_images[idx]
        bboxes = self.bboxes[idx]
        labels = self.labels[idx]

        if self.bbox_format == "PASCAL_VOC" and self.target_bbox_format == "OBB":
            for ix, bbox_tr in enumerate(bboxes):

                xmin, ymin, xmax, ymax = bbox_tr

                top_left = [xmin, ymin]
                top_right = [xmax, ymin]
                bottom_right = [xmax, ymax]
                bottom_left = [xmin, ymax]

                bboxes[ix] = [top_left, top_right, bottom_right, bottom_left]

        if self.bbox_format == "OBB" and self.target_bbox_format == "PASCAL_VOC":
            for ix, bbox in enumerate(bboxes):
                xmin = min(v[0] for v in bbox)
                ymin = min(v[1] for v in bbox)
                xmax = max(v[0] for v in bbox)
                ymax = max(v[1] for v in bbox)
                
                bbox_mod = [xmin, ymin, xmax, ymax]
                
                bboxes[ix] = bbox_mod
            

        transform_toTensor = T.ToTensor()
        transform_Normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = transform_toTensor(PIL_image)
        image_tensor = transform_Normalize(image_tensor)

        bboxes_tensor = torch.FloatTensor(bboxes)
        labels_tensor = torch.LongTensor(labels)

        if self.transforms:
            pass

        return image_tensor, bboxes_tensor, labels_tensor
        
    def __len__(self):
        return len(self.PIL_images)
    

def create_train_dataset(DIR=TRAIN_DIR):
    train_ppr = Preprocessor(
        root_dir=DIR,
    )
    
    train_ppr.resize_images(target_width=IMAGE_SIZE,
                            target_height=IMAGE_SIZE)
    
    train_ppr.transform_images(get_train_transforms())

    data = {
    'images': train_ppr.images,
    'boxes': train_ppr.bboxes,
    'labels': train_ppr.labels
    }

    train_dataset = GeoDataset(data)
    return train_dataset


def create_val_dataset(DIR=VAL_DIR):
    val_ppr = Preprocessor(
        root_dir=DIR,
    )
    
    val_ppr.resize_images(target_width=IMAGE_SIZE,
                            target_height=IMAGE_SIZE)
    val_ppr.transform_images(get_val_transforms())

    data = {
    'images': val_ppr.images,
    'boxes': val_ppr.bboxes,
    'labels': val_ppr.labels
    }

    val_dataset = GeoDataset(data)
    return val_dataset


def create_train_dataloader(train_dataset, num_workers=NUM_WORKERS):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn
    )

    return train_dataloader


def create_val_dataloader(val_dataset, num_workers=NUM_WORKERS):
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn
    )

    return val_dataloader


def collate_fn(batch, model_base=MODEL_BASE):
    # if model_base == "EffDet":
    #     images, targets = tuple(zip(*batch))
    #     images = torch.stack(images)

    #     boxes = [target['boxes'] for target in targets]
    #     labels = [target['labels'] for target in targets]

    #     targets_mod = {
    #         'bbox': boxes,
    #         'cls': labels
    #     }

    #     return images, targets_mod
    
    images = list()
    boxes = list()
    labels = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    images = torch.stack(images, dim=0)

    return images, boxes, labels