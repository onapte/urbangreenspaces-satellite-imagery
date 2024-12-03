import torch
from tqdm.auto import tqdm
from config import DEVICE, IMAGE_SIZE
from functools import partial
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from math import sqrt

from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils, ssd300_vgg16

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, retinanet_resnet50_fpn_v2

from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


class SSD300:
    def __init__(self, num_classes, input_size, weights):
        self.weights = weights
        self.num_classes = num_classes
        self.input_size = input_size
        self.model = None

        self.avg_train_loss = 0

    def create_model(self):
        self.model = ssd300_vgg16(
            weights=self.weights
        )

        in_channels = _utils.retrieve_out_channels(self.model.backbone, (self.input_size, self.input_size))
        num_anchors = self.model.anchor_generator.num_anchors_per_location()

        self.model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )

        self.model.transform.min_size = (self.input_size,)
        self.model.transform.max_size = self.input_size


    def train_step(self, train_dataloader, optimizer):
        print('Training')
        self.model.train()

        prog_bar = tqdm(train_dataloader, total=len(train_dataloader))
        train_loss_sum = 0

        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, boxes, labels = data

            # images = images.to(DEVICE)
            # boxes = [b.to(DEVICE) for b in boxes]
            # labels = [l.to(DEVICE) for l in labels]

            filtered_data = [
                (img, b, l)
                for img, b, l in zip(images, boxes, labels)
                if len(b) > 0 and len(l) > 0
            ]

            images, boxes, labels = zip(*filtered_data)
            images = torch.stack(images).to(DEVICE) 
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            targets = [
                {
                    "boxes": box,
                    "labels": label
                }
                for box, label in zip(boxes, labels)
            ]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()
            train_loss_sum += loss_value

            optimizer.step()
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        self.avg_train_loss = 1 * train_loss_sum / len(train_dataloader)
        
        return loss_value
    
    def validation_step(self, val_dataloader):
        print('Validating')
        
        self.model.eval()

        prog_bar = tqdm(val_dataloader, total=len(val_dataloader))
        target = []
        preds = []
        for i, data in enumerate(prog_bar):
            images, boxes, labels = data

            # images = images.to(DEVICE)
            # boxes = [b.to(DEVICE) for b in boxes]
            # labels = [l.to(DEVICE) for l in labels]

            filtered_data = [
                (img, b, l)
                for img, b, l in zip(images, boxes, labels)
                if len(b) > 0 and len(l) > 0
            ]

            images, boxes, labels = zip(*filtered_data)
            images = torch.stack(images).to(DEVICE) 
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            targets = [
                {
                    "boxes": box,
                    "labels": label
                }
                for box, label in zip(boxes, labels)
            ]
            
            with torch.no_grad():
                outputs = self.model(images, targets)

            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                target.append(true_dict)

        metric = MeanAveragePrecision()
        metric.update(preds, target)
        metric_summary = metric.compute()
        return metric_summary



class EffDet:
    def __init__(self, num_classes, input_size, weights):
        self.model = None
        self.num_classes = num_classes
        self.config = get_efficientdet_config('tf_efficientdet_d0')
        self.input_size = input_size

        self.avg_train_loss = 0

    def create_model(self):
        efficientdet_model_param_dict['tf_efficientdet_d0'] = dict(
            name='tf_efficientdet_d0',
            backbone_name='tf_efficientdet_d0',
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=self.num_classes,
            url=''
        )

        self.config.update({'num_classes': self.num_classes})
        self.config.update({'image_size': (self.input_size, self.input_size)})

        net = EfficientDet(self.config, pretrained_backbone=True)
        net.class_net = HeadNet(
            self.config,
            num_outputs=self.config.num_classes
        )

        self.model = DetBenchTrain(net, self.config)
    
    def train_step(self, train_dataloader, optimizer):
        print('Training')

        self.model.train()
        prog_bar = tqdm(train_dataloader, total=len(train_dataloader))
        train_loss_sum = 0

        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, boxes, labels = data

            # images = images.to(DEVICE)
            # boxes = [b.to(DEVICE) for b in boxes]
            # labels = [l.to(DEVICE) for l in labels]

            filtered_data = [
                (img, b, l)
                for img, b, l in zip(images, boxes, labels)
                if len(b) > 0 and len(l) > 0
            ]

            images, boxes, labels = zip(*filtered_data)
            images = torch.stack(images).to(DEVICE) 
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            targets = {
                    "bbox": boxes,
                    "cls": labels
            }

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_sum += loss_value

            losses.backward()
            optimizer.step()

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        self.avg_train_loss = 1 * train_loss_sum / len(train_dataloader)
        
        return loss_value
    
    def validation_step(self, val_dataloader):
        print('Validating')
        
        self.model.eval()

        prog_bar = tqdm(val_dataloader, total=len(val_dataloader))
        target = []
        preds = []

        for i, data in enumerate(prog_bar):
            images, boxes, labels = data

            # images = images.to(DEVICE)
            # boxes = [b.to(DEVICE) for b in boxes]
            # labels = [l.to(DEVICE) for l in labels]

            filtered_data = [
                (img, b, l)
                for img, b, l in zip(images, boxes, labels)
                if len(b) > 0 and len(l) > 0
            ]

            images, boxes, labels = zip(*filtered_data)
            images = torch.stack(images).to(DEVICE) 
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            targets = {
                    "bbox": boxes,
                    "cls": labels,
                    "img_size": None,
                    "img_scale": None
            }
            
            with torch.no_grad():
                outputs = self.model(images, targets)
            
            detections = outputs['detections']

            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets['bbox'][i].detach().cpu()
                true_dict['labels'] = targets['cls'][i].detach().cpu()
                preds_dict['boxes'] = detections[i][:, :4].detach().cpu()
                preds_dict['scores'] = detections[i][:, 4].detach().detach().cpu()
                preds_dict['labels'] = detections[i][:, 5].detach().cpu().to(torch.int64)
                preds.append(preds_dict)
                target.append(true_dict)

        metric = MeanAveragePrecision()
        metric.update(preds, target)
        metric_summary = metric.compute()
        return metric_summary



class FasterRCNN:
    def __init__(self, num_classes, input_size, weights=None):
        self.num_classes = num_classes
        self.input_size = input_size
        self.weights = weights
        self.model = None
        self.avg_train_loss = 0
    
    def create_model(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
    
    def train_step(self, train_dataloader, optimizer):
        print('Training')

        self.model.train()
        prog_bar = tqdm(train_dataloader, total=len(train_dataloader))
        train_loss_sum = 0

        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, boxes, labels = data

            # images = images.to(DEVICE)
            # boxes = [b.to(DEVICE) for b in boxes]
            # labels = [l.to(DEVICE) for l in labels]

            filtered_data = [
                (img, b, l)
                for img, b, l in zip(images, boxes, labels)
                if len(b) > 0 and len(l) > 0
            ]

            images, boxes, labels = zip(*filtered_data)
            images = torch.stack(images).to(DEVICE) 
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            targets = [
                {
                    "boxes": box,
                    "labels": label
                }
                for box, label in zip(boxes, labels)
            ]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            train_loss_sum += loss_value

            losses.backward()
            optimizer.step()

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        self.avg_train_loss = 1 * train_loss_sum / len(train_dataloader)
        
        return loss_value
    
    def validation_step(self, val_dataloader):
        print('Validating')
        
        self.model.eval()

        prog_bar = tqdm(val_dataloader, total=len(val_dataloader))
        target = []
        preds = []
        for i, data in enumerate(prog_bar):
            images, boxes, labels = data

            # images = images.to(DEVICE)
            # boxes = [b.to(DEVICE) for b in boxes]
            # labels = [l.to(DEVICE) for l in labels]

            filtered_data = [
                (img, b, l)
                for img, b, l in zip(images, boxes, labels)
                if len(b) > 0 and len(l) > 0
            ]

            images, boxes, labels = zip(*filtered_data)
            images = torch.stack(images).to(DEVICE) 
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            targets = [
                {
                    "boxes": box,
                    "labels": label
                }
                for box, label in zip(boxes, labels)
            ]
            
            with torch.no_grad():
                outputs = self.model(images, targets)

            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                target.append(true_dict)

        metric = MeanAveragePrecision()
        metric.update(preds, target)
        metric_summary = metric.compute()
        return metric_summary



class RetinaNet:
    def __init__(self, num_classes, input_size, weights=None):
        self.num_classes = num_classes
        self.input_size = input_size
        self.weights = weights
        self.model = None
        self.avg_train_loss = 0
    
    def create_model(self):
        self.model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        )

        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=self.num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
    
    def train_step(self, train_dataloader, optimizer):
        print('Training')

        self.model.train()
        prog_bar = tqdm(train_dataloader, total=len(train_dataloader))
        train_loss_sum = 0

        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, boxes, labels = data

            images = images.to(DEVICE)
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            targets = [
                {
                    "boxes": box,
                    "labels": label
                }
                for box, label in zip(boxes, labels)
            ]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            train_loss_sum += loss_value

            losses.backward()
            optimizer.step()

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        self.avg_train_loss = 1 * train_loss_sum / len(train_dataloader)
        
        return loss_value
    
    def validation_step(self, val_dataloader):
        print('Validating')
        
        self.model.eval()

        prog_bar = tqdm(val_dataloader, total=len(val_dataloader))
        target = []
        preds = []
        for i, data in enumerate(prog_bar):
            images, boxes, labels = data

            images = images.to(DEVICE)
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            targets = [
                {
                    "boxes": box,
                    "labels": label
                }
                for box, label in zip(boxes, labels)
            ]
            
            with torch.no_grad():
                outputs = self.model(images, targets)

            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                target.append(true_dict)

        metric = MeanAveragePrecision()
        metric.update(preds, target)
        metric_summary = metric.compute()
        return metric_summary
