from models import SSD300, EffDet, FasterRCNN, RetinaNet
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import SSD300_VGG16_Weights
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from tqdm.auto import tqdm
from config import DEVICE, NUM_EPOCHS, OUT_DIR, NUM_WORKERS
from dataset import create_train_dataset, create_val_dataset, create_train_dataloader, create_val_dataloader
import os
import time
import torch
import warnings

from utils import create_loss_plot, create_mAP_plot

warnings.filterwarnings("ignore")

def train(model, train_dataloader, optimizer):
    print('Training')

    prog_bar = tqdm(train_dataloader, total=len(train_dataloader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        # images = list(image.to(DEVICE) for image in images)
        # targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        images = torch.stack([image.to(DEVICE) for image in images])
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        print(images.shape)
        print(len(targets))

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        losses.backward()
        optimizer.step()

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    return loss_value

if __name__ == '__main__':
    #torch.backends.cudnn.enabled = False
    os.makedirs('output', exist_ok=True)
    train_dataset = create_train_dataset()
    val_dataset = create_val_dataset()

    train_loader = create_train_dataloader(train_dataset)
    val_loader = create_val_dataloader(val_dataset)

    print(f"Number of training samples: {len(train_dataset)}")

    model_class = SSD300(num_classes=5,
                         input_size=300,
                         weights=SSD300_VGG16_Weights)

    # model_class = EffDet(num_classes=4,
    #                      input_size=512,
    #                      weights=None)

    # model_class = FasterRCNN(num_classes=5,
    #                          input_size=512,
    #                          weights=None)

    # model_class = RetinaNet(num_classes=4,
    #                          input_size=512,
    #                          weights=None)
    
    model_class.create_model()
    # model = model_class.model
    model_class.model.to(DEVICE)

    params = [p for p in model_class.model.parameters() if p.requires_grad]
    optimizer = SGD(model_class.model.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=15, gamma=0.1, verbose=True
    )

    train_loss_list = []
    map50_list = []
    map_list = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        start = time.time()
        train_loss = model_class.train_step(train_loader, optimizer)
        metric_summary = model_class.validation_step(val_loader, optimizer)
        end = time.time()

        avg_train_loss = model_class.avg_train_loss

        print(f"Epoch #{epoch+1} train loss: {avg_train_loss:.3f}")   
        print(f"Epoch #{epoch+1} mAP: {metric_summary['map']}") 
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        mAP_data = {
            'map50': map50_list,
            'map': map_list
        }

        scheduler.step()

    create_loss_plot(train_loss_list, x_label='Epoch', y_label='Train loss', save_name='Train_Loss')
    create_mAP_plot(mAP_data, save_name='mAP')