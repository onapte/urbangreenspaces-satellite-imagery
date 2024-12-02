from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import SSD300_VGG16_Weights
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD

from config import DEVICE, NUM_EPOCHS, OUT_DIR, NUM_WORKERS, MODEL_BASE
from dataset import create_train_dataset, create_val_dataset, create_train_dataloader, create_val_dataloader
from utils import create_loss_plot, create_mAP_plot
from models import SSD300, EffDet, FasterRCNN, RetinaNet

import os
import time
import warnings
import argparse

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        choices=['ssd300', 'effdet', 'fasterrcnn', 'retinanet'],
                        help='Model architecture to use.')
    
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the dataset.')

    parser.add_argument('--input_size', type=int, default=300, help='Input size of the model.')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation.')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer.')

    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer.')

    parser.add_argument('--step_size', type=int, default=15, help='Step size for the learning rate scheduler.')

    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay factor for the scheduler.')

    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs.')

    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu).')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create datasets
    train_dataset = create_train_dataset()
    val_dataset = create_val_dataset()

    # Create dataloaders
    train_loader = create_train_dataloader(train_dataset)
    val_loader = create_val_dataloader(val_dataset)

    print(f"Number of training samples: {len(train_dataset)}")

    # Change config
    MODEL_BASE = args.model.upper()
    NUM_EPOCHS = args.epochs
    IMAGE_SIZE = args.input_size
    BATCH_SIZE = args.batch_size

    # Select model
    model_class = None

    if args.model == 'ssd300':
        if args.input_size != 300:
            raise ValueError(
                f"ssd300 requires image size 300 X 300. Given {args.input_size} X {args.input_size}."
            )

        model_class = SSD300(num_classes=args.num_classes,
                            input_size=args.input_size,
                            weights=SSD300_VGG16_Weights)
        
    elif args.model == 'effdet':
        model_class = EffDet(num_classes=args.num_classes,
                            input_size=args.input_size,
                            weights=SSD300_VGG16_Weights)
        
    elif args.model == 'fasterrcnn':
        model_class = FasterRCNN(num_classes=args.num_classes,
                            input_size=args.input_size,
                            weights=SSD300_VGG16_Weights)
        
    elif args.model == 'retinanet':
        model_class = RetinaNet(num_classes=args.num_classes,
                            input_size=args.input_size,
                            weights=SSD300_VGG16_Weights)
    
    else:
        raise ValueError(
            f"Selected model ({args.model}) not supported!"
            "Choices: {['ssd300', 'effdet', 'fasterrcnn', 'retinanet']}"
        )

    # Create model
    model_class.create_model()
    model_class.model.to(DEVICE)

    # Initialize hyperparameters
    params = [p for p in model_class.model.parameters() if p.requires_grad]

    optimizer = SGD(model_class.model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    scheduler = StepLR(
        optimizer=optimizer, step_size=args.step_size, gamma=args.gamma, verbose=True
    )

    # Create lists to store train loss and map metrics
    train_loss_list = []
    map50_list = []
    map_list = []

    # Start training
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        start = time.time()
        train_loss = model_class.train_step(train_loader, optimizer)
        metric_summary = model_class.validation_step(val_loader)
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

    # Create loss and map plots
    create_loss_plot(train_loss_list, x_label='Epoch', y_label='Train loss', save_name='Train_Loss')
    create_mAP_plot(mAP_data, save_name='mAP')