import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
from tqdm import tqdm

import transforms as ext_transforms
from models.enet import ENet
from models.unet import Unet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

from data.neuseg import neuseg as neuseg

# Get the arguments
args = get_arguments()

device = torch.device(args.device)

def load_dataset(dataset):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = next(iter(train_loader))
    else:
        images, labels = next(iter(train_loader))
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # Show a batch of samples and labels
    # if args.imshow_batch:
    #     print("Close the figure window to continue...")
    #     label_to_rgb = transforms.Compose([
    #         ext_transforms.LongTensorToRGBPIL(class_encoding),
    #         transforms.ToTensor()
    #     ])
    #     color_labels = utils.batch_transform(labels, label_to_rgb)
    #     utils.imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0
    if args.weighing.lower() == 'enet':
        class_weights = enet_weighing(train_loader, num_classes)
    elif args.weighing.lower() == 'mfb':
        class_weights = median_freq_balancing(train_loader, num_classes)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding


class Predict:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, filepath, device):
        self.model = model
        self.filepath = filepath
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor()
        ])


    def run_epoch(self):
        # 使用 tqdm 加载图片并进行预测
        image_files = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(self.filepath, image_file)
            image = Image.open(image_path)
            image = self.transform(image).unsqueeze(0)  # 添加批次维度
            image = image.to(self.device)

            # 进行预测
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(image)

            # 将预测结果转换为单通道
            _, predictions = torch.max(predictions.data, 1)

            # 确保输出形状为 (200, 200)
            output_np = predictions.cpu().numpy()
            # if output_np.shape[1] != 200 or output_np.shape[2] != 200:
            output_np = np.resize(output_np, (200, 200)).astype(np.uint8)

            # 保存每个输出为单独的 .npy 文件
            np.save(f"predict/{image_file.split('.')[0]}.npy", output_np)


# Run only if this module is being run directly
if __name__ == '__main__':

    loaders, w_class, class_encoding = load_dataset(neuseg)
    train_loader, val_loader, test_loader = loaders

    num_classes = len(class_encoding)
    
    model = ENet(num_classes).to(device)
    # model = Unet(3, num_classes).to(device)

    test_file_path = "data/neuseg/test/images"

    # Load the previoulsy saved model state to the ENet model
    # checkpoint = torch.load('save/ENet', weights_only=True)
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(torch.load('train_Iou73.079.pth', weights_only=True))
    # model.load_state_dict(torch.load('save/ENet'))

    # 创建 Predictor 实例并运行
    predictor = Predict(model, test_file_path, device)
    predictor.run_epoch()