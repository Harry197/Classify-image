import argparse
import os
import csv
import json
from tqdm import tqdm

import timm
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import MyDataset

import torchvision.models as models
from adabelief_pytorch import AdaBelief
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Resize(256, 256),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(), ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1), ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(), ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.025,
                rotate_limit=5,
                shift_limit_x=0.025,
                shift_limit_y=0.025,
                p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif mode == 'test':
        return A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(
                    mean=[
                        0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        )


def dataloader(args):
    trainset = MyDataset(
        csv_file=args.train_csv,
        root_dir=args.root_dir,
        transform=get_transform('train'))
    trainloader = DataLoader(
        trainset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.worker)
    testset = MyDataset(
        csv_file=args.val_csv,
        root_dir=args.root_dir,
        transform=get_transform('test'))
    testloader = DataLoader(
        testset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.worker)
    return trainloader, testloader


def train(args):
    try:
        trainloader, testloader = dataloader(args=args)
        classes = []
        data_acc = {}
        with open(args.classes_path, 'r') as csv_file:
            readers = csv.reader(csv_file)
            for reader in readers:
                classes.append(reader)

        num_classes = len(classes)

        if args.gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device('cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)

        net = timm.create_model(
            "resnet18",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.2)
        net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        acc_best = -1
        print('Start Training')
        for epoch in tqdm(range(args.epochs)):
            print("Epoch {}/{}".format(epoch, args.epochs))
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % 10 == 9:
                    print('Epoch: %d, Iter: %5d, loss: %.3f' %
                        (epoch, i + 1, running_loss / 10))
                    running_loss = 0.0
            print('Eval')
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    scores, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # print(predicted, labels)
                    correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            print(
                'Accuracy of the network on the test images: %d %%' %
                (100 * correct / total))
            if acc > acc_best:
                model_path = os.path.join(
                    args.checkpoint_path,
                    'resnet18_epoch_{}_acc_{}.pth'.format(
                        epoch,
                        round(
                            acc,
                            2)))
                torch.save(net.state_dict(), model_path)
                acc_best = acc
                data_acc['accuracy'] = acc_best
                with open('result.json', 'w') as json_file:
                    json.dump(data_acc, json_file)
            net.train()
            scheduler.step()

        print('Finished Training')
    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--classes_path', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoints/')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--worker', type=int, default=4)
    parser.add_argument('--task_id', type=str, default='',
                        help="Task's ID to update progress")

    args = parser.parse_args()
    os.makedirs(args.checkpoint_path, exist_ok=True)

    train(args=args)
