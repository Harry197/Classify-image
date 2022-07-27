import torch
import torchvision.transforms as transforms

import numpy as np
import timm
from dataset import MyDataset
import matplotlib.pyplot as plt
from PIL import Image
import csv
import torchvision.transforms as transforms
from torch.autograd import Variable

loader = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    # image = Variable(image, requires_grad=True)
    image = torch.unsqueeze(image, 0)  #this is for VGG, may not be needed for ResNet
    # print(image.shape)
    return image

datas=[]
with open('./bhxh/data.csv', 'r') as csv_f:
    reader = csv.reader(csv_f)
    for i in reader:
        datas.append(i)

device = torch.device("cpu")

model = timm.create_model("resnet18",pretrained=True, num_classes=4, drop_rate=0.2)
model.load_state_dict(torch.load('./id_mark_rn18.pth', map_location=device))
model.to(device)

model.eval()

# dict_label = {0:'dl', 1:'id_new', 2:'passport'}

eval = 0
with torch.no_grad():
    with open('./result.csv', 'a+') as csv_file:
        writer = csv.writer(csv_file)
        for data in datas:
            images = image_loader(data[0]).to(device)
            outputs = model(images)
            print(outputs)
            scores, predicted = torch.max(outputs.data, 1)
            print(scores)
            print(predicted)

            writer.writerow([data[0], int(predicted), int(data[1])])
            if predicted == int(data[1]):
                eval += 1
            else:
                print(data[0], '-', int(predicted), '-', int(data[1]))

            break

print(eval)