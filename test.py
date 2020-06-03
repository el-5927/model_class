from PIL import Image
import time
import cv2
import argparse

import torch
import torchvision.transforms as transform
from torchvision import models
import torch.nn as nn

def test(pre_weight, val_txt,):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = transform.Compose([
        transform.Resize(320),
        transform.RandomResizedCrop(224),
        transform.ToTensor()
    ])

    model = models.vgg16(pretrained=False).to(device)
    model.classifier[6] = nn.Linear(4096, 200)
    checkpoint = torch.load(pre_weight)

    # model.load_state_dict(checkpoint['net'])
    model.load_state_dict({k.replace('net.',''):v for k,v in checkpoint['net'].items()})
    # model.load_state_dict(torch.load(pre_weight))
    model = model.to(device)
    model.eval()

    lines = []
    with open("./dataset/CUB/" + val_txt, "r") as f:
        lines = f.readlines()
    # print(lines)

    right = 0
    all1 = len(lines)
    for index,ind in enumerate(lines):

        img_name = ind.rstrip('\n').split(' ')[0]
        label = ind.rstrip('\n').split(' ')[1]

        img_path = "./dataset/CUB/data/" + img_name
        print(str(index) + " " + img_path)

        #处理图片
        img = Image.open(img_path).convert('RGB')
        
        img = transforms(img)
        img = img.unsqueeze(0)
        img_ = img.to(device)

        output = model(img_)

        _, predicted = torch.max(output, 1)
        pre = predicted.item()

        if int(label) - 1 == int(pre):
            right += 1
            # print(right)
    
    print("Total: {}, Right: {}, Acc: {}".format(len(lines), right, float(right)/len(lines)))
    time.sleep(2)
    # print("right: " + str(right))
    # print("all: " + str(all1))
    # print("acc: " + str(float(right)/len(lines)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--pre_weight", default=None, type=str)
    parser.add_argument("--val_txt", default=None, type=str)

    args = parser.parse_args()

    test(args.pre_weight, args.val_txt)
