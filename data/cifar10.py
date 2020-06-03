import torch
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms


class cifar10_data(Dataset):

    def __init__(self, data_root, label_txt, transform=None, augment=None):
        super(cifar10_data, self).__init__()
        
        img_label = []
        with open(os.path.join(data_root, label_txt), 'r') as f:

            for line in f:

                temp = line.rstrip().split(' ')
                img_label.append([temp[0], int(temp[1])])
        
        self.root = data_root
        self.img_label = img_label
        self.transform = transform
        self.data_size = len(img_label)

    def __getitem__(self, index):

        img_path = os.path.join(self.root, self.img_label[index][0])
        label = self.img_label[index][1]

        if not os.path.exists(img_path):
            print("The image is not exist: " + img_path)
            exit()
        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    
    def __len__(self):
        return self.data_size

    
# train_data = cifar10_data('D:\code\model2class\dataset\cifar-10-python\cifar-10-batches-py', 'test.txt', transform=transforms.ToTensor())


