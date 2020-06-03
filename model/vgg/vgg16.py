from torch import nn
import torch.nn.functional as F

class VGG16(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        net = []
        
        #block1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)  #64 * 222 * 222
        self.BN1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1) #64 * 222 * 222
        self.BN2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # 64 * 112 *112

        # block 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1) #128 * 110 * 110
        self.BN3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) #128 * 110 * 110
        self.BN4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) #128 * 56 * 56 

        # block 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1) # 256 * 54 * 54
        self.BN5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1) # 256 * 54 * 54
        self.BN6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1) # 256 * 54 * 54
        self.BN7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # block 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1) # 512 * 26 * 26
        self.BN8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1) # 512 * 26 * 26
        self.BN9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1) # 512 * 26 * 26
        self.BN10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # 512 * 14 * 14

        # block 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1) # 512 * 12 * 12
        self.BN11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1) # 512 * 12 * 12
        self.BN12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1) # 512 * 12 * 12
        self.BN13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # pooling 512 * 7 * 7

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

    
    def forward(self, x):
        
        # x.size(0)即为batch_size
        in_size = x.size(0)
        
        out = self.conv1_1(x) # 222
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.conv1_2(out) # 222
        out = self.BN2(out)
        out = self.relu2(out)
        out = self.maxpool1(out) # 112
        
        out = self.conv2_1(out) # 110
        out = self.BN3(out)
        out = self.relu3(out)
        out = self.conv2_2(out) # 110
        out = self.BN4(out)
        out = self.relu4(out)
        out = self.maxpool2(out) # 56
        
        out = self.conv3_1(out) # 54
        out = self.BN5(out)
        out = self.relu5(out)
        out = self.conv3_2(out) # 54
        out = self.BN6(out)
        out = self.relu6(out)
        out = self.conv3_3(out) # 54
        out = self.BN7(out)
        out = self.relu7(out)
        out = self.maxpool3(out) # 28
        
        out = self.conv4_1(out) # 26
        out = self.BN8(out)
        out = self.relu8(out)
        out = self.conv4_2(out) # 26
        out = self.BN9(out)
        out = self.relu9(out)
        out = self.conv4_3(out) # 26
        out = self.BN10(out)
        out = self.relu10(out)
        out = self.maxpool4(out) # 14
        
        out = self.conv5_1(out) # 12
        out = self.BN11(out)
        out = self.relu11(out)
        out = self.conv5_2(out) # 12
        out = self.BN12(out)
        out = self.relu12(out)
        out = self.conv5_3(out) # 12
        out = self.BN13(out)
        out = self.relu13(out)
        out = self.maxpool5(out) # 7
        
        # 展平
        out = out.view(in_size, -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        
        out = F.log_softmax(out, dim=1)
        
        
        return out

