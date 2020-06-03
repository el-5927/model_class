import torch
import argparse
import os
from datetime import datetime
from tqdm import *

import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from data import build_data
from utils.defaults import get_cfg_defaults
from model.build_model import build_model
from test import *



def train(args, cfg):
    

    #加载数据
    train_loader, test_loader = build_data.build_data_loader(args, cfg)  
    print("------------load dataset success !---------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # net = VGG16_Finetuning.VGG16_Finetuning(cfg.DATASETS.NUM_CLASSES).to(device)
    net = build_model(args, cfg, device)
    print("-------------load model success !----------------")
    
    if "SGD" == cfg.TRAIN_SET.LR_POLOCY: 
        optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN_SET.BASE_LR, momentum=0.9)
    if "Adam" == cfg.Train_SET.LR_POLOCY:
        
        pass
    #optimizer = optim.Adam(net.parameters(), lr=cfg['lr'])

    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,70,100,200],gamma=0.1)

    criterion = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    if args.RESUME:
        
        checkpoint = torch.load(args.path_checkpoint)  # 断点路径

        net.load_state_dict(checkpoint['net']) # 加载断点

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        schedule.load_state_dict(checkpoint['schedule'])   #加载学习率的状态
    else:
        if cfg.MODEL.PRE_TRAIN:
            net.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_MODEL))



    for epoch in range(start_epoch+1, args.epoch):
        running_loss = 0.0
        total = 0.0 
        correct = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            img, label = data
            
            #input处理
            img = img.to(device)
            label = label.to(device)

            img, label = Variable(img), Variable(label)
            
            #推断、损失计算和反向传播
            optimizer.zero_grad()  #梯度清零
            outputs = net(img)     #inference
            loss = criterion(outputs, label)   #求解loss
            loss.backward()        #反向传播求解梯度
            optimizer.step()       #更新权重参数

            #计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += torch.sum(predicted == label.data).cpu().numpy()
            
            running_loss += loss.item()

            if i % 100 == 99:

                loss_avg = running_loss / 100
                print(
                "Time: {:.19s} Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{}"
                .format(
                    str(datetime.now()), epoch, args.epoch, i + 1,
                    len(train_loader), loss_avg, correct / total))

        
        schedule.step()
        

        #模型保存
        if (epoch + 1) % 50 == 0:
            print("epoch: ", epoch)
            print('learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])

            checkpoint = {
                "net": net.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch": epoch,
                "schedule": schedule.state_dict()
            }
            if not os.path.exists(cfg.WORK_SAVE.MODEL_PATH):
                os.mkdir(cfg.WORK_SAVE.MODEL_PATH)
            torch.save(checkpoint, cfg.WORK_SAVE.MODEL_PATH + '/' + str(epoch) + ".pth")
            
            #模型测试
            test(cfg.WORK_SAVE.MODEL_PATH + '/' + str(epoch) + ".pth", cfg.DATASETS.VAL_TXT)
            # torch.save(checkpoint, cfg_data['save_model'] + f'/{epoch}.pth')

            # save_model = os.path.join(cfg_data['save_model'], f'vgg16_{epoch}_model.pth')
            # if not os.path.exists(cfg_data['save_model']):
            #     os.mkdir(cfg_data['save_model'])
            # torch.save(net.state_dict(), save_model)









if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pre_weight", default=True, type=str, help="pre train weight")
    parser.add_argument("--RESUME", default=False, type=bool, help="断点继续训练")
    parser.add_argument("--path_checkpoint", default=None, help="断点加载的模型参数")
    parser.add_argument("--batch_size", default=16, type=int, help="训练batch_size")
    parser.add_argument("--val_size", default=8, type=int, help="val的size")
    parser.add_argument("--epoch", default=2000, type=int, help="训练批次")
    parser.add_argument("--yml", default='VGG16_Finetuning.yml', type=str, help="yml配置文件")

    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./configs/" + args.yml)
    cfg.freeze()

    train(args, cfg)