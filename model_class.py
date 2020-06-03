from torchvision import models
from model import vgg16, ResNet, VGG16_Finetuning
from torch.autograd import Variable
from utils import config

class model_cla():

    def __init__(self, model_type, num_class, device, pre_model):

        self.model_type = model_type
        self.num_class = num_class

        if self.model_type == "VGG16_Finetuning":
            self.model = VGG16_Finetuning.VGG16_Finetuning(self.num_class).to(device)
            
        if self.model_type == 'vgg16':
            self.model = vgg16.VGG16(self.num_class).to(device)
            self.model = self.load_from_pretrain(self.model, pre_model)

        if self.model_type == 'resnet':
            self.model = ResNet.ResNet50(self.num_class).to(device)

        


    def load_from_pretrain(self, net_lfp, premodel_path):
        #https://blog.csdn.net/qq_27036771/article/details/103638134
        #当前网络的参数
        model_dict = net_lfp.state_dict()
        #加载vgg模型
        load_dict = torch.load(premodel_path)

        key = list(model_dict.keys())
        name = list(load_dict.keys())
        weights = list(load_dict.values())
        
        # for k, v in model_dict.items():
        #     print(k, v.shape)
        # exit()

        t = 0
        for i in range(len(weights)):
            #不加载最后的全连接层
            if 'classifier' in name[i]:
                break
            #当前模型使用BN层多一个num_batch_tracked, 但是加载的模型中没有，因此需要跳过
            if 'num_batches_tracked' in key[i+t]:
                t += 1
            model_dict[key[i+t]] = weights[i]
        
        net_lfp.load_state_dict(model_dict, strict=False)

        return net_lfp

    


