from model.vgg import vgg16
from model.vgg import VGG16_Finetuning

def load_from_pretrain(net_lfp, premodel_path):
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


def build_model(args, cfg, device):

    if 'VGG16_Finetuning' == cfg.MODEL.NAME:
        model = VGG16_Finetuning.VGG16_Finetuning(cfg.DATASETS.NUM_CLASSES).to(device)
    
    if 'VGG16' == cfg.MODEL.NAME:
        model = vgg16.VGG16(cfg['num_classes']).to(device)
        # if cfg.MODEL.PRE_TRAIN:
        #     model = load_from_pretrain(model, cfg.MODEL.PRETRAIN_MODEL)

    
    return model