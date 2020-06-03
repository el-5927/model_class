#https://www.jianshu.com/p/5f2f6acdc7d3
from yacs.config import CfgNode as CN

#创建一个配置结点_C
_C = CN()

#配置MODEL节点
_C.MODEL = CN()

#模型名称
_C.MODEL.NAME = 'VGG16_Finetuning'
#预训练模型
_C.MODEL.PRETRAIN_MODEL = "PRETRAIN_MODEL"
#模型输入尺寸
_C.MODEL.INPUT_SIZE = 224
#是否使用预训练模型
_C.MODEL.PRE_TRAIN = False

#配置DATASETS节点
_C.DATASETS = CN()

#数据集名称
_C.DATASETS.NAMES = "NAMES"
#训练集路径
_C.DATASETS.TRAIN_PATH = 'xxxx'
#测试集路径
_C.DATASETS.VAL_TXT = 'xxxx'
#训练txt文件
_C.DATASETS.TRAIN_TXT = 'xxxx'
#测试txt文件
_C.DATASETS.VAL_PATH = 'xxxx'
#数据集分类类别
_C.DATASETS.NUM_CLASSES = 10

#配置TRAIN_SET节点
_C.TRAIN_SET = CN()

#学习算法
_C.TRAIN_SET.LR_POLOCY = 'MULP'
#学习率
_C.TRAIN_SET.BASE_LR = 0.001
#使用的数据集
_C.TRAIN_SET.DATASET_TYPE = "CUB"



#配置WORK_SAVE节点
_C.WORK_SAVE = CN()

#模型保存的路径
_C.WORK_SAVE.MODEL_PATH = './workdir'
#训练logs保存路径
_C.WORK_SAVE.LOGS_PATH =  './logs'

def get_cfg_defaults():

    return _C.clone()