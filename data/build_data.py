from data import cifar10, CUB, dog_class
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader

def build_data_loader(args, cfg):
    
    transforms = transform.Compose([
        transform.Resize(320),
        transform.RandomResizedCrop(cfg.MODEL.INPUT_SIZE),
        transform.ToTensor()
    ])

    if cfg.DATASETS.NAMES == 'cifar10':
        train_data = cifar10.cifar10_data(cfg.DATASETS.TRAIN_PATH, cfg.DATASETS.TRAIN_TXT, transform=transforms)
        test_data =  cifar10.cifar10_data(cfg.DATASETS.VAL_PATH, cfg.DATASETS.VAL_TXT, transform=transforms)

        train_loader = DataLoader(dataset=train_data, batch_size = args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size = args.val_size, shuffle=True)
    
    elif cfg.DATASETS.NAMES == 'dog_class':
        pass
    elif cfg.DATASETS.NAMES == 'CUB':
        train_data = CUB.CUB_data(cfg.DATASETS.TRAIN_PATH, cfg.DATASETS.TRAIN_TXT, transform=transforms)
        test_data =  CUB.CUB_data(cfg.DATASETS.VAL_PATH, cfg.DATASETS.VAL_TXT, transform=transforms)

        train_loader = DataLoader(dataset=train_data, batch_size = args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size = args.val_size, shuffle=True)


    return train_loader, test_loader