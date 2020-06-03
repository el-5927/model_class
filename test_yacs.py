from utils.defaults import get_cfg_defaults

if __name__ == "__main__":
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file("D:\\code\\model2class\\vgg16\\configs\\config.yml")
    cfg.freeze()

    print(cfg)
    print("lr_polocy: {}".format(cfg.TRAIN_SET.LR_POLOCY))
    exit()