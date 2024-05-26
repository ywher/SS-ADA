import yaml

class tmp():
    def __init__(self, config):
        self.config = config
    

if __name__ == '__main__':
    yaml_path = "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/configs/cityscapes_acda_bisenetv1.yaml"
    cfg = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)
    temp = tmp(cfg)
    for i in range(3):
        cfg['data_list'] = str(i)
        cfg['n_sup'] = i
        print('cfg data_list:', cfg['data_list'])
        print('cfg n_sup:', cfg['n_sup'])
        print('temp.config data_list:', temp.config['data_list'])
        print('temp.config n_sup:', temp.config['n_sup'])