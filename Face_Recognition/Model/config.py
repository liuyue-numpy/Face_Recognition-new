from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('../data')
    conf.work_path = Path('../work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.batch_size = 100 # irse net depth 50 
#   conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        #学习率
        conf.lr = 1e-3

        conf.milestones = [12,15,18]
        #动量
        conf.momentum = 0.9
        # 通常情况下，由于虚拟内存技术的存在，数据要么在内存中以锁页（“pinned”）的方式存在，
        # 要么保存在虚拟内存（磁盘）中。而cuda只接受锁页内存传入，所以在声明新的dataloader对象时，直接令其保存在锁页内存中，
        # 后续即可快速传入cuda。否则，数据需要从虚拟内存中先传入锁页内存，再传入cuda，速度将大大增加。其缺点是对于内存的大小要求较高。
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        # 用的是交叉熵损失函数
        conf.ce_loss = CrossEntropyLoss()    
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf