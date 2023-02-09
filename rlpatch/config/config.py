import torch
from PIL import Image

class Config(object):
    "-----MI parameters------"
    # x,y = 50,36
    x,y = 30,21
    model_names = ['cosface50','arcface34','arcface50']
    model_names = ['arcface34','arcface50']
    # model_names = ['facenet']
    threat_name = 'facenet'
    num_classes = 5752
    sticker = Image.new('RGBA',(30,40),(255,255,255,255))
    # label = 5748
    target = 3820#4863
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    display = False

    # width, height = 112, 112
    width, height = 160, 160
    emp_iterations = 100
    sapce_thd = 40
    di = False
    adv_img_folder = 'res'
    dataset_folder = '../lfw'
    targeted = False #
    "------RL parameters-------"
    
    "-----extra parameters-----"
    # input_dir = '/home/guoying/rlpatch/example'
    # batch_size = 10
    
    
    
    
    # env = 'default'
    # backbone =  'SERes50_IR'   #'Sphere20'  SERes50_IR
    # classify = 'softmax'
    # num_classes =  5754 # 16326 10575 8192
    # metric = 'InnerProduct' #'arc_margin' 'sphere2'
    # easy_margin = False
    # use_se = False
    # loss = 'focal_loss' #'focal_loss'

    # display = False
    # finetune = False

    # train_root = '/data/Datasets/webface/CASIA-maxpy-clean-crop-144/'
    # train_list = '/data/Datasets/webface/train_data_13938.txt'
    # val_list = '/data/Datasets/webface/val_data_13938.txt'

    # test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    # test_list = 'test.txt'

    # lfw_root = '/home/guoying/arcface-pytorch-master/data/lfw-align-128'
    # lfw_test_list = '/home/guoying/arcface-pytorch-master/lfw_test_pair.txt'
    # #lfw_test_list = '/home/ubuntu/Documents/RY2020/arcface/lfw_test_pair.txt'

    # checkpoints_path = '/home/guoying/patch/train_model/material/checkpoints3'
    # load_model_path = 'models/resnet18.pth'
    # test_model_path =  '/home/ubuntu/Documents/RY2020/arcface/checkpoints2/10SERes50_IR_69.pth' #'./checkpoints/resnet18_110.pth'
    # save_interval = 5

    # train_batch_size = 64  # batch size
    # test_batch_size = 96

    # #input_shape = (1, 128, 128)
    # input_shape = (112,112,3)

    # optimizer = 'sgd'

    # use_gpu = True  # use GPU or not
    # gpu_id = '0, 1'
    # num_workers = 4  # how many workers for loading data
    # print_freq = 5  # print info every N batch

    # debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    # result_file = 'result.csv'

    # max_epoch = 50
    # lr = 0.1  # initial learning rate
    # lr_step = 10
    # lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    # weight_decay = 5e-4
