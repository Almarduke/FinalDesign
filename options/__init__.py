"""
Copyright (C 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
"""

BASE_OPTION = {
    # experiment specifics
    'experience': 'SPADE',  # 实验的名称
    'gpu_ids': '0',  # 使用的GPU，0 0,1,2, 0,2. CPU为-1
    'checkpoints_dir': './checkpoints',  # 模型保存的位置
    'model': 'pix2pix',  # 使用的模型的名称
    'is_train': True,  # 是否是训练
    'continue_train': True,  # 继续训练，是则加载数据
    'save_log': True,  # 是否保存损失值的变化情况
    # 'phase': 'train',  # 读取的数据集是哪个阶段，train, val, test
}

ADE20K_DATASET_OPTION = {
    # 和数据输入部分相关
    'dataroot': './dataset/ADEChallengeData2016/',  # 数据集所在的根目录
    'dataset': 'ade20k',  # 数据集的名称
    'shuffle': True,  # 是否打乱数据集
    'flip': True,  # phase=True & flip=True时，训练集中的图像有一半概率翻转，数据增强
    'thread_num': 0,  # dataloader用来读取数据集时的线程数，默认值为0
    'no_instance': True,  # 有无instance_map

    # 和训练相关
    'batch_size': 32,  # 'input batch size'
    'preprocess_mode': 'scale_and_crop',  # 图像预处理，"resize"（直接resize会变形）/"scale_and_crop"（把图像缩放到比loadsize大后裁剪）
    'load_size': (256, 256),  # 预处理后图像的size
    'label_nc': 150,  # 标签数，包括无法识别的标签. 参见contain_dontcare_label.'
    'contain_dontcare_label': True,  # 无法识别的标签，对应255
    'output_nc': 3,  # 输出图像的通道数
    'total_epochs': 200,  # 总共有多少个epoch（包括learning rate decay的周期）
    'decay_epochs': 10,  # learning rate decay的周期数
}

# parser.add_argument('--name': 'label2coco',
#                      # 'name of the experiment. It decides where to store samples and models'
# parser.add_argument('--gpu_ids': '0',  # 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
# parser.add_argument('--checkpoints_dir': './checkpoints',  # 'models are saved here'
# parser.add_argument('--model': 'pix2pix',  # 'which model to use'
# parser.add_argument('--norm_G': 'spectralinstance',
#                      # 'instance normalization or batch normalization'
# parser.add_argument('--norm_D': 'spectralinstance',
#                      # 'instance normalization or batch normalization'
# parser.add_argument('--norm_E': 'spectralinstance',
#                      # 'instance normalization or batch normalization'
# parser.add_argument('--phase': 'train',  # 'train, val, test, etc'
#
# # input/output sizes
# parser.add_argument('--batchSize': 1,  # 'input batch size'
# parser.add_argument('--preprocess_mode': 'scale_width_and_crop',
#                      # 'scaling and cropping of images at load time.', choices=(
#     "resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop",
#     "fixed", "none"
# parser.add_argument('--load_size': 1024,
#                      # 'Scale images to this size. The final image will be cropped to --crop_size.'
# parser.add_argument('--crop_size': 512,
#                      # 'Crop to the width of crop_size (after initially scaling the images to load_size.'
# parser.add_argument('--aspect_ratio', type=float, default=1.0,
#                      # 'The ratio width/height. The final height of the load image will be crop_size/aspect_ratio'
# parser.add_argument('--label_nc': 182,
#                      # '# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.'
# parser.add_argument('--contain_dontcare_label', action='store_true',
#                      # 'if the label map contains dontcare label (dontcare=255'
# parser.add_argument('--output_nc': 3,  # '# of output image channels'
#
# # for setting inputs
# parser.add_argument('--dataroot': './datasets/cityscapes/'
# parser.add_argument('--dataset_mode': 'coco'
# parser.add_argument('--serial_batches', action='store_true',
#                      # 'if true, takes images in order to make batches, otherwise takes them randomly'
# parser.add_argument('--no_flip', action='store_true',
#                      # 'if specified, do not flip the images for data argumentation'
# parser.add_argument('--nThreads', default=0, type=int,  # '# threads for loading data'
# parser.add_argument('--max_dataset_size': sys.maxsize,
#                      # 'Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.'
# parser.add_argument('--load_from_opt_file', action='store_true',
#                      # 'load the options from checkpoints and use that as default'
# parser.add_argument('--cache_filelist_write', action='store_true',
#                      # 'saves the current filelist into a text file, so that it loads faster'
# parser.add_argument('--cache_filelist_read', action='store_true',  # 'reads from the file list cache'
#
# # for displays
# parser.add_argument('--display_winsize': 400,  # 'display window size'
#
# # for generator
# parser.add_argument('--netG': 'spade',  # 'selects model to use for netG (pix2pixhd | spade'
# parser.add_argument('--ngf': 64,  # '# of gen filters in first conv layer'
# parser.add_argument('--init_type': 'xavier',
#                      # 'network initialization [normal|xavier|kaiming|orthogonal]'
# parser.add_argument('--init_variance', type=float, default=0.02,  # 'variance of the initialization distribution'
# parser.add_argument('--z_dim': 256,
#                      # "dimension of the latent z vector"
#
# # for instance-wise features
# parser.add_argument('--no_instance', action='store_true',  # 'if specified, do *not* add instance map as input'
# parser.add_argument('--nef': 16,  # '# of encoder filters in the first conv layer'
# parser.add_argument('--use_vae', action='store_true',  # 'enable training with an image encoder.'





        #
        # parser.add_argument('--display_freq': 100,  # 'frequency of showing training results on screen'
        # parser.add_argument('--print_freq': 100,  # 'frequency of showing training results on console'
        # parser.add_argument('--save_latest_freq': 5000,  # 'frequency of saving the latest results'
        # parser.add_argument('--save_epoch_freq': 10,  # 'frequency of saving checkpoints at the end of epochs'
        # parser.add_argument('--no_html', action='store_true',  # 'do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/'
        # parser.add_argument('--debug', action='store_true',  # 'only do one epoch and displays at each iteration'
        # parser.add_argument('--tf_log', action='store_true',  # 'if specified, use tensorboard logging. Requires tensorflow installed'
        #
        # # for training
        # parser.add_argument('--continue_train', action='store_true',  # 'continue training: load the latest model'
        # parser.add_argument('--which_epoch': 'latest',  # 'which epoch to load? set to latest to use latest cached model'
        # parser.add_argument('--niter': 50,  # '# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay'
        # parser.add_argument('--niter_decay': 0,  # '# of iter to linearly decay learning rate to zero'
        # parser.add_argument('--optimizer': 'adam'
        # parser.add_argument('--beta1', type=float, default=0.0,  # 'momentum term of adam'
        # parser.add_argument('--beta2', type=float, default=0.9,  # 'momentum term of adam'
        # parser.add_argument('--no_TTUR', action='store_true',  # 'Use TTUR training scheme'
        #
        # # the default values for beta1 and beta2 differ by TTUR option
        # opt, _ = parser.parse_known_args(
        # if opt.no_TTUR:
        #     parser.set_defaults(beta1=0.5, beta2=0.999
        #
        # parser.add_argument('--lr', type=float, default=0.0002,  # 'initial learning rate for adam'
        # parser.add_argument('--D_steps_per_G': 1,  # 'number of discriminator iterations per generator iterations.'
        #
        # # for discriminators
        # parser.add_argument('--ndf': 64,  # '# of discrim filters in first conv layer'
        # parser.add_argument('--lambda_feat', type=float, default=10.0,  # 'weight for feature matching loss'
        # parser.add_argument('--lambda_vgg', type=float, default=10.0,  # 'weight for vgg loss'
        # parser.add_argument('--no_ganFeat_loss', action='store_true',  # 'if specified, do *not* use discriminator feature matching loss'
        # parser.add_argument('--no_vgg_loss', action='store_true',  # 'if specified, do *not* use VGG feature matching loss'
        # parser.add_argument('--gan_mode': 'hinge',  # '(ls|original|hinge'
        # parser.add_argument('--netD': 'multiscale',  # '(n_layers|multiscale|image'
        # parser.add_argument('--lambda_kld', type=float, default=0.05


# BaseOptions.initialize(self, parser
# parser.add_argument('--results_dir': './results/',  # 'saves results here.'
# parser.add_argument('--which_epoch': 'latest',
#                      # 'which epoch to load? set to latest to use latest cached model'
# parser.add_argument('--how_many': float("inf",  # 'how many test images to run'
#
# parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256
# parser.set_defaults(serial_batches=True
# parser.set_defaults(no_flip=True
# parser.set_defaults(phase='test'
# self.isTrain = False