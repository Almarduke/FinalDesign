
BASE_OPTION = {
    'experience': 'SPADE',  # 实验的名称
    'gpu_ids': [0, 1, 2, 3],  # 使用的GPU，0 0,1,2, 0,2. CPU为-1
    'checkpoints_dir': 'checkpoints',  # 模型保存的位置
    'images_dir': 'images',  # 图像保存的位置
    'model': 'pix2pix',  # 使用的模型的名称
    'is_train': True,  # 是否是训练
    'continue_train': True,  # 继续训练，是则加载数据
    'current_epoch': 0,  # 如果是继续训练，从哪个epoch开始
    'save_log': True,  # 是否保存损失值的变化情况
    'dataset': 'ade20k',  # 数据集的名称
}

MODEL_OPTION = {
    'use_vae': True,  # 使用image encoder参与训练
    'D_model_num': 2,  # 用于multiscale的discriminator的数量
    'lambda_kld': 0.05,  # kld loss的权重
    'lambda_vgg': 10.0,  # perceptual loss的权重
    'z_dim': 256,  # 输入噪声的维度
    'init_variance': 0.02,  # 网络初始化的方差
}

ADE20K_DATASET_OPTION = {
    # 和数据输入部分相关
    'dataset_dir': 'dataset/ADEChallengeData2016/',  # 数据集所在的根目录
    'shuffle': True,  # 是否打乱数据集
    'flip': True,  # is_train=True & flip=True时，训练集中的图像有一半概率翻转，数据增强
    'thread_num': 0,  # dataloader用来读取数据集时的线程数，默认值为0
    'no_instance': True,  # 有无instance_map
    'pairing_check': True,  # 检查输入图像和label是否匹配
    'img_mean': 0.5,  # 输入的图像各个像素点的均值
    'img_var': 0.5,  # 输入的图像各个像素点的方差

    # 和训练相关
    'batch_size': 16,  # 'input batch size'
    'preprocess_mode': 'scale_and_crop',  # 图像预处理，"resize"（直接resize会变形）/"scale_and_crop"（把图像缩放到比loadsize大后裁剪）
    'load_size': (256, 256),  # 预处理后图像的目标size
    'n_label': 150,  # 标签数，包括无法识别的标签. 参见contain_dontcare_label.'
    'contain_dontcare_label': True,  # 无法识别的标签，对应255
    'output_nc': 3,  # 输出图像的通道数
    'total_epochs': 200,  # 总共有多少个epoch（包括learning rate decay的周期）
    'decay_epochs': 100,  # learning rate decay的周期数，10表示最后10个周期内发生decay
    'learning_rate': 0.0002,  # 初始的学习率
    'TTUR': True,  # Two Time-Scale Update Rule
    'beta1': 0.5,  # adam中的动量参数
    'beta2': 0.999,  # adam中的动量参数
}
