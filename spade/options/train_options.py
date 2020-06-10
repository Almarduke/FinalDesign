import argparse
from .process_option import process_option


def TrainOptions():
    parser = argparse.ArgumentParser(description='SPADE训练配置')

    # 训练相关的配置信息
    parser.add_argument('--gpu_ids', type=str, default='-1',
                        help='使用的GPU，0 0,1,2, 0,2. CPU为-1')
    parser.add_argument('--checkpoints_dir', type=str, default='spade/checkpoints',
                        help='模型保存的位置')
    parser.add_argument('--images_dir', type=str, default='spade/images',
                        help='图像保存的位置')
    parser.add_argument('--is_train', action='store_true', help='是否是训练')
    parser.add_argument('--current_epoch', type=int, default=0,
                        help='如果是继续训练，从哪个epoch开始，并根据编号加载数据')
    parser.add_argument('--save_log', action='store_true', help='是否保存损失值的变化情况')
    parser.add_argument('--dataset', type=str, default='ade20k',
                        help='数据集的名称, "_"之后的是备注')

    # 模型相关的配置信息
    parser.add_argument('--use_vae', type=bool, default=True, help='使用image encoder参与训练')
    parser.add_argument('--D_model_num', type=int, default=2,
                        help='用于multiscale的discriminator的数量')
    parser.add_argument('--lambda_kld', type=float, default=0.05,
                        help='KLD Loss的权重')
    parser.add_argument('--lambda_vgg', type=float, default=10.0,
                        help='Perceptual Loss的权重')
    parser.add_argument('--lambda_feature', type=float, default=10.0,
                        help='GAN Feature matching Loss的权重')
    parser.add_argument('--z_dim', type=int, default=256,
                        help='输入噪声的维度')
    parser.add_argument('--init_variance', type=float, default=0.02,
                        help='网络初始化的方差')

    # 数据集数据预处理的配置信息
    parser.add_argument('--dataset_dir', type=str, default='spade/dataset/ADEChallengeData2016',
                        help='数据集所在的根目录')
    parser.add_argument('--flip', type=bool, default=True,
                        help='is_train=True & flip=True时，训练集中的图像有一半概率翻转，数据增强')
    parser.add_argument('--pairing_check', action='store_true',
                        help='检查输入图像和label是否匹配')
    parser.add_argument('--img_mean', type=float, default=0.5,
                        help='输入的图像各个像素点的均值')
    parser.add_argument('--img_var', type=float, default=0.5,
                        help='输入的图像各个像素点的方差')
    parser.add_argument('--preprocess_mode', type=str, default='scale_and_crop',
                        help='图像预处理，"resize"（直接resize会变形）/"scale_and_crop"（把图像缩放到比loadsize大后裁剪）')

    # 数据集和训练相关的配置信息
    parser.add_argument('--batch_size', type=int, default=4,
                        help='训练时batch的大小')
    parser.add_argument('--load_size', type=tuple, default=(256, 256),
                        help='预处理后图像的目标size')
    parser.add_argument('--n_label', type=int, default=150,
                        help='标签数，不包括无法识别的标签. 参见contain_dontcare_label.')
    parser.add_argument('--contain_dontcare_label', type=bool, default=True,
                        help='无法识别的标签')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='输出图像的通道数')
    parser.add_argument('--total_epochs', type=int, default=200,
                        help='总共有多少个epoch（包括learning rate decay的周期）')
    parser.add_argument('--decay_epochs', type=int, default=100,
                        help='learning rate decay的周期数，10表示最后10个周期内发生decay')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='初始的学习率')
    parser.add_argument('--TTUR', type=bool, default=True,
                        help='Two Time-Scale Update Rule')
    parser.add_argument('--beta1', type=float, default=0,
                        help='adam中的动量参数beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='adam中的动量参数beta2')

    opt = parser.parse_args()
    return process_option(opt)
