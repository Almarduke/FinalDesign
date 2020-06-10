import argparse
from .process_option import process_option


def TestOptions():
    parser = argparse.ArgumentParser(description='SPADE测试配置')

    # 基本配置信息
    parser.add_argument('--gpu_ids', type=str, default='-1',
                        help='使用的GPU，0 0,1,2, 0,2. CPU为-1')
    parser.add_argument('--label_path', type=str, default=None,
                        help='语义分割图文件路径 (默认使用彩色的语义分割图)')
    parser.add_argument('--style_path', type=str, default=None,
                        help='风格图像文件路径')
    parser.add_argument('--result_path', type=str, default=None,
                        help='生成结果保存路径')
    parser.add_argument('--use_greylabel', action='store_true',
                        help='是否使用原始的灰度语义分割图, 默认使用彩色的语义分割图')

    # 读取模型文件的配置信息
    parser.add_argument('--is_train', action='store_true', help='是否是训练')
    parser.add_argument('--checkpoints_dir', type=str, default='spade/checkpoints',
                        help='模型保存的位置')
    parser.add_argument('--current_epoch', type=int, default=100,
                        help='如果是继续训练，从哪个epoch开始，并根据编号加载数据')
    parser.add_argument('--dataset', type=str, default='natural',
                        help='数据集的名称, "_"之后的是备注')

    # 模型结构的配置信息
    parser.add_argument('--use_vae', type=bool, default=True, help='使用image encoder参与训练')
    parser.add_argument('--z_dim', type=int, default=256,
                        help='输入噪声的维度')
    parser.add_argument('--init_variance', type=float, default=0.02,
                        help='网络初始化的方差')

    # 数据处理相关的配置信息
    parser.add_argument('--img_mean', type=float, default=0.5,
                        help='输入的图像各个像素点的均值')
    parser.add_argument('--img_var', type=float, default=0.5,
                        help='输入的图像各个像素点的方差')
    parser.add_argument('--load_size', type=tuple, default=(512, 512),
                        help='预处理后图像的目标size')
    parser.add_argument('--n_label', type=int, default=12,
                        help='标签数，不包括无法识别的标签. 参见contain_dontcare_label.')
    parser.add_argument('--contain_dontcare_label', type=bool, default=True,
                        help='无法识别的标签')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='输出图像的通道数')

    opt = parser.parse_args()
    return process_option(opt)
