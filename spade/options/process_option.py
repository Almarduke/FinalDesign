import os

def process_option(opt):
    dir_names = ['checkpoints_dir', 'images_dir', 'dataset_dir']
    for dir_name in dir_names:
        if hasattr(opt, dir_name):
            dir_path = getattr(opt, dir_name)
            setattr(opt, dir_name, os.path.join(os.getcwd(), dir_path))

    # 语义标签，label + 无法识别
    opt.n_semantic = opt.n_label
    if opt.contain_dontcare_label:
        opt.n_semantic += 1

    return opt