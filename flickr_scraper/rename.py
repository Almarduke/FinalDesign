import os
from shutil import copyfile

def load_files(root_dir):
    assert os.path.isdir(root_dir) or os.path.islink(root_dir), \
        f'"{root_dir}" is not a valid directory'
    file_paths = []
    for dir_path, dir_names, file_names in os.walk(root_dir):
        for file_name in file_names:
            filepath = os.path.join(dir_path, file_name)
            file_paths.append(filepath)
    return file_paths


root_dir = './download/images'

file_paths = load_files(root_dir)
dst_dir = './download/renamed'

for index, filepath in enumerate(file_paths):
    index = (5 - len(str(index))) * '0' + str(index)
    copyfile(filepath, f'{dst_dir}/{index}.jpg')

