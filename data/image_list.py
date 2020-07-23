import os
import nibabel as nib

'''
Currently, it's only for reading images regardless of whether there is a label or not
TO DO: incorporate label
'''

def create_image_list(opt, train=False, valid=False):
    root = opt["datasets"]["path"]["root"]
    folders = opt["datasets"]["path"]["folders"]
    root_folders = [os.path.join(root, folder) for folder in folders]
    image_dir = []
    for folder in root_folders:
        for _, _, files in os.walk(folder):
            for file in files:
                if not file[0] == '.':
                    image_dir.append(os.path.join(folder, file))
    if train is True:
        image_dir = image_dir[:opt["datasets"]["train"]["no_of_images"]]
    elif valid is True:
        image_dir = image_dir[-opt["datasets"]["valid"]["no_of_images"]:]
    else:
        raise NotImplementedError("Neither training or validating")

    brain_tumour_list = []
    for index, image in enumerate(image_dir):
        brain_tumour_list.append(nib.load(image))
        if index%50==0:
            print("{}/{} images loaded".format(index, len(image_dir)))
    return brain_tumour_list
