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


def create_image_list_with_label(opt, train=False, valid=False):
    root = opt["datasets"]["path"]["root"]
    folders = opt["datasets"]["path"]["folders"]
    tumour_images_folder = os.path.join(root, folders["tumour_images"])
    normal_images_folder = os.path.join(root, folders["normal_images"])
    tumour_labels_folder = os.path.join(root, folders["tumour_labels"])
    if train:
        no_tumour_images = opt["datasets"]["train"]["no_of_images"]["tumour"]
        no_normal_images = opt["datasets"]["train"]["no_of_images"]["normal"]
    elif valid:
        no_tumour_images = opt["datasets"]["valid"]["no_of_images"]["tumour"]
        no_normal_images = opt["datasets"]["valid"]["no_of_images"]["normal"]

    tumour_images_dir = []
    for _, _, files in os.walk(tumour_images_folder):
        for file in files:
            if not file[0] == '.':
                tumour_images_dir.append(os.path.join(tumour_images_folder, file))

    normal_images_dir = []
    for _, _, files in os.walk(normal_images_folder):
        for file in files:
            if not file[0] == '.':
                normal_images_dir.append(os.path.join(normal_images_folder, file))

    tumour_labels_dir = []
    for _, _, files in os.walk(tumour_labels_folder):
        for file in files:
            if not file[0] == '.':
                tumour_labels_dir.append(os.path.join(tumour_labels_folder, file))

    if train:
        tumour_images_dir = tumour_images_dir[:no_tumour_images]
        normal_images_dir = normal_images_dir[:no_normal_images]
        tumour_labels_dir = tumour_labels_dir[:no_tumour_images]
    elif valid:
        tumour_images_dir = tumour_images_dir[-no_tumour_images:]
        normal_images_dir = normal_images_dir[-no_normal_images:]
        tumour_labels_dir = tumour_labels_dir[-no_tumour_images:]

    brain_tumour_list = []
    for tumour_no in range(len(tumour_images_dir)):
        brain_tumour_list.append(dict(image=nib.load(tumour_images_dir[tumour_no]),
                                      label=nib.load(tumour_labels_dir[tumour_no])))
        if tumour_no % 50 == 0:
            print(tumour_no)

    for normal_no in range(len(normal_images_dir)):
        brain_tumour_list.append(dict(image=nib.load(normal_images_dir[normal_no]),
                                      label=None))
        if normal_no % 50 == 0:
            print(normal_no)

    return brain_tumour_list