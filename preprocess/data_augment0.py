aws = False  # True

import os
import random
from datetime import datetime

if aws:
    dataset = '/home/ubuntu/dataset/stanford_mobile_images/'
else:
    dataset = '/home/payam/dataset/stanford_mobile_images/'


def prepare_augmented_images(conf):
    dataset_root = conf['dataset_root']
    result = {}
    for folder in os.listdir(dataset_root): # iterate over each set
        path0 = dataset_root + folder + '/'
        if os.path.isdir(path0):
            subres = {}
            for subfolder in os.listdir(path0):
                path = path0 + subfolder + '/'
                if os.path.isdir(path) and subfolder != 'Reference':
                    # test file
                    test = path0 + subfolder + '_test.txt'
                    with open(test, 'w') as out_test:
                        out_test.write(read_file_names(path))

                    # train file
                    train = path0 + subfolder + '_train.txt'
                    with open(train, 'w') as out_train:
                        for sf in os.listdir(path0):
                            p = path0 + sf + '/'
                            if os.path.isdir(p) and sf != subfolder:
                                out_train.write(read_file_names(p))
                    t = (train, test)
                    subres[subfolder] = t
        result[folder] = subres

    return result

def read_file_names(folder):
    res = ''
    for f in os.listdir(folder):
        file_name = folder + f
        if os.path.isfile(file_name) and file_name.endswith('.jpg'):
            res += '{0} {1}\n'.format(file_name, int(f.split('.')[0]) - 1)
    return res


def data_layer_formatter(folder, threshold=0):
    folder = dataset + folder
    train = '{0}/train{1}.txt'.format(folder, str(threshold).zfill(3))
    test = '{0}/test.txt'.format(folder)

    if os.path.isfile(train):
        os.remove(train)
    if os.path.isfile(test):
        os.remove(test)

    test_counter = 0
    train_counter = 0
    random.seed(datetime.now)

    for subfolder in os.listdir(folder):
        with open(train, 'a') as out_train:
            with open(test, 'a') as out_test:
                if os.path.isdir(folder + '/' + subfolder):
                    for f in os.listdir(folder + '/' + subfolder):
                        if os.path.isfile(folder + '/' + subfolder + '/' + f) and f.endswith('.jpg'):
                            if subfolder != 'Reference':
                                out_test.write(
                                    '{0}/{1}/{2} {3}\n'.format(folder, subfolder, f, int(f.split('.')[0]) - 1))
                                # in the format of: /path/to/file/0xx.jpg 0xx.jpg(class)
                                test_counter += 1
                                if random.random() * 100 < threshold:
                                    out_train.write(
                                        '{0}/{1}/{2} {3}\n'.format(folder, subfolder, f, int(f.split('.')[0]) - 1))
                            else:
                                out_train.write(
                                    '{0}/{1}/{2} {3}\n'.format(folder, subfolder, f, int(f.split('.')[0]) - 1))
                                train_counter += 1
    return [train, train_counter, test, test_counter]