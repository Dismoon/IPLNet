# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
import h5py
import math
import glob
import os


def psnr(img1, img2):
    return tf.reduce_mean(tf.image.psnr(img1, img2, 1.0))


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def prepare_data(config):

    if config.is_train:

        input_dir = os.path.join(os.path.join(os.getcwd(), config.train_set_input))
        input_list = glob.glob(os.path.join(input_dir, "*png"))
        input_list.sort()
        label_dir = os.path.join(os.path.join(os.getcwd(), config.train_set_label))
        label_list = glob.glob(os.path.join(label_dir, "*png"))
        label_list.sort()

        eval_input_dir = os.path.join(os.path.join(os.getcwd(), config.eval_set_input))
        eval_input_list = glob.glob(os.path.join(eval_input_dir, "*png"))
        eval_input_list.sort()
        eval_label_dir = os.path.join(os.path.join(os.getcwd(), config.eval_set_label))
        eval_label_list = glob.glob(os.path.join(eval_label_dir, "*png"))
        eval_label_list.sort()
        return input_list, label_list, eval_input_list, eval_label_list

    else:

        test_dir = os.path.join(os.getcwd(), config.test_set)
        test_list = glob.glob(os.path.join(test_dir, "*.png"))

        return test_list


def input_setup(config):

    input_list, label_list, eval_input_list, eval_label_list = prepare_data(config)
    print('Prepare training data...')
    make_sub_data(input_list, label_list, config, 'train')
    print('Prepare evaluating data...')
    make_sub_data(eval_input_list, eval_label_list, config, 'eval')


def make_data_hf(input_, label_, config, str, times):

    assert input_.shape == label_.shape
    if not os.path.isdir(os.path.join(os.getcwd(), config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), "checkpoint"))
    if str == 'train':
        savepath = os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'train.h5')
    elif str == 'eval':
        savepath = os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'eval.h5')

    else:
        savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'test.h5')

    if times == 0:  
        if os.path.exists(savepath):
            print("\n%s have existed!\n" % (savepath))
            return False
        else:
            hf = h5py.File(savepath, 'w')
            if config.is_train:
                input_h5 = hf.create_dataset("input", (1, config.image_size, config.image_size, config.c_dim),
                                             maxshape=(None, config.image_size, config.image_size, config.c_dim),
                                             chunks=(1, config.image_size, config.image_size, config.c_dim),
                                             dtype='float32')

                label_h5 = hf.create_dataset("label", (1, config.image_size, config.image_size, config.c_dim),
                                             maxshape=(None, config.image_size, config.image_size, config.c_dim),
                                             chunks=(1, config.image_size, config.image_size, config.c_dim),
                                             dtype='float32')


            else:
                input_h5 = hf.create_dataset("input", (1, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             maxshape=(None, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             chunks=(1, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             dtype='float32')
                label_h5 = hf.create_dataset("label", (1, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             maxshape=(None, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             chunks=(1, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             dtype='float32')
    else:  
        hf = h5py.File(savepath, 'a')
        input_h5 = hf["input"]
        label_h5 = hf["label"]


    if config.is_train:
        input_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim])
        input_h5[times: times + 1] = input_
        label_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim])
        label_h5[times: times + 1] = label_

    else:
        input_h5.resize([times + 1, input_.shape[0], input_.shape[1], input_.shape[2]])
        input_h5[times: times + 1] = input_
        label_h5.resize([times + 1, label_.shape[0], label_.shape[1], label_.shape[2]])
        label_h5[times: times + 1] = label_

    hf.close()
    return True


def make_sub_data(input_list, label_list, config, str):


    assert len(input_list) == len(label_list)
    times = 0  
    for i in range(len(input_list)):
        name =  os.path.basename(input_list[i])
        ratio = float(name[4:6])
        input_ = cv2.imread(input_list[i], -1)
        input_ = input_ * ratio
        label_ = cv2.imread(label_list[i], -1)

        # print(label.shape)
        assert input_.shape == label_.shape

        if len(input_.shape) == 3:
            h, w, c = input_.shape
        else:
            h, w = input_.shape

        if not config.is_train:
            input_ = input_ / 255.0
            label_ = label_ / 255.0

            make_data_hf(input_, label_, config, times)
            return input_list, label_list

        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):
                x_ = int(x / 2)
                y_ = int(y / 2)
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]
                sub_input = sub_input / 255.0  
                sub_label = label_[x:x + config.image_size, y:y + config.image_size]
                sub_label = sub_label / 255.0 

                save_flag = make_data_hf(sub_input, sub_label, config, str, times)
                if not save_flag:
                    return input_list, label_list
                times += 1
        print("image: [%2d], total: [%2d]" % (i, len(input_list)))
    return input_list, label_list



def get_data_num(path):
    with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        return input_.shape[0]



def get_data_dir(checkpoint_dir, is_train):
    if is_train:
        return os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'train.h5'), os.path.join(
            os.path.join(os.getcwd(), "checkpoint"), 'eval.h5')

    else:
        return os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'test.h5')


def get_batch(path, data_num, batch_size):
    with h5py.File(path, 'r')as hf:
        input_ = hf["input"]
        label_ = hf["label"]
        random_batch = np.random.rand(batch_size) * (data_num - 1)

        batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
        batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])

        for i in range(batch_size):
            batch_images[i, :, :, :] = np.asarray(input_[int(random_batch[i])])
            batch_labels[i, :, :, :] = np.asarray(label_[int(random_batch[i])])

        random_aug = np.random.rand(2)  
        batch_images = augmentation(batch_images, random_aug)
        batch_labels = augmentation(batch_labels, random_aug)
        return batch_images, batch_labels


def augmentation(batch, random):
    if random[0] < 0.3:

        batch_flip = np.flip(batch, 1)
    elif random[0] > 0.7:

        batch_flip = np.flip(batch, 2)
    else:

        batch_flip = batch

    if random[1] < 0.5:

        batch_rot = np.rot90(batch_flip, 1, [1, 2])
    else:

        batch_rot = batch_flip

    return batch_rot



def show_img(img, k):
    if k == 'm':
        img = img
        cv2.imshow("img_merge", img)
        cv2.waitKey(0)
    if k == '0':
        img = img[:, :, 0]
        cv2.imshow("img_0", img)
        cv2.waitKey(0)
    if k == '45':
        img = img[:, :, 1]
        cv2.imshow("img_45", img)
        cv2.waitKey(0)
    if k == '90':
        img = img[:, :, 2]
        cv2.imshow("img_90", img)
        cv2.waitKey(0)
    if k == '135':
        img = img[:, :, 3]
        cv2.imshow("img_135", img)
        cv2.waitKey(0)


def imsave(image, path):
    cv2.imwrite(os.path.join(os.getcwd(), path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def save_rgb(config, img, output_name):
    path_c4 = os.path.join(config.output_dir, "rgb-c4")
    if not os.path.isdir(path_c4):
        os.mkdir(path_c4)
    imsave(img * 255, path_c4 + '/%s-c4.png' % output_name)


def save_r(config, img, output_name):
    dofp = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]
    img135 = img[:, :, 3]
    img0 = img0.astype(float)
    img45 = img45.astype(float)
    img90 = img90.astype(float)
    img135 = img135.astype(float)

    dofp[0:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img90
    dofp[1:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img135
    dofp[0:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img45
    dofp[1:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img0

    path_r = os.path.join(config.output_dir, "r")
    if not os.path.isdir(path_r):
        os.mkdir(path_r)
    imsave(dofp, path_r + '/%s.png' % output_name)


def save_rr(config, img, output_name):
    dofp = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    dolp = img[:, :, 0]
    aop = img[:, :, 1]

    path_r = os.path.join(config.output_dir, "r")
    if not os.path.isdir(path_r):
        os.mkdir(path_r)
    imsave(dolp, path_r + '/%s-d.png' % output_name)
    imsave(aop, path_r + '/%s-a.png' % output_name)


def save_gg(config, img, output_name):
    dofp = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    dolp = img[:, :, 0]
    aop = img[:, :, 1]

    path_g = os.path.join(config.output_dir, "g")
    if not os.path.isdir(path_g):
        os.mkdir(path_g)
    imsave(dolp, path_g + '/%s-d.png' % output_name)
    imsave(aop, path_g + '/%s-a.png' % output_name)


def save_bb(config, img, output_name):
    dofp = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    dolp = img[:, :, 0]
    aop = img[:, :, 1]

    path_b = os.path.join(config.output_dir, "b")
    if not os.path.isdir(path_b):
        os.mkdir(path_b)
    imsave(dolp, path_b + '/%s-d.png' % output_name)
    imsave(aop, path_b + '/%s-a.png' % output_name)


def save_g(config, img, output_name):
    dofp = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]
    img135 = img[:, :, 3]
    img0 = img0.astype(float)
    img45 = img45.astype(float)
    img90 = img90.astype(float)
    img135 = img135.astype(float)

    dofp[0:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img90
    dofp[1:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img135
    dofp[0:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img45
    dofp[1:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img0

    path_g = os.path.join(config.output_dir, "g")
    if not os.path.isdir(path_g):
        os.mkdir(path_g)
    imsave(dofp, path_g + '/%s.png' % output_name)


def save_b(config, img, output_name):
    dofp = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]
    img135 = img[:, :, 3]
    img0 = img0.astype(float)
    img45 = img45.astype(float)
    img90 = img90.astype(float)
    img135 = img135.astype(float)

    dofp[0:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img90
    dofp[1:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img135
    dofp[0:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img45
    dofp[1:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img0

    path_b = os.path.join(config.output_dir, "b")
    if not os.path.isdir(path_b):
        os.mkdir(path_b)
    imsave(dofp, path_b + '/%s.png' % output_name)


def save_image(config, img, output_name):
    dofp = np.zeros((img.shape[0] * 2, img.shape[1] * 2))

    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]
    img135 = img[:, :, 3]
    img0 = img0.astype(float)
    img45 = img45.astype(float)
    img90 = img90.astype(float)
    img135 = img135.astype(float)

    s0 = (img0 + img45 + img90 + img135) / 2.0
    s1 = img0 - img90
    s2 = img45 - img135

    aop = 1 / 2 * np.arctan2(s2, s1)
    aop = aop + math.pi / 4.0  
    aop = np.clip(aop, 0, math.pi / 2.0)
    aop = aop / (math.pi / 2.0)  
    aop = aop * 255  

    dolp = (np.power(np.power(s1, 2) + np.power(s2, 2), 0.5) / (s0 + 0.00001))
    dolp = np.clip(dolp * 255, 0, 255)

    dofp[0:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img90
    dofp[1:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img135
    dofp[0:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img45
    dofp[1:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img0

    path_f = os.path.join(config.output_dir, "full_size")
    if not os.path.isdir(path_f):
        os.mkdir(path_f)
    path_d = os.path.join(config.output_dir, "dolp")
    if not os.path.isdir(path_d):
        os.mkdir(path_d)
    path_a = os.path.join(config.output_dir, "aop")
    if not os.path.isdir(path_a):
        os.mkdir(path_a)
    imsave(aop, path_a + '/%s-a.png' % output_name)
    imsave(dolp, path_d + '/%s-d.png' % output_name)

    imsave(dofp, path_f + '/%s-f.png' % output_name)
