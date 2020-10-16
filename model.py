# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import os
from utilsrgb import *


class RDN(object):

    def __init__(self,
                 sess,  
                 is_train,
                 is_eval,  
                 image_size, 
                 c_dim,  
                 batch_size,
                 D,  
                 C,  
                 G,
                 P_dim,  
                 Pc_dim,  
                 PD,  
                 PC,  
                 PG,
                 kernel_size  
                 ):

        self.sess = sess
        self.is_train = is_train
        self.is_eval = is_eval
        self.image_size = image_size
        self.c_dim = c_dim
        self.P_dim = P_dim
        self.Pc_dim = Pc_dim
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.PD = PD
        self.PC = PC
        self.PG = PG
        self.kernel_size = kernel_size

    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer
        for i in range(1, self.D + 1):
            x = rdb_in
            for j in range(1, self.C + 1):
                tmp = slim.conv2d(x, self.G, [3, 3], rate=1, activation_fn=lrelu,
                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                x = tf.concat([x, tmp], axis=3)

            # local feature fusion
            x = slim.conv2d(x, self.G, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            rdb_in = tf.add(x, rdb_in)
            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3)

    def R_RDBs(self, input_layer):
        r_rdb_concat = list()
        r_rdb_in = input_layer
        for i in range(1, self.PD + 1):
            r_x = r_rdb_in
            for j in range(1, self.PC + 1):
                r_tmp = slim.conv2d(r_x, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                r_x = tf.concat([r_x, r_tmp], axis=3)

            # local feature fusion
            r_x = slim.conv2d(r_x, self.PG, [1, 1], rate=1, activation_fn=None,
                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            r_rdb_in = tf.add(r_x, r_rdb_in)
            r_rdb_concat.append(r_rdb_in)
        return tf.concat(r_rdb_concat, axis=3)

    def G_RDBs(self, input_layer):
        g_rdb_concat = list()
        g_rdb_in = input_layer
        for i in range(1, self.PD + 1):
            g_x = g_rdb_in
            for j in range(1, self.PC + 1):
                g_tmp = slim.conv2d(g_x, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                g_x = tf.concat([g_x, g_tmp], axis=3)

            # local feature fusion
            g_x = slim.conv2d(g_x, self.PG, [1, 1], rate=1, activation_fn=None,
                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            g_rdb_in = tf.add(g_x, g_rdb_in)
            g_rdb_concat.append(g_rdb_in)
        return tf.concat(g_rdb_concat, axis=3)

    def B_RDBs(self, input_layer):
        b_rdb_concat = list()
        b_rdb_in = input_layer
        for i in range(1, self.PD + 1):
            b_x = b_rdb_in
            for j in range(1, self.PC + 1):
                b_tmp = slim.conv2d(b_x, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                b_x = tf.concat([b_x, b_tmp], axis=3)

            # local feature fusion
            b_x = slim.conv2d(b_x, self.PG, [1, 1], rate=1, activation_fn=None,
                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            b_rdb_in = tf.add(b_x, b_rdb_in)
            b_rdb_concat.append(b_rdb_in)
        return tf.concat(b_rdb_concat, axis=3)

    def model(self):
        F_1 = slim.conv2d(self.images, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0 = slim.conv2d(F_1, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD = self.RDBs(F0)

        FGF1 = slim.conv2d(FD, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2 = slim.conv2d(FGF1, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FDF = tf.add(FGF2, F_1)  
        IHR = slim.conv2d(FDF, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        PSNR = psnr(IHR, self.labels)

        Pred_B = IHR[:, :, :, 0]
        Pred_B = Pred_B[:, :, :, np.newaxis]
        Pred_G = IHR[:, :, :, 1]
        Pred_G = Pred_G[:, :, :, np.newaxis]
        Pred_R = IHR[:, :, :, 2]
        Pred_R = Pred_R[:, :, :, np.newaxis]

        GT_B = self.labels[:, :, :, 0]
        GT_B = GT_B[:, :, :, np.newaxis]
        GT_G = self.labels[:, :, :, 1]
        GT_G = GT_G[:, :, :, np.newaxis]
        GT_R = self.labels[:, :, :, 2]
        GT_R = GT_R[:, :, :, np.newaxis]

        GT_B90 = GT_B[:, 0:GT_B.shape[1]:2, 0:GT_B.shape[2]:2]
        GT_B135 = GT_B[:, 1:GT_B.shape[1]:2, 0:GT_B.shape[2]:2]
        GT_B45 = GT_B[:, 0:GT_B.shape[1]:2, 1:GT_B.shape[2]:2]
        GT_B0 = GT_B[:, 1:GT_B.shape[1]:2, 1:GT_B.shape[2]:2]
        GT_BM = tf.concat([GT_B0, GT_B45, GT_B90, GT_B135], axis=-1)

        GT_G90 = GT_G[:, 0:GT_G.shape[1]:2, 0:GT_G.shape[2]:2]
        GT_G135 = GT_G[:, 1:GT_G.shape[1]:2, 0:GT_G.shape[2]:2]
        GT_G45 = GT_G[:, 0:GT_G.shape[1]:2, 1:GT_G.shape[2]:2]
        GT_G0 = GT_G[:, 1:GT_G.shape[1]:2, 1:GT_G.shape[2]:2]
        GT_GM = tf.concat([GT_G0, GT_G45, GT_G90, GT_G135], axis=-1)

        GT_R90 = GT_R[:, 0:GT_R.shape[1]:2, 0:GT_R.shape[2]:2]
        GT_R135 = GT_R[:, 1:GT_R.shape[1]:2, 0:GT_R.shape[2]:2]
        GT_R45 = GT_R[:, 0:GT_R.shape[1]:2, 1:GT_R.shape[2]:2]
        GT_R0 = GT_R[:, 1:GT_R.shape[1]:2, 1:GT_R.shape[2]:2]
        GT_RM = tf.concat([GT_R0, GT_R45, GT_R90, GT_R135], axis=-1)

        # R
        RF_1 = slim.conv2d(Pred_R, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        RF0 = slim.conv2d(RF_1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        RFD = self.R_RDBs(RF0)
        RFGF1 = slim.conv2d(RFD, self.PG, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        RFGF2 = slim.conv2d(RFGF1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        RIHR1 = slim.conv2d(RFGF2, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        RIHR2 = slim.max_pool2d(RIHR1, [2, 2], padding='SAME')
        RIHR = slim.conv2d(RIHR2, self.Pc_dim, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        Pred_R0 = RIHR[:, :, :, 0]
        Pred_R0 = Pred_R0[:, :, :, np.newaxis]
        Pred_R45 = RIHR[:, :, :, 1]
        Pred_R45 = Pred_R45[:, :, :, np.newaxis]
        Pred_R90 = RIHR[:, :, :, 2]
        Pred_R90 = Pred_R90[:, :, :, np.newaxis]
        Pred_R135 = RIHR[:, :, :, 3]
        Pred_R135 = Pred_R135[:, :, :, np.newaxis]
        Pred_RS0 = (Pred_R0 + Pred_R45 + Pred_R90 + Pred_R135) / 2
        Pred_RS1 = Pred_R0 - Pred_R90
        Pred_RS2 = Pred_R45 - Pred_R135

        PRED_RDOLP = tf.div(tf.sqrt(tf.square(Pred_RS1) + tf.square(Pred_RS2)), (Pred_RS0 + 0.00001))
        PRED_RDOLP = tf.clip_by_value(PRED_RDOLP, 0, 1)
        PRED_RAOP = 1 / 2 * tf.atan2(Pred_RS2, Pred_RS1)
        PRED_RAOP = tf.clip_by_value(PRED_RAOP, - math.pi / 4.0, math.pi / 4.0)
        PRED_RAOP = (PRED_RAOP + math.pi / 4.0) / (math.pi / 2.0)

        GT_RS0 = (GT_R0 + GT_R45 + GT_R90 + GT_R135) / 2
        GT_RS1 = GT_R0 - GT_R90
        GT_RS2 = GT_R45 - GT_R135
        GT_RDOLP = tf.div(tf.sqrt(tf.square(GT_RS1) + tf.square(GT_RS2)), (GT_RS0 + 0.00001))
        GT_RDOLP = tf.clip_by_value(GT_RDOLP, 0, 1)
        GT_RAOP = 1 / 2 * tf.atan2(GT_RS2, GT_RS1)
        GT_RAOP = tf.clip_by_value(GT_RAOP, - math.pi / 4.0, math.pi / 4.0)
        GT_RAOP = (GT_RAOP + math.pi / 4.0) / (math.pi / 2.0)


        R_PSNR = psnr(RIHR, GT_RM)
        Rd_PSNR = psnr(PRED_RDOLP, GT_RDOLP)
        Ra_PSNR = psnr(PRED_RAOP, GT_RAOP)

        # G
        GF_1 = slim.conv2d(Pred_G, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GF0 = slim.conv2d(GF_1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GFD = self.G_RDBs(GF0)

        GFGF1 = slim.conv2d(GFD, self.PG, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GFGF2 = slim.conv2d(GFGF1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GIHR1 = slim.conv2d(GFGF2, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        GIHR2 = slim.max_pool2d(GIHR1, [2, 2], padding='SAME')
        GIHR = slim.conv2d(GIHR2, self.Pc_dim, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        Pred_G0 = GIHR[:, :, :, 0]
        Pred_G0 = Pred_G0[:, :, :, np.newaxis]
        Pred_G45 = GIHR[:, :, :, 1]
        Pred_G45 = Pred_G45[:, :, :, np.newaxis]
        Pred_G90 = GIHR[:, :, :, 2]
        Pred_G90 = Pred_G90[:, :, :, np.newaxis]
        Pred_G135 = GIHR[:, :, :, 3]
        Pred_G135 = Pred_G135[:, :, :, np.newaxis]
        Pred_GS0 = (Pred_G0 + Pred_G45 + Pred_G90 + Pred_G135) / 2
        Pred_GS1 = Pred_G0 - Pred_G90
        Pred_GS2 = Pred_G45 - Pred_G135

        PRED_GDOLP = tf.div(tf.sqrt(tf.square(Pred_GS1) + tf.square(Pred_GS2)), (Pred_GS0 + 0.00001))
        PRED_GDOLP = tf.clip_by_value(PRED_GDOLP, 0, 1)
        PRED_GAOP = 1 / 2 * tf.atan2(Pred_GS2, Pred_GS1)
        PRED_GAOP = tf.clip_by_value(PRED_GAOP, - math.pi / 4.0, math.pi / 4.0)
        PRED_GAOP = (PRED_GAOP + math.pi / 4.0) / (math.pi / 2.0)

        GT_GS0 = (GT_G0 + GT_G45 + GT_G90 + GT_G135) / 2
        GT_GS1 = GT_G0 - GT_G90
        GT_GS2 = GT_G45 - GT_G135
        GT_GDOLP = tf.div(tf.sqrt(tf.square(GT_GS1) + tf.square(GT_GS2)), (GT_GS0 + 0.00001))
        GT_GDOLP = tf.clip_by_value(GT_GDOLP, 0, 1)
        GT_GAOP = 1 / 2 * tf.atan2(GT_GS2, GT_GS1)
        GT_GAOP = tf.clip_by_value(GT_GAOP, - math.pi / 4.0, math.pi / 4.0)
        GT_GAOP = (GT_GAOP + math.pi / 4.0) / (math.pi / 2.0)

        G_PSNR = psnr(GIHR, GT_GM)
        Gd_PSNR = psnr(PRED_GDOLP, GT_GDOLP)
        Ga_PSNR = psnr(PRED_GAOP, GT_GAOP)

        # B
        BF_1 = slim.conv2d(Pred_B, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        BF0 = slim.conv2d(BF_1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        BFD = self.B_RDBs(BF0)

        BFGF1 = slim.conv2d(BFD, self.PG, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        BFGF2 = slim.conv2d(BFGF1, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        BIHR1 = slim.conv2d(BFGF2, self.PG, [3, 3], rate=1, activation_fn=lrelu,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        BIHR2 = slim.max_pool2d(BIHR1, [2, 2], padding='SAME')
        BIHR = slim.conv2d(BIHR2, self.Pc_dim, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())


        Pred_B0 = BIHR[:, :, :, 0]
        Pred_B0 = Pred_B0[:, :, :, np.newaxis]
        Pred_B45 = BIHR[:, :, :, 1]
        Pred_B45 = Pred_B45[:, :, :, np.newaxis]
        Pred_B90 = BIHR[:, :, :, 2]
        Pred_B90 = Pred_B90[:, :, :, np.newaxis]
        Pred_B135 = BIHR[:, :, :, 3]
        Pred_B135 = Pred_B135[:, :, :, np.newaxis]
        Pred_BS0 = (Pred_B0 + Pred_B45 + Pred_B90 + Pred_B135) / 2
        Pred_BS1 = Pred_B0 - Pred_B90
        Pred_BS2 = Pred_B45 - Pred_B135

        PRED_BDOLP = tf.div(tf.sqrt(tf.square(Pred_BS1) + tf.square(Pred_BS2)), (Pred_BS0 + 0.00001))
        PRED_BDOLP = tf.clip_by_value(PRED_BDOLP, 0, 1)
        PRED_BAOP = 1 / 2 * tf.atan2(Pred_BS2, Pred_BS1)
        PRED_BAOP = tf.clip_by_value(PRED_BAOP, - math.pi / 4.0, math.pi / 4.0)
        PRED_BAOP = (PRED_BAOP + math.pi / 4.0) / (math.pi / 2.0)

        GT_BS0 = (GT_B0 + GT_B45 + GT_B90 + GT_B135) / 2
        GT_BS1 = GT_B0 - GT_B90
        GT_BS2 = GT_B45 - GT_B135
        GT_BDOLP = tf.div(tf.sqrt(tf.square(GT_BS1) + tf.square(GT_BS2)), (GT_BS0 + 0.00001))
        GT_BDOLP = tf.clip_by_value(GT_BDOLP, 0, 1)
        GT_BAOP = 1 / 2 * tf.atan2(GT_BS2, GT_BS1)
        GT_BAOP = tf.clip_by_value(GT_BAOP, - math.pi / 4.0, math.pi / 4.0)
        GT_BAOP = (GT_BAOP + math.pi / 4.0) / (math.pi / 2.0)

        B_PSNR = psnr(BIHR, GT_BM)
        Bd_PSNR = psnr(PRED_BDOLP, GT_BDOLP)
        Ba_PSNR = psnr(PRED_BAOP, GT_BAOP)

        PRED_DOLP = tf.concat([PRED_BDOLP, PRED_GDOLP, PRED_RDOLP], axis=-1)
        PRED_AOP = tf.concat([PRED_BAOP, PRED_GAOP, PRED_RAOP], axis=-1)
        GT_DOLP = tf.concat([GT_BDOLP, GT_GDOLP, GT_RDOLP], axis=-1)
        GT_AOP = tf.concat([GT_BAOP, GT_GAOP, GT_RAOP], axis=-1)
        GT_S0 = tf.concat([GT_BS0, GT_GS0, GT_RS0], axis=-1)
        GT_S1 = tf.concat([GT_BS1, GT_GS1, GT_RS1], axis=-1)
        GT_S2 = tf.concat([GT_BS2, GT_GS2, GT_RS2], axis=-1)
        PRED_S0 = tf.concat([Pred_BS0, Pred_GS0, Pred_RS0], axis=-1)
        PRED_S1 = tf.concat([Pred_BS1, Pred_GS1, Pred_RS1], axis=-1)
        PRED_S2 = tf.concat([Pred_BS2, Pred_GS2, Pred_RS2], axis=-1)
        P_PSNR = (B_PSNR + G_PSNR + R_PSNR)/3
        Pd_PSNR  = (Bd_PSNR + Gd_PSNR + Rd_PSNR)/3
        Pa_PSNR = (Ba_PSNR + Ga_PSNR + Ra_PSNR) / 3

        return IHR, BIHR, GIHR, RIHR, GT_BM, GT_GM, GT_RM, PRED_DOLP, PRED_AOP, GT_DOLP, GT_AOP, PRED_S0, PRED_S1, PRED_S2, GT_S0, GT_S1, GT_S2, PSNR, P_PSNR, Pd_PSNR, Pa_PSNR


    def build_model(self, images_shape, labels_shape):
        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        self.labels = tf.placeholder(tf.float32, labels_shape, name='labels')

        self.pred, self.bpred, self.gpred, self.rpred, self.gt_b, self.gt_g, self.gt_r,self.pred_dolp, self.pred_aop, self.gt_dolp, self.gt_aop, self.pred_s0, self.pred_s1, self.pred_s2, self.gt_s0, self.gt_s1, self.gt_s2, self.psnr, self.p_psnr, self.dolp_psnr, self.aop_psnr = self.model()


        self.closs = tf.reduce_mean(tf.abs(self.labels - self.pred)) * 0.2 #0.2~1
        self.b_loss = tf.reduce_mean(tf.abs(self.gt_b - self.bpred))
        self.g_loss = tf.reduce_mean(tf.abs(self.gt_g - self.gpred))
        self.r_loss = tf.reduce_mean(tf.abs(self.gt_r - self.rpred))
        wr = self.r_loss / (self.r_loss + self.g_loss + self.b_loss)
        wg = self.g_loss / (self.r_loss + self.g_loss + self.b_loss)
        wb = self.b_loss / (self.r_loss + self.g_loss + self.b_loss)
        self.p_loss = wr * self.r_loss + wg * self.g_loss + wb * self.b_loss

        self.dolp_loss = tf.reduce_mean(tf.abs(self.gt_dolp - self.pred_dolp))
        self.aop_loss = tf.reduce_mean(tf.abs(self.gt_aop - self.pred_aop)) * 0.2 #0.2~0.5
        wd = self.dolp_loss / (self.aop_loss + self.dolp_loss)
        wa = self.aop_loss / (self.aop_loss + self.dolp_loss)
        self.pad_loss = wd * self.dolp_loss + wa * self.aop_loss

        bdy_true, bdx_true = tf.image.image_gradients(self.gt_b)
        bdy_pred, bdx_pred = tf.image.image_gradients(self.bpred)
        gdy_true, gdx_true = tf.image.image_gradients(self.gt_g)
        gdy_pred, gdx_pred = tf.image.image_gradients(self.gpred)
        rdy_true, rdx_true = tf.image.image_gradients(self.gt_r)
        rdy_pred, rdx_pred = tf.image.image_gradients(self.rpred)
        self.gradrloss = tf.reduce_mean(tf.abs(rdy_pred - rdy_true) + tf.abs(rdx_pred - rdx_true))
        self.gradgloss = tf.reduce_mean(tf.abs(gdy_pred - gdy_true) + tf.abs(gdx_pred - gdx_true))
        self.gradbloss = tf.reduce_mean(tf.abs(bdy_pred - bdy_true) + tf.abs(bdx_pred - bdx_true))

        wgr = self.gradrloss / (self.gradrloss + self.gradgloss + self.gradbloss)
        wgg = self.gradgloss / ( self.gradrloss + self.gradgloss + self.gradbloss)
        wgb = self.gradbloss / ( self.gradrloss + self.gradgloss + self.gradbloss)
        self.gradloss = (wgr * self.gradrloss + wgg * self.gradgloss + wgb * self.gradbloss) * 0.5 #0.1~1

        wc = self.closs / (self.closs + self.p_loss + self.pad_loss + self.gradloss)
        wp = self.p_loss / (self.closs + self.p_loss + self.pad_loss + self.gradloss)
        wpad = self.pad_loss / (self.closs + self.p_loss + self.pad_loss + self.gradloss)
        wgrad = self.gradloss / (self.closs + self.p_loss + self.pad_loss + self.gradloss)

        self.loss = wc * self.closs  + wp * self.p_loss + wpad * self.pad_loss + wgrad * self.gradloss

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('closs', self.closs)
        tf.summary.scalar('p_loss', self.p_loss)
        tf.summary.scalar('pad_loss', self.pad_loss)
        tf.summary.scalar('grad_loss', self.gradloss)
        tf.summary.scalar('PSNR', self.psnr)
        tf.summary.scalar('P_PSNR', self.p_psnr)
        tf.summary.scalar('D_PSNR', self.dolp_psnr)
        tf.summary.scalar('A_PSNR', self.aop_psnr)

        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self, config):

        input_setup(config) 

        train_data_dir, eval_data_dir = get_data_dir(config.checkpoint_dir, config.is_train)
        train_data_num = get_data_num(train_data_dir)
        batch_num = train_data_num // config.batch_size
        eval_data_num = get_data_num(eval_data_dir)
        # print("train_data_num",train_data_num)
        images_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_shape = [None, self.image_size, self.image_size, self.c_dim]
        # Plabels_shape = [None, self.Pimage_size, self.Pimage_size, self.Pc_dim]
        self.build_model(images_shape, labels_shape)

        epoch, counter = self.load(config.checkpoint_dir)
        global_step = tf.Variable(counter, trainable=False)
        learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.lr_decay_steps * batch_num,
                                                   config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        learning_step = optimizer.minimize(self.loss, global_step=global_step)

        tf.global_variables_initializer().run(session=self.sess)

        merged_summary_op = tf.summary.merge_all()
        summary_train_path = os.path.join(config.checkpoint_dir, "train_%s_%s_%s" % (self.D, self.C, self.G))
        summary_eval_path = os.path.join(config.checkpoint_dir, "eval_%s_%s_%s" % (self.D, self.C, self.G))

        summary_writer_train = tf.summary.FileWriter(summary_train_path, self.sess.graph)
        summary_writer_validate = tf.summary.FileWriter(summary_eval_path)

        time_all = time.time()
        print("\nNow Start Training...\n")
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))

        for ep in range(epoch, config.epoch):
            # Run by batch images

            for idx in range(0, batch_num):
                batch_images, batch_labels = get_batch(train_data_dir, train_data_num, config.batch_size)
                eval_batch_images, eval_batch_labels = get_batch(eval_data_dir, eval_data_num, config.batch_size)
                counter += 1
                assert batch_images.shape == batch_labels.shape
                assert eval_batch_images.shape == eval_batch_labels.shape
                _, closs, p_loss,pad_loss, gradloss, loss, lr, psnr, p_psnr, d_psnr, a_psnr, = self.sess.run(
                    [learning_step, self.closs, self.p_loss, self.pad_loss, self.gradloss, self.loss, learning_rate, self.psnr, self.p_psnr,
                     self.dolp_psnr, self.aop_psnr],
                    feed_dict={self.images: batch_images, self.labels: batch_labels})
                eval_loss = self.sess.run(self.loss,
                                          feed_dict={self.images: eval_batch_images,
                                                     self.labels: eval_batch_labels})
                if counter % 10 == 0:
                    print(
                        "Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%d]min, psnr:[%2.2f], p_psnr:[%2.2f], d_psnr:[%2.2f], a_psnr:[%2.2f], train_loss: [%.4f], c_loss: [%.4f], p_loss: [%.4f], pad_loss: [%.4f], grad_loss: [%.4f], eval_loss:[%.4f]" % (
                            ep + 1, idx, batch_num, counter, int((time.time() - time_all) / 60), psnr, p_psnr, d_psnr,
                            a_psnr, loss, closs, p_loss, pad_loss, gradloss, eval_loss))

                if counter % 100 == 0:
                    print(int((time.time() - time_all) / 60))
                    self.save(config.checkpoint_dir, ep + 1, counter)
                    summary_train = self.sess.run(merged_summary_op,
                                                  feed_dict={self.images: batch_images, self.labels: batch_labels})
                    summary_writer_train.add_summary(summary_train, counter)
                    summary_eval = self.sess.run(merged_summary_op,
                                                 feed_dict={self.images: eval_batch_images,
                                                            self.labels: eval_batch_labels})
                    summary_writer_validate.add_summary(summary_eval, counter)
                if counter > 0 and counter == batch_num * config.epoch:
                    print("Congratulation !  Train Finished.")
                    print("Congratulation !  Train Finished.")
                    print("Congratulation !  Train Finished.")
                    return

    def test(self, config):
        print("\nPrepare Testing Data...\n")
        paths = prepare_data(config)  
        data_num = len(paths)

        print("\nNow Start Testing...\n")
        for idx in range(data_num):
            output_name = paths[idx].split("/")[-1].split('.')[0]
            # input_ = imread(paths[idx]) 
            ratio = float(output_name[4:6])
            input_ = cv2.imread(paths[idx], -1)
            # input_ = input_ * ratio
            input_ = input_[np.newaxis, :]  
            images_shape = input_.shape
            labels_shape = input_.shape
            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess)

            self.load(config.checkpoint_dir)
            result, rresult, gresult, bresult = self.sess.run([self.pred, self.rpred, self.gpred, self.bpred],
                                                              feed_dict={self.images: input_ / 255})
            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()

            img = np.squeeze(result)
            # img = 255 * img
            save_rgb(config, img, output_name)

            imgr = np.squeeze(rresult)
            imgr = 255 * imgr
            save_r(config, imgr, output_name + 'r')

            imgg = np.squeeze(gresult)
            imgg = 255 * imgg
            save_g(config, imgg, output_name + 'g')

            imgb = np.squeeze(bresult)
            imgb = 255 * imgb
            save_b(config, imgb, output_name + 'b')
        print("\n All Done ! ")

    def load(self, checkpoint_dir):
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            print(os.path.join(os.getcwd(), ckpt_path))
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            step = int(ckpt_path.split('-')[-1])
            epoch = int(ckpt_path.split('-')[1])
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            epoch = 0
            print("\nCheckpoint Loading Failed! \n")

        return epoch, step

    def save(self, checkpoint_dir, epoch, step):
        model_name = "RDN.model"
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_name = os.path.join(checkpoint_dir, model_name + '-{}'.format(epoch + 1))
        self.saver.save(self.sess,
                        save_name,
                        global_step=step)

