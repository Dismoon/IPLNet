# -*- coding: utf-8 -*-
import cv2
import glob
import os
from utilsrgb import *

import tensorflow as tf
from model import RDN

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", False, "if the train")  
flags.DEFINE_boolean("is_eval",False, "if the evaluation")  
flags.DEFINE_string("train_set_input", "train/input", "name of the train input set")  
flags.DEFINE_string("train_set_label", "train/label", "name of the train label set")  
# flags.DEFINE_string("train_set_Rlabel", "train/rlabel", "name of the train Rlabel set")  
# flags.DEFINE_string("train_set_Glabel", "train/glabel", "name of the train Glabel set")  
# flags.DEFINE_string("train_set_Blabel", "train/blabel", "name of the train Blabel set")  
flags.DEFINE_string("eval_set_input", 'eval/input','eval_set_input')  
flags.DEFINE_string("eval_set_label", 'eval/label','eval_set_label')  


flags.DEFINE_string("output_dir", "test/output", "test output")  
flags.DEFINE_string("test_set", "test/input", "test input")  
flags.DEFINE_integer("image_size", 64, "the height of image input")  
flags.DEFINE_integer("c_dim", 3, "size of channel")  
flags.DEFINE_integer("P_dim", 1, "the size of channel")  
flags.DEFINE_integer("Pc_dim", 4, "the size of channel")  
flags.DEFINE_integer("stride", 64, "the size of stride")  
flags.DEFINE_integer("epoch", 50, "number of epoch")  
flags.DEFINE_integer("batch_size", 32, "the size of batch")  
flags.DEFINE_float("learning_rate", 1e-4, "the learning rate")	
flags.DEFINE_float("lr_decay_steps", 10, "steps of learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.5, "rate of learning rate decay")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of the checkpoint directory")  
flags.DEFINE_integer("D", 4, "D")  
flags.DEFINE_integer("C", 5, "C")  
flags.DEFINE_integer("G", 32, "G")  
flags.DEFINE_integer("PD", 4, "PD")  
flags.DEFINE_integer("PC", 5, "PC") 
flags.DEFINE_integer("PG", 32, "PG")  
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")  

def main(_):
    rdn = RDN(tf.Session(),
              is_train = FLAGS.is_train,
              is_eval = FLAGS.is_eval,
              image_size = FLAGS.image_size,
              # Pimage_size=FLAGS.Pimage_size,
              c_dim = FLAGS.c_dim,
              P_dim = FLAGS.P_dim,
              Pc_dim = FLAGS.Pc_dim,
              batch_size = FLAGS.batch_size,
              D = FLAGS.D,
              C = FLAGS.C,
              G = FLAGS.G,
              PD=FLAGS.PD,
              PC=FLAGS.PC,
              PG=FLAGS.PG,
              kernel_size = FLAGS.kernel_size
              )

    if rdn.is_train:
        rdn.train(FLAGS)    
    else:
        if rdn.is_eval:
            rdn.eval(FLAGS)
        else:
            rdn.test(FLAGS)
if __name__=='__main__':
    tf.app.run()


